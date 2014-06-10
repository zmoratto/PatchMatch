#include <ScanlineOpt.h>

#include <vw/Image/Interpolation.h>

using namespace vw;

#define PENALTY1 (15.f/255.f)
#define PENALTY2 (100.f/255.f)

template <int MAX_DISP>
void scanline_optimization(std::vector<float> a,
                           std::vector<float> b,
                           std::vector<int> disp) {

  VW_ASSERT(MAX_DISP == b.size() - a.size(),
            LogicErr() << "They don't match");
  // Calculate all comparisons
  std::vector<float> costs(a.size() * MAX_DISP);
  for (int i = 0; i < a.size(); i++ ) {
    for (int d = 0; d < MAX_DISP; d++ ) {
      costs[i * MAX_DISP + d] =
        fabs(a[i] - b[i+d]);
    }
  }

  // Perform dynamic propogation
  std::vector<float>
    forward_propogation(a.size() * MAX_DISP),
    backward_propogation(a.size() * MAX_DISP);
  // Calculate forward propogation values
  memcpy(&forward_propogation[0],
         &costs[0], a.size() * MAX_DISP * sizeof(float));
  for (int i = 1; i < a.size(); i++ ) {
    for (int d = 0; d < MAX_DISP; d++ ) {
      float e_smooth = std::numeric_limits<float>::max();
      for (int d_p = 0; d_p < MAX_DISP; d_p++ ) {
        float& comp_element = forward_propogation[(i-1)*MAX_DISP + d_p];
        if (d_p - d == 0) {
          // No penality
          e_smooth = std::min(e_smooth,comp_element);
        } else {
          e_smooth = std::min(e_smooth,comp_element+PENALTY1);
        }
      }
      forward_propogation[i*MAX_DISP +  d] += e_smooth;
    }
    // Find the min of prior
    float min_of_prior =
      *std::min_element(&forward_propogation[(i-1)*MAX_DISP],
                        &forward_propogation[i*MAX_DISP]);
    // Normalize current propogations by previous min
    for (int d = 0; d < MAX_DISP; d++ ) {
      forward_propogation[i*MAX_DISP + d] /= min_of_prior;
    }
  }

  // Calculate backward propogation values
  memcpy(&backward_propogation[0],
         &costs[0], a.size() * MAX_DISP * sizeof(float));
  for (int i = a.size() - 2; i >= 0; i-- ) {
    for (int d = 0; d < MAX_DISP; d++ ) {
      float e_smooth = std::numeric_limits<float>::max();
      for (int d_p = 0; d_p < MAX_DISP; d_p++ ) {
        float& comp_element = backward_propogation[(i+1)*MAX_DISP + d_p];
        if (d_p - d == 0) {
          // No penality
          e_smooth = std::min(e_smooth,comp_element);
        } else {
          e_smooth = std::min(e_smooth,comp_element+PENALTY1);
        }
      }
      backward_propogation[i*MAX_DISP +  d] += e_smooth;
    }
    // Find the min of prior
    float min_of_prior =
      *std::min_element(&backward_propogation[(i+1)*MAX_DISP],
                        &backward_propogation[(i+2)*MAX_DISP]);
    // Normalize current propogations by previous min
    for (int d = 0; d < MAX_DISP; d++ ) {
      backward_propogation[i*MAX_DISP + d] /= min_of_prior;
    }
  }

  // Calculate sum
  for (int i = 0; i < a.size()*MAX_DISP; i++ ) {
    forward_propogation[i] += backward_propogation[i];
  }

  // Write minimum to disp
  for (int i = 0; i < a.size(); i++ ) {
    disp[i] =
      std::distance(&forward_propogation[i*MAX_DISP],
                    std::min_element(&forward_propogation[i*MAX_DISP],
                                     &forward_propogation[(i+1)*MAX_DISP]));
  }
}

void scanline_fill( BlobIndexThreaded const& blobs_in_a,
                    vw::ImageView<float> const& a,
                    vw::ImageView<float> const& b,
                    vw::ImageView<vw::Vector2f> & disparity ) {
  InterpolationView<EdgeExtensionView<ImageView<float>, ConstantEdgeExtension>, BilinearInterpolation> b_interp = interpolate(b);

  // iterate through blobs
  for (BlobIndexThreaded::const_blob_iterator blob =
         blobs_in_a.begin(); blob != blobs_in_a.end(); blob++ ) {
    Vector2i row_absolute = blob->min();
    // iterate through rows
    for (int row_r = 0; row_r < blob->num_rows(); row_r++) {
      // iterate through row segments
      std::list<int32>::const_iterator
        start = blob->start(row_r).begin(),
        start_last = blob->start(row_r).end(),
        stop = blob->end(row_r).begin();
      while (start != start_last) {
        if ( row_absolute.x() + *start <= 0 ||
             row_absolute.x() + *stop >= a.cols() ) {
          std::cout << "scanline can't be processed: " << row_absolute + Vector2i(*start,0) << " - " << row_absolute + Vector2i(*stop,0) << std::endl;
        } else {
          Vector2f front_disp =
            disparity(row_absolute.x() + *start - 1, row_absolute.y());
          Vector2f back_disp =
            disparity(row_absolute.x() + *stop, row_absolute.y());
          int a_length = *stop - *start;
          int b_length = a_length + 10;
          Vector2f front_b_outside =
            Vector2f(row_absolute + Vector2i(*start-1,0)) + front_disp;
          Vector2f back_b_outside =
            Vector2f(row_absolute + Vector2i(*stop,0)) + back_disp;
          Vector2f b_sample_unit =
            (back_b_outside - front_b_outside) / float(a_length + 1);
          if ( norm_2(b_sample_unit) < .1 ) {
            std::cout << "Crazy is happening: " << b_sample_unit << std::endl;
          }
          Vector2f disp_offset_at_idx_0 =
            (back_disp - front_disp) / float(a_length + 1) + front_disp;

          // These numbers should be the same
          //std::cout << back_b_outside << std::endl;
          //std::cout << Vector2f(row_absolute + Vector2i(*start,0)) + disp_offset_at_idx_0 + a_length * b_sample_unit << std::endl;

          // Resample the imagery to linear arrays
          std::vector<float> a_sampled(a_length);
          for (int i = 0; i < a_length; i++ ) {
            a_sampled[i] = a(row_absolute.x() + *start + i, row_absolute.y());
          }
          std::vector<float> b_sampled(b_length);
          for (int i = 0; i < b_length; i++ ) {
            Vector2f sample_location = Vector2f(row_absolute + Vector2i(*start,0)) + disp_offset_at_idx_0 + float(i-5) * b_sample_unit;
            b_sampled[i] = b_interp(sample_location.x(), sample_location.y());
          }

          // Perform scan line optimization
          std::vector<int> a_disparity(a_length);
          scanline_optimization<10>(a_sampled, b_sampled, a_disparity);

          // Write the result back to the disparity image
          for (int i = 0; i < a_length; i++ ) {
            Vector2f prior_disparity =
              float(i+1) * (back_disp - front_disp) / float(a_length + 1) + front_disp;
            disparity(row_absolute.x() + *start + i, row_absolute.y()) =
              prior_disparity + b_sample_unit * (-5 + a_disparity[i]);
            if ( !BBox2f(Vector2f(-70,-10),Vector2f(105,10)).contains(
                                                                      disparity(row_absolute.x() + *start + i, row_absolute.y()) ))
              throw;
          }
          /*
          std::cout << "Scanline: " << row_absolute + Vector2i(*start,0) << " - " << row_absolute + Vector2i(*stop,0) << std::endl;
          std::cout << " d_qnt: " << disp_quantum  << std::endl;
          std::cout << " a len: " << a_length << " b len: " << b_length  << std::endl;
          std::cout << " b location: " << front_b << " " << back_b << std::endl;
          */
        }
        start++;
        stop++;
      }
      row_absolute.y()++;
    }
  }
}
