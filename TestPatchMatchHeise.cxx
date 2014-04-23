// These are sloppily taken ideas from the PM-Huber paper
#include <vw/Core.h>
#include <vw/Math.h>
#include <vw/Image.h>
#include <vw/FileIO.h>
#include <vw/stereo/DisparityMap.h>
#include <vw/stereo/Correlate.h>
#include <vw/stereo/CorrelationView.h>

#include <gtest/gtest.h>
#include <numeric>

#include <PatchMatchSimple.h>
#include <DisparityFromIP.h>
#include <TVMin.h>

namespace vw {
  template<> struct PixelFormatID<Vector2f> { static const PixelFormatEnum value = VW_PIXEL_GENERIC_2_CHANNEL; };
  template<> struct PixelFormatID<Vector4f> { static const PixelFormatEnum value = VW_PIXEL_GENERIC_4_CHANNEL; };
}

using namespace vw;

template <class ImageT, class TransformT>
TransformView<InterpolationView<ImageT, BilinearInterpolation>, TransformT>
inline transform_no_edge( ImageViewBase<ImageT> const& v,
                          TransformT const& transform_func ) {
  return TransformView<InterpolationView<ImageT, BilinearInterpolation>, TransformT>( InterpolationView<ImageT, BilinearInterpolation>( v.impl() ), transform_func );
}

// To avoid casting higher for uint8 subtraction
template <class PixelT>
struct AbsDiffFunc : public vw::ReturnFixedType<PixelT> {
  inline PixelT operator()( PixelT const& a, PixelT const& b ) const {
    return fabs( a - b );
  }
};

// Casting function
struct CastVec2fFunc : public vw::ReturnFixedType<PixelMask<Vector2f> > {
  inline PixelMask<Vector2f> operator()( Vector4f const& a ) const {
    return subvector(a,0,2);
  }
};

// Simple square kernels
float calculate_cost( Vector2f const& a_loc, Vector2f const& disparity,
                      ImageView<float> const& a, ImageView<float> const& b,
                      BBox2i const& a_roi, BBox2i const& b_roi, Vector2i const& kernel_size ) {
  BBox2i kernel_roi( -kernel_size/2, kernel_size/2 + Vector2i(1,1) );

  float result =
    sum_of_pixel_values
    (per_pixel_filter
     (crop( a, kernel_roi + a_loc - a_roi.min() ),
      crop( transform_no_edge(b, TranslateTransform(-(a_loc.x() + disparity[0] - float(b_roi.min().x())),
                                                    -(a_loc.y() + disparity[1] - float(b_roi.min().y())))),
            kernel_roi ), AbsDiffFunc<float>() ));
  return result;
}

// Evaluates current disparity and writes its cost
void evaluate_disparity( ImageView<float> const& a, ImageView<float> const& b,
                         BBox2i const& a_roi, BBox2i const& b_roi,
                         Vector2i const& kernel_size,
                         ImageView<Vector2f> const& smooth_disparity,
                         float smooth_scalar,
                         ImageView<Vector2f>& ab_disparity,
                         ImageView<float>& ab_cost ) {
  for ( int j = 0; j < ab_disparity.rows(); j++ ) {
    for ( int i = 0; i < ab_disparity.cols(); i++ ) {
      ab_cost(i,j) =
        calculate_cost( Vector2f(i,j),
                        ab_disparity(i,j),
                        a, b, a_roi, b_roi, kernel_size );
      ab_cost(i,j) += smooth_scalar * norm_2_sqr(ab_disparity(i,j) - smooth_disparity(i,j));
    }
  }
}

void keep_lowest_cost( ImageView<Vector2f>& dest_disp,
                       ImageView<float>& dest_cost,
                       ImageView<Vector2f> const& src_disp,
                       ImageView<float> const& src_cost ) {
  for ( int j = 0; j < dest_disp.rows(); j++ ) {
    for ( int i = 0; i < dest_disp.cols(); i++ ) {
      if ( dest_cost(i,j) > src_cost(i,j) ) {
        dest_cost(i,j) = src_cost(i,j);
        dest_disp(i,j) = src_disp(i,j);
      }
    }
  }
}

// Propogates from the 3x3 neighbor hood
void evaluate_8_connected( ImageView<float> const& a,
                           ImageView<float> const& b,
                           BBox2i const& a_roi, BBox2i const& b_roi,
                           Vector2i const& kernel_size,
                           ImageView<Vector2f> const& ab_disparity_smooth,
                           float smooth_scalar,
                           ImageView<Vector2f> const& ba_disparity,
                           ImageView<Vector2f> const& ab_disparity_in,
                           ImageView<float> const& ab_cost_in,
                           ImageView<Vector2f>& ab_disparity_out,
                           ImageView<float>& ab_cost_out ) {
  float cost;
  Vector2f d_new;
  BBox2i ba_box = bounding_box(ba_disparity);
  for ( int j = 0; j < ab_disparity_out.rows(); j++ ) {
    for ( int i = 0; i < ab_disparity_out.cols(); i++ ) {
      Vector2f loc(i,j);

      if ( i > 0 ) {
        // Compare left
        d_new = ab_disparity_in(i-1,j);
        if ( ba_box.contains(d_new + loc)) {
          cost = calculate_cost(loc, d_new, a, b, a_roi, b_roi, kernel_size);
          cost += smooth_scalar * norm_2_sqr(d_new - ab_disparity_smooth(i,j));
          if (cost < ab_cost_in(i,j)) {
            ab_cost_out(i,j) = cost;
            ab_disparity_out(i,j) = d_new;
          }
        }
      }
      if ( j > 0 ) {
        // Compare up
        d_new = ab_disparity_in(i,j-1);
        if ( ba_box.contains(d_new + loc)) {
          cost = calculate_cost(loc, d_new, a, b, a_roi, b_roi, kernel_size);
          cost += smooth_scalar * norm_2_sqr(d_new - ab_disparity_smooth(i,j));
          if (cost < ab_cost_in(i,j)) {
            ab_cost_out(i,j) = cost;
            ab_disparity_out(i,j) = d_new;
          }
        }
      }
      if ( i > 0 && j > 0 ) {
        // Compare upper left
        d_new = ab_disparity_in(i-1,j-1);
        if ( ba_box.contains(d_new + loc)) {
          cost = calculate_cost(loc, d_new, a, b, a_roi, b_roi, kernel_size);
          cost += smooth_scalar * norm_2_sqr(d_new - ab_disparity_smooth(i,j));
          if (cost < ab_cost_in(i,j)) {
            ab_cost_out(i,j) = cost;
            ab_disparity_out(i,j) = d_new;
          }
        }
      }
      if ( i < ab_disparity_in.cols() - 1) {
        // Compare right
        d_new = ab_disparity_in(i+1,j);
        if ( ba_box.contains(d_new + loc)) {
          cost = calculate_cost(loc, d_new, a, b, a_roi, b_roi, kernel_size);
          cost += smooth_scalar * norm_2_sqr(d_new - ab_disparity_smooth(i,j));
          if (cost < ab_cost_in(i,j)) {
            ab_cost_out(i,j) = cost;
            ab_disparity_out(i,j) = d_new;
          }
        }
      }
      if ( i < ab_disparity_in.cols() - 1 && j > 0 ) {
        // Compare upper right
        d_new = ab_disparity_in(i+1,j-1);
        if ( ba_box.contains(d_new + loc)) {
          cost = calculate_cost(loc, d_new, a, b, a_roi, b_roi, kernel_size);
          cost += smooth_scalar * norm_2_sqr(d_new - ab_disparity_smooth(i,j));
          if (cost < ab_cost_in(i,j)) {
            ab_cost_out(i,j) = cost;
            ab_disparity_out(i,j) = d_new;
          }
        }
      }
      if ( i < ab_disparity_in.cols() - 1 && j < ab_disparity_in.rows() - 1 ) {
        // Compare lower right
        d_new = ab_disparity_in(i+1,j+1);
        if ( ba_box.contains(d_new + loc)) {
          cost = calculate_cost(loc, d_new, a, b, a_roi, b_roi, kernel_size);
          cost += smooth_scalar * norm_2_sqr(d_new - ab_disparity_smooth(i,j));
          if (cost < ab_cost_in(i,j)) {
            ab_cost_out(i,j) = cost;
            ab_disparity_out(i,j) = d_new;
          }
        }

      }
      if ( j < ab_disparity_in.rows() - 1 ) {
        // Compare lower
        d_new = ab_disparity_in(i,j+1);
        if ( ba_box.contains(d_new + loc)) {
          cost = calculate_cost(loc, d_new, a, b, a_roi, b_roi, kernel_size);
          cost += smooth_scalar * norm_2_sqr(d_new - ab_disparity_smooth(i,j));
          if (cost < ab_cost_in(i,j)) {
            ab_cost_out(i,j) = cost;
            ab_disparity_out(i,j) = d_new;
          }
        }
      }
      if ( i > 0 && j < ab_disparity_in.rows() - 1 ) {
        // Compare lower left
        d_new = ab_disparity_in(i-1,j+1);
        if ( ba_box.contains(d_new + loc)) {
          cost = calculate_cost(loc, d_new, a, b, a_roi, b_roi, kernel_size);
          cost += smooth_scalar * norm_2_sqr(d_new - ab_disparity_smooth(i,j));
          if (cost < ab_cost_in(i,j)) {
            ab_cost_out(i,j) = cost;
            ab_disparity_out(i,j) = d_new;
          }
        }
      }
      {
        // Compare LR alternative
        Vector2f d = ab_disparity_in(i,j);
        d_new = -ba_disparity(i+d[0], j+d[1]);
        if ( ba_box.contains(d_new + loc)) {
          cost = calculate_cost(loc, d_new, a, b, a_roi, b_roi, kernel_size);
          cost += smooth_scalar * norm_2_sqr(d_new - ab_disparity_smooth(i,j));
          if (cost < ab_cost_in(i,j)) {
            ab_cost_out(i,j) = cost;
            ab_disparity_out(i,j) = d_new;
          }
        }
      }
    }
  }
}

TEST( PatchMatchHeise, Basic ) {
  ImageView<float > left_image, right_image;
  read_image(left_image,"arctic/asp_al-L.crop.8.tif");
  read_image(right_image,"arctic/asp_al-R.crop.8.tif");
  //read_image(left_image,"../SemiGlobalMatching/data/cones/im2.png");
  //read_image(right_image,"../SemiGlobalMatching/data/cones/im6.png");

  ImageView<Vector2f> lr_disparity(left_image.cols(),left_image.rows()),
    rl_disparity(right_image.cols(),right_image.rows()),
    lr_disparity_copy(left_image.cols(), left_image.rows()),
    rl_disparity_copy(right_image.cols(), right_image.rows()),
    lr_disparity_smooth(left_image.cols(), left_image.rows()),
    rl_disparity_smooth(right_image.cols(), right_image.rows());
  BBox2f search_range(Vector2f(-70,-25),Vector2f(105,46)); // exclusive
  //BBox2f search_range(Vector2f(-70,-10),Vector2f(105,10)); // exclusive
  //BBox2f search_range(Vector2f(-128,-2), Vector2f(2,2));
  BBox2f search_range_rl( -search_range.max(), -search_range.min() );
  Vector2i kernel_size(15,15);

  // Filling in the disparity guess
  DisparityFromIP("arctic/asp_al-L.crop.8__asp_al-R.crop.8.match", lr_disparity, false);
  DisparityFromIP("arctic/asp_al-L.crop.8__asp_al-R.crop.8.match", rl_disparity, true);

  ImageView<float> lr_cost( lr_disparity.cols(), lr_disparity.rows() ),
    rl_cost( rl_disparity.cols(), rl_disparity.rows() ),
    lr_cost_copy(lr_disparity.cols(), lr_disparity.rows()),
    rl_cost_copy(rl_disparity.cols(), rl_disparity.rows());
  BBox2i left_expanded_roi = bounding_box( left_image );
  BBox2i right_expanded_roi = bounding_box( right_image );
  left_expanded_roi.min() -= kernel_size/2;      // Expand by kernel size
  left_expanded_roi.max() += kernel_size/2;
  right_expanded_roi.min() -= kernel_size/2;
  right_expanded_roi.max() += kernel_size/2;
  left_expanded_roi.expand( BilinearInterpolation::pixel_buffer );
  right_expanded_roi.expand( BilinearInterpolation::pixel_buffer );
  const float lambda1 = 1.0;
  const float lambda2 = 1.0 / 25.0;

  ImageView<float> left_expanded( crop(edge_extend(left_image), left_expanded_roi ) ),
    right_expanded( crop(edge_extend(right_image), right_expanded_roi ) );

  write_image("0000_lr_input.tif", lr_disparity);
  write_image("0000_rl_input.tif", rl_disparity);

  for ( int iteration = 0; iteration < 50; iteration++ ) {
    float smooth_scalar = 0.0000001f;
    if (iteration > 0) {
      smooth_scalar = (1.0f - 1.0f / float(iteration))*(1.0f - 1.0f / float(iteration));
        //pow(2.0f,float(iteration-1))/10;
    }
    //smooth_scalar = 0.25;
    std::cout << "Smooth Scalar: " << smooth_scalar << std::endl;

    {
      Timer timer("\tEvaluate Disparity", InfoMessage);
      // Evaluate the first cost
      evaluate_disparity( left_expanded, right_expanded,
                          left_expanded_roi, right_expanded_roi,
                          kernel_size, lr_disparity_smooth, lambda1 * smooth_scalar, lr_disparity, lr_cost );
      evaluate_disparity( right_expanded, left_expanded,
                          right_expanded_roi, left_expanded_roi,
                          kernel_size, rl_disparity_smooth, lambda1 * smooth_scalar, rl_disparity, rl_cost );
      std::cout << "Starting Summed cost in LR: "
                << std::accumulate(lr_cost.data(),
                                   lr_cost.data() + lr_cost.cols() * lr_cost.rows(),
                                   double(0))
                << std::endl;

    }

    // Add noise to find lower cost
    {
      lr_disparity_copy = copy(lr_disparity);
      rl_disparity_copy = copy(rl_disparity);
      lr_cost_copy = copy(lr_cost);
      rl_cost_copy = copy(rl_cost);

      //      Vector2f search_range_size = search_range.size();
      Vector2f search_range_size(20,20);
      float scaling_size = 1.0/pow(2.0,iteration);
      //search_range_size *= scaling_size;
      Vector2f search_range_size_half = search_range_size / 2.0;
      search_range_size_half[0] = std::max(0.25f, search_range_size_half[0]);
      search_range_size_half[1] = std::max(0.25f, search_range_size_half[1]);
      std::cout << search_range_size_half << std::endl;
      {
        Timer timer("\tAddDisparityNoise", InfoMessage);
        AddDisparityNoise(search_range,
                          BBox2f(-search_range_size_half,search_range_size_half),
                          bounding_box(rl_disparity), lr_disparity_copy);
        AddDisparityNoise(search_range_rl,
                          BBox2f(-search_range_size_half,search_range_size_half),
                          bounding_box(lr_disparity), rl_disparity_copy);
      }

      {
        Timer timer("\tEvaluate Disparity", InfoMessage);
        evaluate_disparity( left_expanded, right_expanded,
                            left_expanded_roi, right_expanded_roi,
                            kernel_size, lr_disparity_smooth, lambda1 * smooth_scalar, lr_disparity_copy, lr_cost_copy );
        evaluate_disparity( right_expanded, left_expanded,
                            right_expanded_roi, left_expanded_roi,
                            kernel_size, rl_disparity_smooth, lambda1 * smooth_scalar, rl_disparity_copy, rl_cost_copy );
      }

      {
        Timer timer("\tKeep Lowest Cost", InfoMessage);
        keep_lowest_cost( lr_disparity, lr_cost,
                          lr_disparity_copy, lr_cost_copy );
        keep_lowest_cost( rl_disparity, rl_cost,
                          rl_disparity_copy, rl_cost_copy );
      }
    }

    // Now we must propogate from the neighbors
    {
      Timer timer("\tEvaluate 8 Connected", InfoMessage);
      lr_disparity_copy = copy(lr_disparity);
      rl_disparity_copy = copy(rl_disparity);
      lr_cost_copy = copy(lr_cost);
      rl_cost_copy = copy(rl_cost);
      evaluate_8_connected(left_expanded, right_expanded,
                           left_expanded_roi, right_expanded_roi,
                           kernel_size, lr_disparity_smooth,
                           lambda1 * smooth_scalar,
                           rl_disparity, lr_disparity,
                           lr_cost, lr_disparity,
                           lr_cost);
      evaluate_8_connected(right_expanded, left_expanded,
                           right_expanded_roi, left_expanded_roi,
                           kernel_size, rl_disparity_smooth,
                           lambda1 * smooth_scalar,
                           lr_disparity, rl_disparity,
                           rl_cost, rl_disparity,
                           rl_cost);
      //lr_disparity = copy(lr_disparity_copy);
      //rl_disparity = copy(rl_disparity_copy);
    }

    // Solve for smooth disparity
    {
      Timer timer("\tTV Minimization", InfoMessage);
      imROF(lr_disparity, smooth_scalar * lambda2, 10, lr_disparity_smooth);
      imROF(rl_disparity, smooth_scalar * lambda2, 10, rl_disparity_smooth);
    }
    {
      Timer timer("\tWrite images", InfoMessage);
      char prefix[5];
      snprintf(prefix, 5, "%04d", iteration);
      write_image(std::string(prefix) + "_lr_u.tif", lr_disparity);
      write_image(std::string(prefix) + "_lr_v.tif", lr_disparity_smooth);
      write_image(std::string(prefix) + "_rl_u.tif", rl_disparity);
      write_image(std::string(prefix) + "_rl_v.tif", rl_disparity_smooth);
    }
    std::cout << "Summed cost in LR: "
              << std::accumulate(lr_cost.data(),
                                 lr_cost.data() + lr_cost.cols() * lr_cost.rows(),
                                 double(0))
              << std::endl;
  }

  // Write out the final trusted disparity
  ImageView<PixelMask<Vector2f> > final_disparity = lr_disparity;
  stereo::cross_corr_consistency_check( final_disparity,
                                        rl_disparity, 1.0, true );
  write_image("final_disp_heise-D.tif", final_disparity );
}

TEST( PatchMatchHeise, DISABLED_VerifyCostLower ) {
  ImageView<float > left_image, right_image;
  read_image(left_image,"arctic/asp_al-L.crop.8.tif");
  read_image(right_image,"arctic/asp_al-R.crop.8.tif");

  BBox2f search_range(Vector2f(-70,-25),Vector2f(105,46)); // exclusive
  Vector2i kernel_size(27,27);

  ImageView<float> lr_cost( left_image.cols(), left_image.rows() );
  BBox2i left_expanded_roi = bounding_box( left_image );
  BBox2i right_expanded_roi = bounding_box( right_image );
  left_expanded_roi.min() -= kernel_size/2;      // Expand by kernel size
  left_expanded_roi.max() += kernel_size/2;
  right_expanded_roi.min() -= kernel_size/2;
  right_expanded_roi.max() += kernel_size/2;
  left_expanded_roi.expand( BilinearInterpolation::pixel_buffer );
  right_expanded_roi.expand( BilinearInterpolation::pixel_buffer );

  ImageView<float> left_expanded( crop(edge_extend(left_image), left_expanded_roi ) ),
    right_expanded( crop(edge_extend(right_image), right_expanded_roi ) );

  ImageView<Vector2f> patch_disparity, asp_disparity;
  ImageView<PixelMask<Vector2f> > asp_mask_disparity;
  read_image(patch_disparity, "arctic/patch_match_result.tif");
  read_image(asp_mask_disparity, "arctic/asp_result.tif");

  asp_disparity = copy(patch_disparity);
  for (int j = 0; j < asp_disparity.rows(); j++ ) {
    for (int i = 0; i < asp_disparity.cols(); i++ ) {
      if (is_valid(asp_mask_disparity(i,j))) {
        asp_disparity(i,j) = asp_mask_disparity(i,j).child();
      }
    }
  }

  for ( int k = 27; k >= 5; k -= 2 ) {
    kernel_size = Vector2i(k,k);
    std::cout << kernel_size << std::endl;
    evaluate_disparity( left_expanded, right_expanded,
                        left_expanded_roi, right_expanded_roi,
                        kernel_size, patch_disparity, 0, patch_disparity, lr_cost );
    std::cout << "PatchMatch cost: "
              << std::accumulate(lr_cost.data(),
                                 lr_cost.data() + lr_cost.rows() * lr_cost.cols(),
                                 double(0)) << std::endl;
    evaluate_disparity( left_expanded, right_expanded,
                        left_expanded_roi, right_expanded_roi,
                        kernel_size, patch_disparity, 0, asp_disparity, lr_cost );

  std::cout << "ASP cost: "
            << std::accumulate(lr_cost.data(),
                               lr_cost.data() + lr_cost.rows() * lr_cost.cols(),
                               double(0)) << std::endl;

  }
  write_image("patch_augment.tif", patch_disparity);
  write_image("asp_augment.tif", asp_disparity);
}

template <class ImageT>
void block_write_image( const std::string &filename,
                        vw::ImageViewBase<ImageT> const& image,
                        vw::ProgressCallback const& progress_callback = vw::ProgressCallback::dummy_instance() ) {
  boost::scoped_ptr<vw::DiskImageResourceGDAL> rsrc
    (new vw::DiskImageResourceGDAL(filename, image.impl().format(), Vector2i(256,256)));
  vw::block_write_image( *rsrc, image.impl(), progress_callback );
}

TEST( PatchMatchHeise, BruteForceSearch ) {
  ImageView<float > left_image, right_image;
  read_image(left_image,"arctic/asp_al-L.crop.8.tif");
  read_image(right_image,"arctic/asp_al-R.crop.8.tif");
  //read_image(left_image,"../SemiGlobalMatching/data/cones/im2.png");
  //read_image(right_image,"../SemiGlobalMatching/data/cones/im6.png");

  //BBox2i search_range(Vector2f(-128,-2),Vector2f(2,2)); // exclusive
  BBox2f search_range(Vector2f(-70,-25),Vector2f(105,46)); // exclusive
  block_write_image( "blog_article/brute_force_arctic-D.tif",
                     stereo::correlate(left_image, right_image,
                                       stereo::NullOperation(),
                                       search_range, Vector2i(3, 3)),
                     TerminalProgressCallback("test","BruteForce:") );
}

int main( int argc, char **argv ) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
