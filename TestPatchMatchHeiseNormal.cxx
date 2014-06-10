// These are sloppily taken ideas from the PM-Huber paper
#include <vw/Core.h>
#include <vw/Math.h>
#include <vw/Image.h>
#include <vw/FileIO.h>
#include <vw/Stereo/DisparityMap.h>
#include <vw/Stereo/Correlate.h>
#include <vw/Stereo/CorrelationView.h>
#include <vw/Stereo/PreFilter.h>

#include <asp/Core/InpaintView.h>

#include <gtest/gtest.h>
#include <numeric>
#include <boost/random/linear_congruential.hpp>

#include <PatchMatchSimple.h>
#include <DisparityFromIP.h>
#include <TVMin.h>
#include <ScanlineOpt.h>

namespace vw {
  template<> struct PixelFormatID<Vector2f> { static const PixelFormatEnum value = VW_PIXEL_GENERIC_2_CHANNEL; };
  template<> struct PixelFormatID<Vector4f> { static const PixelFormatEnum value = VW_PIXEL_GENERIC_4_CHANNEL; };
}

#define DISPARITY_SMOOTHNESS_SIGMA 30.0f
#define NORMAL_SMOOTHNESS_SIGMA 0.1f
#define INTENSITY_SIGMA 0.002f
#define NORMAL_MAX 0.25f

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

struct SearchParameters {
  SearchParameters(int cols, int rows) :
    affine(cols, rows), disparity(cols, rows),
    normal(cols, rows) {}
  ImageView<Matrix2x2f> affine; // Cache, that is calculated from normals
  ImageView<Vector2f> disparity; // Pixel offset
  ImageView<Vector2f> normal; // nx and ny components of a unit vector

  void copy( SearchParameters const& other ) {
    affine = vw::copy(other.affine);
    disparity = vw::copy(other.disparity);
    normal = vw::copy(other.normal);
  }
};

// Simple square kernels
float calculate_cost( Vector2i const& a_idx, // a_loc - a_roi.min
                      Vector2i const& ba_tf, // a_roi.min - b_roi.min
                      Vector2f const& disparity,
                      Vector2f const& normal,
                      Matrix2x2f const& affine,
                      double theta, Vector2f const& disparity_smooth, Vector2f const& normal_smooth,
                      ImageView<float> const& a, ImageView<float> const& b,
                      BBox2i const& kernel_roi, double inv_kernel_area ) {
  Vector2f t = -affine * (Vector2f(a_idx + ba_tf) + disparity);

  float result =
    sum_of_pixel_values
    (per_pixel_filter
     (crop( a, kernel_roi + a_idx ),
      crop( transform_no_edge(b,
                              AffineTransform(affine, t)), kernel_roi ),
      AbsDiffFunc<float>() ));

  // calculate cost is a sum so we are going to normalize by kernel size
  result *= (1.0/INTENSITY_SIGMA) * inv_kernel_area;
  // Add the smoothness constraint against disparity values
  result += theta * (1.0/DISPARITY_SMOOTHNESS_SIGMA) * norm_2(disparity - disparity_smooth);
  // Add the smoothness constraint against normal values
  result += theta * (1.0/NORMAL_SMOOTHNESS_SIGMA) * norm_2(normal - normal_smooth);

  return result;
}

void calculate_affine( SearchParameters& params ) {
  for ( int j = 0; j < params.normal.rows(); j++ ) {
    for ( int i = 0; i < params.normal.cols(); i++ ) {
      Vector2f& n = params.normal(i,j);
      Vector3f normal3(n.x(), n.y(), sqrt(1 - n.x()*n.x() - n.y()*n.y()));
      Vector3f axis = cross_prod(Vector3(0,0,1), normal3);
      float angle = acos(normal3.z());
      Matrix2x2f skew(0, -axis.z(), axis.z(), 0);
      Matrix2x2f tensor_prod(axis.x()*axis.x(), axis.x()*axis.y(),
                             axis.x()*axis.y(), axis.y()*axis.y());
      params.affine(i,j) =
        inverse(normal3.z() * identity_matrix(2) +
                sin(angle) * skew + (1 - normal3.z()) * tensor_prod);
    }
  }
}

void write_template( Vector2f const& a_loc, Vector2f const& disparity, Vector2f const& normal,
                     ImageView<float> const& a, ImageView<float> const& b,
                     BBox2i const& a_roi, BBox2i const& b_roi, Vector2i const& kernel_size, std::string const& prefix ) {
  BBox2i kernel_roi( -kernel_size/2, kernel_size/2 + Vector2i(1,1) );

  Vector3f normal3(normal.x(), normal.y(), sqrt(1 - normal.x()*normal.x() - normal.y()*normal.y()));
  Vector3f axis = cross_prod(Vector3(0,0,1), normal3);
  float angle = acos(normal3.z());
  Matrix2x2f skew(0, -axis.z(), axis.z(), 0);
  Matrix2x2f tensor_prod(axis.x()*axis.x(), axis.x()*axis.y(),
                         axis.x()*axis.y(), axis.y()*axis.y());
  Matrix2x2f r = inverse(normal3.z() * identity_matrix(2) +
                         sin(angle) * skew + (1 - normal3.z()) * tensor_prod);
  Vector2f t = -r * (a_loc + disparity - Vector2f(b_roi.min()));

  write_image(prefix+"_kernel_a.tif",
              crop( a, kernel_roi + a_loc - a_roi.min()));
  write_image(prefix+"_kernel_b.tif",
              crop( transform_no_edge(b,
                                      AffineTransform(r, t)), kernel_roi ));
}

// Evaluates current disparity and writes its cost
void evaluate_disparity( ImageView<float> const& a, ImageView<float> const& b,
                         BBox2i const& a_roi, BBox2i const& b_roi,
                         Vector2i const& kernel_size,
                         ImageView<Vector2f> const& smooth_disparity,
                         ImageView<Vector2f> const& smooth_normal,
                         float theta,
                         SearchParameters const& ab,
                         ImageView<float>& ab_cost ) {
  float* output = &ab_cost(0,0);
  Vector2i ba_tf = a_roi.min() - b_roi.min();
  Vector2i idx;
  BBox2i kernel_roi( -kernel_size/2, kernel_size/2 + Vector2i(1,1) );
  double inv_kernel_area = 1.0 / prod(kernel_size);
  for ( idx.y() = 0; idx.y() < ab.disparity.rows(); idx.y()++ ) {
    for ( idx.x() = 0; idx.x() < ab.disparity.cols(); idx.x()++ ) {
      *output =
        calculate_cost( idx - a_roi.min(), ba_tf,
                        ab.disparity(idx.x(), idx.y()),
                        ab.normal(idx.x(), idx.y()), ab.affine(idx.x(), idx.y()),
                        theta, smooth_disparity(idx.x(), idx.y()),
                        smooth_normal(idx.x(), idx.y()),
                        a, b, kernel_roi, inv_kernel_area );
      output++;
    }
  }
}

void keep_lowest_cost( SearchParameters& dest,
                       ImageView<float>& dest_cost,
                       SearchParameters const& src,
                       ImageView<float> const& src_cost ) {
  for ( int j = 0; j < dest.disparity.rows(); j++ ) {
    for ( int i = 0; i < dest.disparity.cols(); i++ ) {
      if ( dest_cost(i,j) > src_cost(i,j) ) {
        dest_cost(i,j) = src_cost(i,j);
        dest.disparity(i,j) = src.disparity(i,j);
        dest.normal(i,j) = src.normal(i,j);
        dest.affine(i,j) = src.affine(i,j);
      }
    }
  }
}

void evaluate_lr_connected( ImageView<float> const& a,
                           ImageView<float> const& b,
                           BBox2i const& a_roi, BBox2i const& b_roi,
                           Vector2i const& kernel_size,
                           ImageView<Vector2f> const& ab_disparity_smooth,
                           ImageView<Vector2f> const& ab_normal_smooth,
                           float theta,
                           SearchParameters const& ba_in,
                           SearchParameters const& ab_in,
                           ImageView<float> const& ab_cost_in,
                           SearchParameters& ab_out,
                           ImageView<float>& ab_cost_out ) {
  typedef boost::variate_generator<boost::rand48, boost::random::uniform_01<> > vargen_type;
  static vargen_type random_source(boost::rand48(0), boost::random::uniform_01<>());

  float cost;
  Vector2f d_new, n_new;
  Matrix2x2f affine_new;
  BBox2i ba_box = bounding_box(ba_in.disparity);
  BBox2i ab_box = bounding_box(ab_in.disparity);
  Vector2i ba_tf = a_roi.min() - b_roi.min();
  BBox2i kernel_roi( -kernel_size/2, kernel_size/2 + Vector2i(1,1) );
  double inv_kernel_area = 1.0 / prod(kernel_size);
  for ( int j = 0; j < ab_out.disparity.rows(); j++ ) {
    for ( int i = 0; i < ab_out.disparity.cols(); i++ ) {
      Vector2f loc(i,j);

      if ( i > 0 ) {
        // Compare left
        d_new = ab_in.disparity(i-1,j);
        n_new = ab_in.normal(i-1,j);
        affine_new = ab_in.affine(i-1,j);
        if ( ba_box.contains(d_new + loc)) {
          cost = calculate_cost(loc - a_roi.min(), ba_tf, d_new, n_new, affine_new,
                                theta, ab_disparity_smooth(i-1,j), ab_normal_smooth(i-1,j),
                                a, b, kernel_roi, inv_kernel_area);
          if (cost < ab_cost_in(i,j)) {
            ab_cost_out(i,j) = cost;
            ab_out.disparity(i,j) = d_new;
            ab_out.normal(i,j) = n_new;
            ab_out.affine(i,j) = affine_new;
          }
        }
      }
      if ( j > 0 ) {
        // Compare up
        d_new = ab_in.disparity(i,j-1);
        n_new = ab_in.normal(i,j-1);
        affine_new = ab_in.affine(i,j-1);
        if ( ba_box.contains(d_new + loc)) {
          cost = calculate_cost(loc - a_roi.min(), ba_tf, d_new, n_new, affine_new,
                                theta, ab_disparity_smooth(i,j-1), ab_normal_smooth(i,j-1),
                                a, b, kernel_roi, inv_kernel_area);
          if (cost < ab_cost_in(i,j)) {
            ab_cost_out(i,j) = cost;
            ab_out.disparity(i,j) = d_new;
            ab_out.normal(i,j) = n_new;
            ab_out.affine(i,j) = affine_new;
          }
        }
      }
      { // Compare with random pixels that can be 20 pixels away
        Vector2i offset(40 * random_source() - 20,
                        40 * random_source() - 20);
        if ( ab_box.contains(loc + offset) ) {
          d_new = ab_in.disparity(i+offset.x(),j+offset.y());
          n_new = ab_in.normal(i+offset.x(),j+offset.y());
          affine_new = ab_in.affine(i+offset.x(),j+offset.y());
          if ( ba_box.contains(d_new + loc)) {
            cost = calculate_cost(loc - a_roi.min(), ba_tf, d_new, n_new, affine_new,
                                  theta,
                                  ab_disparity_smooth(i+offset.x(),j+offset.y()),
                                  ab_normal_smooth(i+offset.x(),j+offset.y()),
                                  a, b, kernel_roi, inv_kernel_area);
            if (cost < ab_cost_in(i,j)) {
              ab_cost_out(i,j) = cost;
              ab_out.disparity(i,j) = d_new;
              ab_out.normal(i,j) = n_new;
              ab_out.affine(i,j) = affine_new;
            }
          }
        }
      }
      {
        // Compare LR alternative
        Vector2f d = ab_in.disparity(i,j);
        d_new = -ba_in.disparity(i+d[0], j+d[1]);
        if ( ba_box.contains(d_new + loc)) {
          n_new = ba_in.normal(i+d[0], j+d[1]);

          Vector3f normal3(n_new.x(), n_new.y(),
                           sqrt(1 - n_new.x()*n_new.x() - n_new.y()*n_new.y()));
          Vector3f axis = cross_prod(Vector3(0,0,1), normal3);
          float angle = acos(normal3.z());
          Matrix2x2f skew(0, -axis.z(), axis.z(), 0);
          Matrix2x2f tensor_prod(axis.x()*axis.x(), axis.x()*axis.y(),
                                 axis.x()*axis.y(), axis.y()*axis.y());
          affine_new =
            inverse(normal3.z() * identity_matrix(2) +
                    sin(angle) * skew + (1 - normal3.z()) * tensor_prod);


          cost = calculate_cost(loc - a_roi.min(), ba_tf, d_new, n_new, affine_new,
                                theta, ab_disparity_smooth(i+d[0],j+d[1]), ab_normal_smooth(i+d[0],j+d[1]),
                                a, b, kernel_roi, inv_kernel_area);
          if (cost < ab_cost_in(i,j)) {
            ab_cost_out(i,j) = cost;
            ab_out.disparity(i,j) = d_new;
            ab_out.normal(i,j) = n_new;
            ab_out.affine(i,j) = affine_new;
          }
        }
      }
    }
  }
}

void evaluate_rl_connected( ImageView<float> const& a,
                           ImageView<float> const& b,
                           BBox2i const& a_roi, BBox2i const& b_roi,
                           Vector2i const& kernel_size,
                           ImageView<Vector2f> const& ab_disparity_smooth,
                           ImageView<Vector2f> const& ab_normal_smooth,
                           float theta,
                           SearchParameters const& ba_in,
                           SearchParameters const& ab_in,
                           ImageView<float> const& ab_cost_in,
                           SearchParameters& ab_out,
                           ImageView<float>& ab_cost_out ) {
  typedef boost::variate_generator<boost::rand48, boost::random::uniform_01<> > vargen_type;
  static vargen_type random_source(boost::rand48(0), boost::random::uniform_01<>());

  float cost;
  Vector2f d_new, n_new;
  Matrix2x2f affine_new;
  BBox2i ba_box = bounding_box(ba_in.disparity);
  BBox2i ab_box = bounding_box(ab_in.disparity);
  Vector2i ba_tf = a_roi.min() - b_roi.min();
  BBox2i kernel_roi( -kernel_size/2, kernel_size/2 + Vector2i(1,1) );
  double inv_kernel_area = 1.0 / prod(kernel_size);
  for ( int j = ab_out.disparity.rows() - 1; j >= 0; j-- ) {
    for ( int i = ab_out.disparity.cols() - 1; i >= 0; i-- ) {
      Vector2f loc(i,j);

      if ( i < ab_in.disparity.cols() - 1) {
        // Compare right
        d_new = ab_in.disparity(i+1,j);
        n_new = ab_in.normal(i+1,j);
        affine_new = ab_in.affine(i+1,j);
        if ( ba_box.contains(d_new + loc)) {
          cost = calculate_cost(loc - a_roi.min(), ba_tf, d_new, n_new, affine_new,
                                theta, ab_disparity_smooth(i+1,j), ab_normal_smooth(i+1,j),
                                a, b, kernel_roi, inv_kernel_area);
          if (cost < ab_cost_in(i,j)) {
            ab_cost_out(i,j) = cost;
            ab_out.disparity(i,j) = d_new;
            ab_out.normal(i,j) = n_new;
            ab_out.affine(i,j) = affine_new;
          }
        }
      }
      if ( j < ab_in.disparity.rows() - 1 ) {
        // Compare lower
        d_new = ab_in.disparity(i,j+1);
        n_new = ab_in.normal(i,j+1);
        affine_new = ab_in.affine(i,j+1);
        if ( ba_box.contains(d_new + loc)) {
          cost = calculate_cost(loc - a_roi.min(), ba_tf, d_new, n_new, affine_new,
                                theta, ab_disparity_smooth(i,j+1), ab_normal_smooth(i,j+1),
                                a, b, kernel_roi, inv_kernel_area);
          if (cost < ab_cost_in(i,j)) {
            ab_cost_out(i,j) = cost;
            ab_out.disparity(i,j) = d_new;
            ab_out.normal(i,j) = n_new;
            ab_out.affine(i,j) = affine_new;
          }
        }
      }
      { // Compare with random pixels that can be 20 pixels away
        Vector2i offset(40 * random_source() - 20,
                        40 * random_source() - 20);
        if ( ab_box.contains(loc + offset) ) {
          d_new = ab_in.disparity(i+offset.x(),j+offset.y());
          n_new = ab_in.normal(i+offset.x(),j+offset.y());
          affine_new = ab_in.affine(i+offset.x(),j+offset.y());
          if ( ba_box.contains(d_new + loc)) {
            cost = calculate_cost(loc - a_roi.min(), ba_tf, d_new, n_new, affine_new,
                                  theta,
                                  ab_disparity_smooth(i+offset.x(),j+offset.y()),
                                  ab_normal_smooth(i+offset.x(),j+offset.y()),
                                  a, b, kernel_roi, inv_kernel_area);
            if (cost < ab_cost_in(i,j)) {
              ab_cost_out(i,j) = cost;
              ab_out.disparity(i,j) = d_new;
              ab_out.normal(i,j) = n_new;
              ab_out.affine(i,j) = affine_new;
            }
          }
        }
      }
      {
        // Compare LR alternative
        Vector2f d = ab_in.disparity(i,j);
        d_new = -ba_in.disparity(i+d[0], j+d[1]);
        if ( ba_box.contains(d_new + loc)) {
          n_new = ba_in.normal(i+d[0], j+d[1]);

          Vector3f normal3(n_new.x(), n_new.y(),
                           sqrt(1 - n_new.x()*n_new.x() - n_new.y()*n_new.y()));
          Vector3f axis = cross_prod(Vector3(0,0,1), normal3);
          float angle = acos(normal3.z());
          Matrix2x2f skew(0, -axis.z(), axis.z(), 0);
          Matrix2x2f tensor_prod(axis.x()*axis.x(), axis.x()*axis.y(),
                                 axis.x()*axis.y(), axis.y()*axis.y());
          affine_new =
            inverse(normal3.z() * identity_matrix(2) +
                    sin(angle) * skew + (1 - normal3.z()) * tensor_prod);


          cost = calculate_cost(loc - a_roi.min(), ba_tf, d_new, n_new, affine_new,
                                theta, ab_disparity_smooth(i+d[0],j+d[1]), ab_normal_smooth(i+d[0],j+d[1]),
                                a, b, kernel_roi, inv_kernel_area);
          if (cost < ab_cost_in(i,j)) {
            ab_cost_out(i,j) = cost;
            ab_out.disparity(i,j) = d_new;
            ab_out.normal(i,j) = n_new;
            ab_out.affine(i,j) = affine_new;
          }
        }
      }
    }
  }
}

TEST( PatchMatchHeise, Basic ) {
  ImageView<PixelGray<float> > left_image_g, right_image_g;
  read_image(left_image_g,"arctic/asp_al-L.crop.8.tif");
  read_image(right_image_g,"arctic/asp_al-R.crop.8.tif");
  //read_image(left_image_g,"../SemiGlobalMatching/data/cones/im2.png");
  //read_image(right_image_g,"../SemiGlobalMatching/data/cones/im6.png");
  stereo::SubtractedMean filter(7.0);
  ImageView<float>
    left_image = filter.filter(pixel_cast<float>(left_image_g)),
    right_image = filter.filter(pixel_cast<float>(right_image_g));

  SearchParameters lr(left_image.cols(), left_image.rows()),
    rl(right_image.cols(), right_image.rows()),
    lr_copy(left_image.cols(), left_image.rows()),
    rl_copy(right_image.cols(), right_image.rows());
  ImageView<Vector2f>
    lr_disparity_smooth(left_image.cols(), left_image.rows()),
    rl_disparity_smooth(right_image.cols(), right_image.rows()),
    lr_normal_smooth(left_image.cols(), left_image.rows()),
    rl_normal_smooth(right_image.cols(), right_image.rows());
  // BBox2f search_range(Vector2f(-70,-25),Vector2f(105,46)); // exclusive
  BBox2f search_range(Vector2f(-70,-10),Vector2f(105,10)); // exclusive
  //BBox2f search_range(Vector2f(-128,-2), Vector2f(2,2));
  BBox2f search_range_rl( -search_range.max(), -search_range.min() );
  Vector2i kernel_size(7, 7);

  // Filling in the disparity guess
  AddDisparityNoise(search_range, search_range,
                    bounding_box(rl.disparity), lr.disparity);
  AddDisparityNoise(search_range_rl, search_range_rl,
                    bounding_box(lr.disparity), rl.disparity);
  AddDisparityNoise(search_range, search_range,
                    bounding_box(rl.disparity), lr_disparity_smooth);
  AddDisparityNoise(search_range_rl, search_range_rl,
                    bounding_box(lr.disparity), rl_disparity_smooth);
  AddDisparityNoise(BBox2f(-NORMAL_MAX, -NORMAL_MAX, 2*NORMAL_MAX, 2*NORMAL_MAX),
                    BBox2f(-Vector2f(.05,.05),
                           Vector2f(.05,.05)),
                    bounding_box(rl.disparity), lr.normal);
  AddDisparityNoise(BBox2f(-NORMAL_MAX, -NORMAL_MAX, 2*NORMAL_MAX, 2*NORMAL_MAX),
                    BBox2f(-Vector2f(.05,.05),
                           Vector2f(.05,.05)),
                    bounding_box(lr.disparity), rl.normal);
  DisparityFromIP("arctic/asp_al-L.crop.8__asp_al-R.crop.8.match", lr_disparity_smooth, false);
  DisparityFromIP("arctic/asp_al-L.crop.8__asp_al-R.crop.8.match", rl_disparity_smooth, true);
  for (int j = 0; j < lr.disparity.rows(); j += 2) {
    for (int i = 0; i < lr.disparity.cols(); i += 2) {
      lr.disparity(i,j) = lr_disparity_smooth(i,j);
    }
  }
  for (int j = 0; j < rl.disparity.rows(); j += 2) {
    for (int i = 0; i < rl.disparity.cols(); i += 2) {
      rl.disparity(i,j) = rl_disparity_smooth(i,j);
    }
  }

  lr_normal_smooth = copy(lr.normal);
  rl_normal_smooth = copy(rl.normal);

  calculate_affine(lr);
  calculate_affine(rl);

  ImageView<float> lr_cost( lr.disparity.cols(), lr.disparity.rows() ),
    rl_cost( rl.disparity.cols(), rl.disparity.rows() ),
    lr_cost_copy(lr.disparity.cols(), lr.disparity.rows()),
    rl_cost_copy(rl.disparity.cols(), rl.disparity.rows());
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

  write_image("0000_lr_input.tif", lr.disparity);
  write_image("0000_rl_input.tif", rl.disparity);
  write_image("0000_lr_input_sm.tif", lr_disparity_smooth);
  write_image("0000_rl_input_sm.tif", rl_disparity_smooth);

  for ( int iteration = 0; iteration < 10; iteration++ ) {
    //float theta = (1. / 10.f) * float(iteration);
    float theta = 0;
    //float theta = 1.f/10.f;
    // if (iteration > 0) {
    //   theta = (1.0f - 1.0f / float(iteration))*(1.0f - 1.0f / float(iteration));
    //     //pow(2.0f,float(iteration-1))/10;
    // }
    // theta = std::max(0.01f, theta);
    std::cout << "Smooth Scalar: " << theta << std::endl;

    {
      Timer timer("\tEvaluate Disparity", InfoMessage);
      // Evaluate the first cost
      evaluate_disparity( left_expanded, right_expanded,
                          left_expanded_roi, right_expanded_roi,
                          kernel_size, lr_disparity_smooth,
                          lr_normal_smooth,
                          theta, lr, lr_cost);
      evaluate_disparity( right_expanded, left_expanded,
                          right_expanded_roi, left_expanded_roi,
                          kernel_size, rl_disparity_smooth,
                          rl_normal_smooth,
                          theta, rl, rl_cost);
      std::cout << "Starting Avg Cost Per Pixel in LR: "
                << std::accumulate(lr_cost.data(),
                                   lr_cost.data() + lr_cost.cols() * lr_cost.rows(),
                                   double(0)) / (lr_cost.cols() * lr_cost.rows())
                << std::endl;

    }

    // Add noise to find lower cost
    {
      lr_copy.copy(lr);
      rl_copy.copy(rl);

      Vector2f search_range_size = search_range.size();
      float scaling_size = 1.0/pow(2,iteration);
      search_range_size *= scaling_size;
      Vector2f search_range_size_half = search_range_size / 2.0;
      Vector2f normal_search_range = scaling_size * Vector2f(NORMAL_MAX,NORMAL_MAX);
      search_range_size_half[0] = std::max(0.2f, search_range_size_half[0]);
      search_range_size_half[1] = std::max(0.2f, search_range_size_half[1]);
      std::cout << search_range_size_half << std::endl;
      {
        Timer timer("\tAddDisparityNoise", InfoMessage);
        AddDisparityNoise(search_range,
                          BBox2f(-search_range_size_half,search_range_size_half),
                          bounding_box(rl.disparity), lr_copy.disparity);
        AddDisparityNoise(search_range_rl,
                          BBox2f(-search_range_size_half,search_range_size_half),
                          bounding_box(lr.disparity), rl_copy.disparity);
        AddDisparityNoise(BBox2f(-NORMAL_MAX, -NORMAL_MAX, 2*NORMAL_MAX, 2*NORMAL_MAX),
                          BBox2f(-normal_search_range,normal_search_range),
                          bounding_box(rl.disparity), lr_copy.normal);
        AddDisparityNoise(BBox2f(-NORMAL_MAX, -NORMAL_MAX, 2*NORMAL_MAX, 2*NORMAL_MAX),
                          BBox2f(-normal_search_range,normal_search_range),
                          bounding_box(lr.disparity), rl_copy.normal);
        calculate_affine(lr_copy);
        calculate_affine(rl_copy);
      }

      {
        Timer timer("\tEvaluate Disparity", InfoMessage);
        evaluate_disparity( left_expanded, right_expanded,
                            left_expanded_roi, right_expanded_roi,
                            kernel_size, lr_disparity_smooth,
                            lr_normal_smooth,
                            theta, lr_copy, lr_cost_copy );
        evaluate_disparity( right_expanded, left_expanded,
                            right_expanded_roi, left_expanded_roi,
                            kernel_size, rl_disparity_smooth,
                            rl_normal_smooth,
                            theta, rl_copy, rl_cost_copy );
      }

      {
        Timer timer("\tKeep Lowest Cost", InfoMessage);
        keep_lowest_cost( lr, lr_cost,
                          lr_copy, lr_cost_copy );
        keep_lowest_cost( rl, rl_cost,
                          rl_copy, rl_cost_copy );
      }
    }

    // Now we must propogate from the neighbors
    {
      Timer timer("\tEvaluate 8 Connected", InfoMessage);
      evaluate_lr_connected(left_expanded, right_expanded,
                           left_expanded_roi, right_expanded_roi,
                           kernel_size, lr_disparity_smooth,
                           lr_normal_smooth,
                           theta, rl, lr, lr_cost,
                           lr, lr_cost);
      evaluate_lr_connected(right_expanded, left_expanded,
                           right_expanded_roi, left_expanded_roi,
                           kernel_size, rl_disparity_smooth,
                           rl_normal_smooth,
                           theta, lr, rl, rl_cost,
                           rl, rl_cost);
      evaluate_rl_connected(left_expanded, right_expanded,
                           left_expanded_roi, right_expanded_roi,
                           kernel_size, lr_disparity_smooth,
                           lr_normal_smooth,
                           theta, rl, lr, lr_cost,
                           lr, lr_cost);
      evaluate_rl_connected(right_expanded, left_expanded,
                           right_expanded_roi, left_expanded_roi,
                           kernel_size, rl_disparity_smooth,
                           rl_normal_smooth,
                           theta, lr, rl, rl_cost,
                           rl, rl_cost);
    }

    // Solve for smooth disparity
    if (0) {
      Timer timer("\tTV Minimization", InfoMessage);
      int rof_iterations = 100;
      if ( iteration == 0 ) {
        rof_iterations = 500;
      }
      imROF(lr.disparity, theta * (1.0/DISPARITY_SMOOTHNESS_SIGMA),
            rof_iterations, lr_disparity_smooth);
      imROF(rl.disparity, theta * (1.0/DISPARITY_SMOOTHNESS_SIGMA),
            rof_iterations, rl_disparity_smooth);
      imROF(lr.normal, theta * (1.0/NORMAL_SMOOTHNESS_SIGMA),
            rof_iterations, lr_normal_smooth);
      imROF(rl.normal, theta * (1.0/NORMAL_SMOOTHNESS_SIGMA),
            rof_iterations, rl_normal_smooth);
    }

    char prefix[5];
    snprintf(prefix, 5, "%04d", iteration);
    {
      Timer timer("\tWrite images", InfoMessage);
      write_image(std::string(prefix) + "_lr_u.tif", lr.disparity);
      write_image(std::string(prefix) + "_lr_n_u.tif", lr.normal);
      //write_image(std::string(prefix) + "_lr_n_v.tif", lr_normal_smooth);
      //write_image(std::string(prefix) + "_lr_v.tif", lr_disparity_smooth);
      write_image(std::string(prefix) + "_rl_u.tif", rl.disparity);
      write_image(std::string(prefix) + "_rl_n_u.tif", rl.normal);
      //write_image(std::string(prefix) + "_rl_n_v.tif", rl_normal_smooth);
      //write_image(std::string(prefix) + "_rl_v.tif", rl_disparity_smooth);
      /*
      write_template(
                     Vector2f(450, 450), lr.disparity(450, 450),
                     lr.normal(450, 450), left_expanded,
                     right_expanded, left_expanded_roi,
                     right_expanded_roi, kernel_size, std::string(prefix) );
      */
    }
    {
      Timer timer("\tFill consistency failures", InfoMessage);
      ImageView<PixelMask<Vector2f> > lr_mask_copy = lr.disparity;
      stereo::cross_corr_consistency_check( lr_mask_copy,
                                            rl.disparity, 2.0, true );
      ImageView<PixelMask<Vector2f> > rl_mask_copy = rl.disparity;
      stereo::cross_corr_consistency_check( rl_mask_copy,
                                            lr.disparity, 2.0, true );
      BlobIndexThreaded lr_holes( invert_mask( lr_mask_copy ), 100000 );
      scanline_fill(lr_holes, left_image, right_image, lr.disparity);
      BlobIndexThreaded rl_holes( invert_mask( rl_mask_copy ), 100000 );
      scanline_fill(rl_holes, right_image, left_image, rl.disparity);
      write_image(std::string(prefix) + "_lr-filled.tif", lr.disparity);
      write_image(std::string(prefix) + "_rl-filled.tif", rl.disparity);
    }
    std::cout << "Summed cost in LR: "
              << std::accumulate(lr_cost.data(),
                                 lr_cost.data() + lr_cost.cols() * lr_cost.rows(),
                                 double(0)) / (lr_cost.cols() * lr_cost.rows())
              << std::endl;
  }

  // Write out the final trusted disparity
  ImageView<PixelMask<Vector2f> > final_disparity = lr.disparity;
  stereo::cross_corr_consistency_check( final_disparity,
                                        rl.disparity, 1.0, true );
  write_image("final_disp_heise-D.tif", final_disparity );
}

TEST(PatchMatchHeise, ShowPivot) {
  Vector2f disparity(20, 20);
  Vector2f tl(-7,-7), tr(7,-7), bl(-7,7), br(7,7);
  Vector2f ct(0,0);
  for ( int n = 0; n < 20; n++ ) {
    float nf = -.7 + n*1.4/20.0;
    Vector3f normal3(nf, nf, sqrt(1 - nf*nf - nf*nf));
    Vector3f axis = cross_prod(Vector3(0,0,1), normal3);
    float angle = acos(normal3.z());
    Matrix2x2f skew(0, -axis.z(), axis.z(), 0);
    Matrix2x2f tensor_prod(axis.x()*axis.x(), axis.x()*axis.y(),
                           axis.x()*axis.y(), axis.y()*axis.y());
    Matrix2x2f r = inverse(normal3.z() * identity_matrix(2) +
                           sin(angle) * skew + (1 - normal3.z()) * tensor_prod);
    Vector2f t = -r * (disparity);
    AffineTransform tx(r, t);
    std::cout << normal3 << std::endl;
    std::cout << "\t" << ct << " " << tx.reverse(ct) << std::endl;
    std::cout << "\t" << tl << " " << tx.reverse(tl) << std::endl;
    std::cout << "\t" << tr << " " << tx.reverse(tr) << std::endl;
    std::cout << "\t" << bl << " " << tx.reverse(bl) << std::endl;
    std::cout << "\t" << br << " " << tx.reverse(br) << std::endl;
  }
}

int main( int argc, char **argv ) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
