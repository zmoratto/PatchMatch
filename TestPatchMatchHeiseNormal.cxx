// These are sloppily taken ideas from the PM-Huber paper
#include <vw/Core.h>
#include <vw/Math.h>
#include <vw/Image.h>
#include <vw/FileIO.h>
#include <vw/stereo/DisparityMap.h>
#include <vw/stereo/Correlate.h>
#include <vw/stereo/CorrelationView.h>
#include <vw/stereo/PreFilter.h>

#include <gtest/gtest.h>
#include <numeric>

#include <PatchMatchSimple.h>
#include <DisparityFromIP.h>
#include <TVMin.h>

namespace vw {
  template<> struct PixelFormatID<Vector2f> { static const PixelFormatEnum value = VW_PIXEL_GENERIC_2_CHANNEL; };
  template<> struct PixelFormatID<Vector4f> { static const PixelFormatEnum value = VW_PIXEL_GENERIC_4_CHANNEL; };
}

#define DISPARITY_SMOOTHNESS_SIGMA 3.0f
#define NORMAL_SMOOTHNESS_SIGMA 0.05f
#define INTENSITY_SIGMA 0.002f

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

// Simple square kernels
float calculate_cost( Vector2f const& a_loc, Vector2f const& disparity, Vector2f const& normal,
                      double theta, Vector2f const& disparity_smooth, Vector2f const& normal_smooth,
                      ImageView<float> const& a, ImageView<float> const& b,
                      BBox2i const& a_roi, BBox2i const& b_roi, Vector2i const& kernel_size ) {
  BBox2i kernel_roi( -kernel_size/2, kernel_size/2 + Vector2i(1,1) );

  Vector3f normal3(normal.x(), normal.y(), sqrt(1 - normal.x()*normal.x() - normal.y()*normal.y()));
  Vector3f axis = cross_prod(Vector3(0,0,1), normal3);
  float angle = acos(normal3.z());
  Matrix2x2f skew(0, -axis.z(), axis.z(), 0);
  Matrix2x2f tensor_prod(axis.x()*axis.x(), axis.x()*axis.y(),
                         axis.x()*axis.y(), axis.y()*axis.y());
  Matrix2x2f r = normal3.z() * identity_matrix(2) +
    sin(angle) * skew + (1 - normal3.z()) * tensor_prod;
  Vector2f t = -r * (a_loc + disparity - Vector2f(b_roi.min()));

  float result =
    sum_of_pixel_values
    (per_pixel_filter
     (crop( a, kernel_roi + a_loc - a_roi.min() ),
      crop( transform_no_edge(b,
                              AffineTransform(r, t)), kernel_roi ),
      AbsDiffFunc<float>() ));

  float inv_kernel_area = 1.0f/float(prod(kernel_size));

  // calculate cost is a sum so we are going to normalize by kernel size
  result *= (1.0/INTENSITY_SIGMA) * inv_kernel_area;
  // Add the smoothness constraint against disparity values
  result += theta * (1.0/DISPARITY_SMOOTHNESS_SIGMA) * norm_2(disparity - disparity_smooth);
  // Add the smoothness constraint against normal values
  result += theta * (1.0/NORMAL_SMOOTHNESS_SIGMA) * norm_2(normal - normal_smooth);

  return result;
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
  Matrix2x2f r = normal3.z() * identity_matrix(2) +
    sin(angle) * skew + (1 - normal3.z()) * tensor_prod;
  Vector2f t = -r * (a_loc + disparity - Vector2f(b_roi.min()));

  write_image(prefix+"_kernel_a.tif", crop( a, kernel_roi + a_loc - a_roi.min() ));
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
                         ImageView<Vector2f> const& ab_disparity,
                         ImageView<Vector2f> const& ab_normal,
                         ImageView<float>& ab_cost ) {
  for ( int j = 0; j < ab_disparity.rows(); j++ ) {
    for ( int i = 0; i < ab_disparity.cols(); i++ ) {
      ab_cost(i,j) =
        calculate_cost( Vector2f(i,j), ab_disparity(i,j), ab_normal(i, j),
                        theta, smooth_disparity(i, j), smooth_normal(i, j),
                        a, b, a_roi, b_roi, kernel_size );
    }
  }
}

void keep_lowest_cost( ImageView<Vector2f>& dest_disp,
                       ImageView<Vector2f>& dest_normal,
                       ImageView<float>& dest_cost,
                       ImageView<Vector2f> const& src_disp,
                       ImageView<Vector2f> const& src_normal,
                       ImageView<float> const& src_cost ) {
  for ( int j = 0; j < dest_disp.rows(); j++ ) {
    for ( int i = 0; i < dest_disp.cols(); i++ ) {
      if ( dest_cost(i,j) > src_cost(i,j) ) {
        dest_cost(i,j) = src_cost(i,j);
        dest_disp(i,j) = src_disp(i,j);
        dest_normal(i,j) = src_normal(i,j);
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
                           ImageView<Vector2f> const& ab_normal_smooth,
                           float theta,
                           ImageView<Vector2f> const& ba_disparity,
                           ImageView<Vector2f> const& ba_normal,
                           ImageView<Vector2f> const& ab_disparity_in,
                           ImageView<Vector2f> const& ab_normal_in,
                           ImageView<float> const& ab_cost_in,
                           ImageView<Vector2f>& ab_disparity_out,
                           ImageView<Vector2f>& ab_normal_out,
                           ImageView<float>& ab_cost_out ) {
  float cost;
  Vector2f d_new, n_new;
  BBox2i ba_box = bounding_box(ba_disparity);
  for ( int j = 0; j < ab_disparity_out.rows(); j++ ) {
    for ( int i = 0; i < ab_disparity_out.cols(); i++ ) {
      Vector2f loc(i,j);

      if ( i > 0 ) {
        // Compare left
        d_new = ab_disparity_in(i-1,j);
        n_new = ab_normal_in(i-1,j);
        if ( ba_box.contains(d_new + loc)) {
          cost = calculate_cost(loc, d_new, n_new,
                                theta, ab_disparity_smooth(i-1,j), ab_normal_smooth(i-1,j),
                                a, b, a_roi, b_roi, kernel_size);
          if (cost < ab_cost_in(i,j)) {
            ab_cost_out(i,j) = cost;
            ab_disparity_out(i,j) = d_new;
            ab_normal_out(i,j) = n_new;
          }
        }
      }
      if ( j > 0 ) {
        // Compare up
        d_new = ab_disparity_in(i,j-1);
        n_new = ab_normal_in(i,j-1);
        if ( ba_box.contains(d_new + loc)) {
          cost = calculate_cost(loc, d_new, n_new,
                                theta, ab_disparity_smooth(i,j-1), ab_normal_smooth(i,j-1),
                                a, b, a_roi, b_roi, kernel_size);
          if (cost < ab_cost_in(i,j)) {
            ab_cost_out(i,j) = cost;
            ab_disparity_out(i,j) = d_new;
            ab_normal_out(i,j) = n_new;
          }
        }
      }
      if ( i > 0 && j > 0 ) {
        // Compare upper left
        d_new = ab_disparity_in(i-1,j-1);
        n_new = ab_normal_in(i-1,j-1);
        if ( ba_box.contains(d_new + loc)) {
          cost = calculate_cost(loc, d_new, n_new,
                                theta, ab_disparity_smooth(i-1,j-1), ab_normal_smooth(i-1,j-1),
                                a, b, a_roi, b_roi, kernel_size);
          if (cost < ab_cost_in(i,j)) {
            ab_cost_out(i,j) = cost;
            ab_disparity_out(i,j) = d_new;
            ab_normal_out(i,j) = n_new;
          }
        }
      }
      if ( i < ab_disparity_in.cols() - 1) {
        // Compare right
        d_new = ab_disparity_in(i+1,j);
        n_new = ab_normal_in(i+1,j);
        if ( ba_box.contains(d_new + loc)) {
          cost = calculate_cost(loc, d_new, n_new,
                                theta, ab_disparity_smooth(i+1,j), ab_normal_smooth(i+1,j),
                                a, b, a_roi, b_roi, kernel_size);
          if (cost < ab_cost_in(i,j)) {
            ab_cost_out(i,j) = cost;
            ab_disparity_out(i,j) = d_new;
            ab_normal_out(i,j) = n_new;
          }
        }
      }
      if ( i < ab_disparity_in.cols() - 1 && j > 0 ) {
        // Compare upper right
        d_new = ab_disparity_in(i+1,j-1);
        n_new = ab_normal_in(i+1,j-1);
        if ( ba_box.contains(d_new + loc)) {
          cost = calculate_cost(loc, d_new, n_new,
                                theta, ab_disparity_smooth(i+1,j-1), ab_normal_smooth(i+1,j-1),
                                a, b, a_roi, b_roi, kernel_size);
          if (cost < ab_cost_in(i,j)) {
            ab_cost_out(i,j) = cost;
            ab_disparity_out(i,j) = d_new;
            ab_normal_out(i,j) = n_new;
          }
        }
      }
      if ( i < ab_disparity_in.cols() - 1 && j < ab_disparity_in.rows() - 1 ) {
        // Compare lower right
        d_new = ab_disparity_in(i+1,j+1);
        n_new = ab_normal_in(i+1,j+1);
        if ( ba_box.contains(d_new + loc)) {
          cost = calculate_cost(loc, d_new, n_new,
                                theta, ab_disparity_smooth(i+1,j+1), ab_normal_smooth(i+1,j+1),
                                a, b, a_roi, b_roi, kernel_size);
          if (cost < ab_cost_in(i,j)) {
            ab_cost_out(i,j) = cost;
            ab_disparity_out(i,j) = d_new;
            ab_normal_out(i,j) = n_new;
          }
        }

      }
      if ( j < ab_disparity_in.rows() - 1 ) {
        // Compare lower
        d_new = ab_disparity_in(i,j+1);
        n_new = ab_normal_in(i,j+1);
        if ( ba_box.contains(d_new + loc)) {
          cost = calculate_cost(loc, d_new, n_new,
                                theta, ab_disparity_smooth(i,j+1), ab_normal_smooth(i,j+1),
                                a, b, a_roi, b_roi, kernel_size);
          if (cost < ab_cost_in(i,j)) {
            ab_cost_out(i,j) = cost;
            ab_disparity_out(i,j) = d_new;
            ab_normal_out(i,j) = n_new;
          }
        }
      }
      if ( i > 0 && j < ab_disparity_in.rows() - 1 ) {
        // Compare lower left
        d_new = ab_disparity_in(i-1,j+1);
        n_new = ab_normal_in(i-1,j+1);
        if ( ba_box.contains(d_new + loc)) {
          cost = calculate_cost(loc, d_new, n_new,
                                theta, ab_disparity_smooth(i-1,j+1), ab_normal_smooth(i-1,j+1),
                                a, b, a_roi, b_roi, kernel_size);
          if (cost < ab_cost_in(i,j)) {
            ab_cost_out(i,j) = cost;
            ab_disparity_out(i,j) = d_new;
            ab_normal_out(i,j) = n_new;
          }
        }
      }
      {
        // Compare LR alternative
        Vector2f d = ab_disparity_in(i,j);
        d_new = -ba_disparity(i+d[0], j+d[1]);
        n_new = -ba_normal(i+d[0], j+d[1]);
        if ( ba_box.contains(d_new + loc)) {
          cost = calculate_cost(loc, d_new, n_new,
                                theta, ab_disparity_smooth(i+d[0],j+d[1]), ab_normal_smooth(i+d[0],j+d[1]),
                                a, b, a_roi, b_roi, kernel_size);
          if (cost < ab_cost_in(i,j)) {
            ab_cost_out(i,j) = cost;
            ab_disparity_out(i,j) = d_new;
            ab_normal_out(i,j) = n_new;
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

  ImageView<Vector2f> lr_disparity(left_image.cols(),left_image.rows()),
    rl_disparity(right_image.cols(),right_image.rows()),
    lr_disparity_copy(left_image.cols(), left_image.rows()),
    rl_disparity_copy(right_image.cols(), right_image.rows()),
    lr_disparity_smooth(left_image.cols(), left_image.rows()),
    rl_disparity_smooth(right_image.cols(), right_image.rows()),
    lr_normal(left_image.cols(), left_image.rows()),
    rl_normal(right_image.cols(), right_image.rows()),
    lr_normal_copy(left_image.cols(), left_image.rows()),
    rl_normal_copy(right_image.cols(), right_image.rows()),
    lr_normal_smooth(left_image.cols(), left_image.rows()),
    rl_normal_smooth(right_image.cols(), right_image.rows());
  // BBox2f search_range(Vector2f(-70,-25),Vector2f(105,46)); // exclusive
  BBox2f search_range(Vector2f(-70,-10),Vector2f(105,10)); // exclusive
  //BBox2f search_range(Vector2f(-128,-2), Vector2f(2,2));
  BBox2f search_range_rl( -search_range.max(), -search_range.min() );
  Vector2i kernel_size(7, 7);

  // Filling in the disparity guess
  fill(lr_normal, Vector2f(0, 0));
  fill(rl_normal, Vector2f(0, 0));
  fill(lr_normal_copy, Vector2f(0, 0));
  fill(rl_normal_copy, Vector2f(0, 0));
  AddDisparityNoise(search_range, search_range,
                    bounding_box(rl_disparity), lr_disparity);
  AddDisparityNoise(search_range_rl, search_range_rl,
                    bounding_box(lr_disparity), rl_disparity);
  DisparityFromIP("arctic/asp_al-L.crop.8__asp_al-R.crop.8.match", lr_disparity, false);
  DisparityFromIP("arctic/asp_al-L.crop.8__asp_al-R.crop.8.match", rl_disparity, true);
  lr_normal_smooth = copy(lr_normal);
  rl_normal_smooth = copy(rl_normal);
  lr_disparity_smooth = copy(lr_disparity);
  rl_disparity_smooth = copy(rl_disparity);

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

  ImageView<float> left_expanded( crop(edge_extend(left_image), left_expanded_roi ) ),
    right_expanded( crop(edge_extend(right_image), right_expanded_roi ) );

  write_image("0000_lr_input.tif", lr_disparity);
  write_image("0000_rl_input.tif", rl_disparity);

  for ( int iteration = 0; iteration < 50; iteration++ ) {
    float theta = (1. / 50.f) * float(iteration+1);
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
                          theta, lr_disparity,
                          lr_normal, lr_cost );
      evaluate_disparity( right_expanded, left_expanded,
                          right_expanded_roi, left_expanded_roi,
                          kernel_size, rl_disparity_smooth,
                          rl_normal_smooth,
                          theta, rl_disparity,
                          rl_normal, rl_cost );
      std::cout << "Starting Avg Cost Per Pixel in LR: "
                << std::accumulate(lr_cost.data(),
                                   lr_cost.data() + lr_cost.cols() * lr_cost.rows(),
                                   double(0)) / (lr_cost.cols() * lr_cost.rows())
                << std::endl;

    }

    // Add noise to find lower cost
    {
      lr_disparity_copy = copy(lr_disparity);
      rl_disparity_copy = copy(rl_disparity);
      lr_normal_copy = copy(lr_normal);
      rl_normal_copy = copy(rl_normal);
      lr_cost_copy = copy(lr_cost);
      rl_cost_copy = copy(rl_cost);

      Vector2f search_range_size = search_range.size();
      float scaling_size = 1.0/float(iteration);
      if ( iteration == 0 ) {
        scaling_size = 1.0;
      }
      search_range_size *= scaling_size;
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
        AddDisparityNoise(BBox2f(-.7, -.7, 1.4, 1.4),
                          BBox2f(-Vector2f(.1,.1),Vector2f(.1,.1)),
                          bounding_box(rl_disparity), lr_normal_copy);
        AddDisparityNoise(BBox2f(-.7, -.7, 1.4, 1.4),
                          BBox2f(-Vector2f(.1,.1),Vector2f(.1,.1)),
                          bounding_box(lr_disparity), rl_normal_copy);

      }

      {
        Timer timer("\tEvaluate Disparity", InfoMessage);
        evaluate_disparity( left_expanded, right_expanded,
                            left_expanded_roi, right_expanded_roi,
                            kernel_size, lr_disparity_smooth,
                            lr_normal_smooth,
                            theta, lr_disparity_copy,
                            lr_normal_copy, lr_cost_copy );
        evaluate_disparity( right_expanded, left_expanded,
                            right_expanded_roi, left_expanded_roi,
                            kernel_size, rl_disparity_smooth,
                            rl_normal_smooth,
                            theta, rl_disparity_copy,
                            rl_normal_copy, rl_cost_copy );
        write_image("guess_lr.tif", lr_disparity_copy);
        write_image("guess_rl.tif", rl_disparity_copy);
      }

      {
        Timer timer("\tKeep Lowest Cost", InfoMessage);
        std::cout << lr_cost_copy(50,50) << std::endl;
        std::cout << lr_cost(50,50) << std::endl;
        keep_lowest_cost( lr_disparity, lr_normal, lr_cost,
                          lr_disparity_copy, lr_normal_copy, lr_cost_copy );
        keep_lowest_cost( rl_disparity, rl_normal, rl_cost,
                          rl_disparity_copy, rl_normal_copy, rl_cost_copy );
        std::cout << lr_cost(50,50) << std::endl;
        write_image("kept_lr.tif", lr_disparity);
        write_image("kept_rl.tif", rl_disparity);

      }
    }

    // Now we must propogate from the neighbors
    {
      Timer timer("\tEvaluate 8 Connected", InfoMessage);
      evaluate_8_connected(left_expanded, right_expanded,
                           left_expanded_roi, right_expanded_roi,
                           kernel_size, lr_disparity_smooth,
                           lr_normal_smooth,
                           theta,
                           rl_disparity, rl_normal, lr_disparity,
                           lr_normal, lr_cost, lr_disparity,
                           lr_normal, lr_cost);
      evaluate_8_connected(right_expanded, left_expanded,
                           right_expanded_roi, left_expanded_roi,
                           kernel_size, rl_disparity_smooth,
                           rl_normal_smooth,
                           theta,
                           lr_disparity, lr_normal, rl_disparity,
                           rl_normal, rl_cost, rl_disparity,
                           rl_normal, rl_cost);
    }

    // Solve for smooth disparity
    {
      Timer timer("\tTV Minimization", InfoMessage);
      int rof_iterations = 5;
      if ( iteration == 0 ) {
        rof_iterations = 20;
      }
      imROF(lr_disparity, theta * theta * (1.0/DISPARITY_SMOOTHNESS_SIGMA),
            rof_iterations, lr_disparity_smooth);
      imROF(rl_disparity, theta * theta * (1.0/DISPARITY_SMOOTHNESS_SIGMA),
            rof_iterations, rl_disparity_smooth);
      imROF(lr_normal, theta * theta * (1.0/NORMAL_SMOOTHNESS_SIGMA),
            rof_iterations, lr_normal_smooth);
      imROF(rl_normal, theta * theta * (1.0/NORMAL_SMOOTHNESS_SIGMA),
            rof_iterations, rl_normal_smooth);
    }
    {
      Timer timer("\tWrite images", InfoMessage);
      char prefix[5];
      snprintf(prefix, 5, "%04d", iteration);
      write_image(std::string(prefix) + "_lr_u.tif", lr_disparity);
      write_image(std::string(prefix) + "_lr_n_u.tif", lr_normal);
      write_image(std::string(prefix) + "_lr_n_v.tif", lr_normal_smooth);
      write_image(std::string(prefix) + "_lr_v.tif", lr_disparity_smooth);
      write_image(std::string(prefix) + "_rl_u.tif", rl_disparity);
      write_image(std::string(prefix) + "_rl_n_u.tif", rl_normal);
      write_image(std::string(prefix) + "_rl_n_v.tif", rl_normal_smooth);
      write_image(std::string(prefix) + "_rl_v.tif", rl_disparity_smooth);
      write_template( 
                      Vector2f(50, 50), lr_disparity(50, 50),
                      lr_normal(50, 50), left_expanded,
                      right_expanded, left_expanded_roi,
                      right_expanded_roi, kernel_size, std::string(prefix) );
    }
    std::cout << "Summed cost in LR: "
              << std::accumulate(lr_cost.data(),
                                 lr_cost.data() + lr_cost.cols() * lr_cost.rows(),
                                 double(0)) / (lr_cost.cols() * lr_cost.rows())
              << std::endl;
  }

  // Write out the final trusted disparity
  ImageView<PixelMask<Vector2f> > final_disparity = lr_disparity;
  stereo::cross_corr_consistency_check( final_disparity,
                                        rl_disparity, 1.0, true );
  write_image("final_disp_heise-D.tif", final_disparity );
}

int main( int argc, char **argv ) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
