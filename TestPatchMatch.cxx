#include <vw/Core.h>
#include <vw/Image.h>
#include <vw/FileIO.h>
#include <vw/Stereo/DisparityMap.h>
#include <vw/Stereo/Correlate.h>
#include <gtest/gtest.h>

#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/lexical_cast.hpp>

namespace vw {
  template<> struct PixelFormatID<Vector2f> { static const PixelFormatEnum value = VW_PIXEL_GENERIC_2_CHANNEL; };
  template<> struct PixelFormatID<Vector4f> { static const PixelFormatEnum value = VW_PIXEL_GENERIC_4_CHANNEL; };
}

using namespace vw;

typedef Vector2f DispT;

// Possibly replace with something that keep resampling when out of bounds?
inline Vector2f
clip_to_search_range( Vector2f in, BBox2f const& search_range ) {
  if ( in.x() < search_range.min().x() ) in.x() = search_range.min().x();
  if ( in.x() > search_range.max().x() ) in.x() = search_range.max().x();
  if ( in.y() < search_range.min().y() ) in.y() = search_range.min().y();
  if ( in.y() > search_range.max().y() ) in.y() = search_range.max().y();
  return in;
}

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
    return abs( a - b );
  }
};

// Casting function
struct CastVec2fFunc : public vw::ReturnFixedType<PixelMask<Vector2f> > {
  inline PixelMask<Vector2f> operator()( Vector4f const& a ) const {
    return subvector(a,0,2);
  }
};

// Simple square kernels
float calculate_cost_d( Vector2f const& a_loc, Vector2f const& disparity,
                        ImageView<uint8> const& a, ImageView<uint8> const& b,
                        BBox2i const& a_roi, BBox2i const& b_roi, Vector2i const& kernel_size ) {
  BBox2i kernel_roi( -kernel_size/2, kernel_size/2 + Vector2i(1,1) );

  float result =
    sum_of_pixel_values
    (per_pixel_filter
     (crop( a, kernel_roi + a_loc - a_roi.min() ),
      crop( transform_no_edge(b, TranslateTransform(-(a_loc.x() + disparity[0] - float(b_roi.min().x())),
                                                    -(a_loc.y() + disparity[1] - float(b_roi.min().y())))),
            kernel_roi ), AbsDiffFunc<uint8>() ));
  return result;
}

// Parallelogram version
float calculate_cost( Vector2f const& a_loc, Vector4f const& disparity,
                      ImageView<uint8> const& a, ImageView<uint8> const& b,
                      BBox2i const& a_roi, BBox2i const& b_roi, Vector2i const& kernel_size ) {
  BBox2i kernel_roi( -kernel_size/2, kernel_size/2 + Vector2i(1,1) );

  Matrix2x2f scaling(1/disparity[2],0,0,1/disparity[3]);
  Vector2f b_offset = a_loc + subvector(disparity,0,2) - Vector2f(b_roi.min());

  float result =
    sum_of_pixel_values
    (per_pixel_filter
     (crop( a, kernel_roi + a_loc - a_roi.min() ),
      crop( transform_no_edge(b, AffineTransform( scaling, -b_offset ) ),
            kernel_roi ), AbsDiffFunc<uint8>() ));
  return result;
}

// Floating point square cost functor that uses adaptive support weights
float  calculate_cost( Vector2f const& a_loc, Vector2f const& disparity,
                       ImageView<uint8> const& a, ImageView<uint8> const& b,
                       BBox2i const& a_roi, BBox2i const& b_roi, Vector2i const& kernel_size ) {
  BBox2i kernel_roi( -kernel_size/2, kernel_size/2 + Vector2i(1,1) );

  ImageView<float> left_kernel
    = crop(a, kernel_roi + a_loc - a_roi.min() );
  ImageView<float> right_kernel
    = crop( transform_no_edge(b,
                              TranslateTransform(-(a_loc.x() + disparity[0] - float(b_roi.min().x())),
                                                 -(a_loc.y() + disparity[1] - float(b_roi.min().y())))),
            kernel_roi );

  // Calculate support weights for left and right
  ImageView<float>
    weight(kernel_size.x(),kernel_size.y());

  Vector2f center_index = kernel_size/2;
  float left_color = left_kernel(center_index[0],center_index[1]),
    right_color = right_kernel(center_index[0],center_index[1]);
  float kernel_diag = norm_2(kernel_size);
  float sum = 0;
  for ( int j = 0; j < kernel_size.y(); j++ ) {
    for ( int i = 0; i < kernel_size.x(); i++ ) {
      float dist = norm_2( Vector2f(i,j) - center_index )/kernel_diag;
      //float dist = 0;
      float lcdist = fabs( left_kernel(i,j) - left_color );
      float rcdist = fabs( right_kernel(i,j) - right_color );
      sum += weight(i,j) = exp(-lcdist/14 - dist) * exp(-rcdist/14 - dist);
    }
  }

  return sum_of_pixel_values( weight * per_pixel_filter(left_kernel,right_kernel,AbsDiffFunc<float>())) / sum;
}

// Propogates Left, Above, and against opposite disparity
//
// Remember a and ab_disparity don't have the same dimensions. A has
// been expanded so we don't have to do an edge extension. The same
// applies to b and ba_disparity.
void evaluate_even_iteration( ImageView<uint8> const& a, ImageView<uint8> const& b,
                              BBox2i const& a_roi, BBox2i const& b_roi,
                              Vector2i const& kernel_size,
                              ImageView<DispT>& ab_disparity,
                              ImageView<DispT>& ba_disparity,
                              ImageView<float>& ab_cost ) {

  BBox2i b_disp_size = bounding_box(ba_disparity);

  // TODO: This could iterate by pixel accessor using ab_disparity
  for ( int j = 0; j < ab_disparity.rows(); j++ ) {
    // Compare to the left
    for ( int i = 1; i < ab_disparity.cols(); i++ ) {
      Vector2f loc(i,j);

      DispT d_new = ab_disparity(i-i,j);
      float cost_new = calculate_cost( loc, d_new, a, b, a_roi, b_roi, kernel_size );
      if ( cost_new < ab_cost(i,j) ) {
        ab_cost(i,j) = cost_new;
        ab_disparity(i,j) = d_new;
      }
    }

    // Compare to the top
    if ( j > 0 ) {
      for ( int i = 0; i < ab_disparity.cols(); i++ ) {
        Vector2f loc(i,j);
        DispT d_new = ab_disparity(i,j-1);
        float cost_new = calculate_cost( loc, d_new, a, b, a_roi, b_roi, kernel_size );
        if ( cost_new < ab_cost(i,j) ) {
          ab_cost(i,j) = cost_new;
          ab_disparity(i,j) = d_new;
        }
      }
    }

    // Comparing against RL
    for ( int i = 0; i < ab_disparity.cols(); i++ ) {
      Vector2f loc(i,j);

      Vector2f d = subvector(ab_disparity(i,j),0,2);
      if ( b_disp_size.contains( d + loc ) ) {
        DispT d_new = ba_disparity(i+d[0],j+d[1]);
        subvector(d_new,0,2) = -subvector(d_new,0,2);
        float cost_new = calculate_cost( loc, d_new, a, b, a_roi, b_roi, kernel_size );
        if ( cost_new < ab_cost(i,j) ) {
          ab_cost(i,j) = cost_new;
          ab_disparity(i,j) = d_new;
        }
      }
    }
  }
}

// Propogates Right, Below, and against opposite disparity
void evaluate_odd_iteration( ImageView<uint8> const& a, ImageView<uint8> const& b,
                             BBox2i const& a_roi, BBox2i const& b_roi,
                             Vector2i const& kernel_size,
                             ImageView<DispT>& ab_disparity,
                             ImageView<DispT>& ba_disparity,
                             ImageView<float>& ab_cost ) {
  BBox2i b_disp_size = bounding_box(ba_disparity);

  // TODO: This could iterate by pixel accessor using ab_disparity
  for ( int j = ab_disparity.rows()-1; j >= 0; j-- ) {
    // Comparing right
    for ( int i = ab_disparity.cols()-2; i >= 0; i-- ) {
      DispT d_new = ab_disparity(i+1,j);
      float cost_new = calculate_cost( Vector2f(i,j), d_new, a, b, a_roi, b_roi, kernel_size );
      if ( cost_new < ab_cost(i,j) ) {
        ab_cost(i,j) = cost_new;
        ab_disparity(i,j) = d_new;
      }
    }

    // Comparing bottom
    if ( j < ab_disparity.rows()-1 ) {
      for ( int i = ab_disparity.cols()-1; i >= 0; i-- ) {
        DispT d_new = ab_disparity(i,j+1);
        float cost_new = calculate_cost( Vector2f(i,j), d_new, a, b, a_roi, b_roi, kernel_size );
        if ( cost_new < ab_cost(i,j) ) {
          ab_cost(i,j) = cost_new;
          ab_disparity(i,j) = d_new;
        }
      }
    }

    for ( int i = ab_disparity.cols()-1; i >= 0; i-- ) {
      // Comparing against RL
      Vector2f d = subvector(ab_disparity(i,j),0,2);
      if ( b_disp_size.contains( d + Vector2f(i,j) ) ) {
        DispT d_new = ba_disparity(i+d[0],j+d[1]);
        subvector(d_new,0,2) = -subvector(d_new,0,2);
        float cost_new = calculate_cost( Vector2f(i,j), d_new, a, b, a_roi, b_roi, kernel_size );
        if ( cost_new < ab_cost(i,j) ) {
          ab_cost(i,j) = cost_new;
          ab_disparity(i,j) = d_new;
        }
      }

    }
  }
}

// Evaluates current disparity and writes its cost
void evaluate_disparity( ImageView<uint8> const& a, ImageView<uint8> const& b,
                         BBox2i const& a_roi, BBox2i const& b_roi,
                         Vector2i const& kernel_size,
                         ImageView<DispT>& ab_disparity,
                         ImageView<float>& ab_cost ) {
  for ( int j = 0; j < ab_disparity.rows(); j++ ) {
    for ( int i = 0; i < ab_disparity.cols(); i++ ) {
      ab_cost(i,j) =
        calculate_cost( Vector2f(i,j),
                        ab_disparity(i,j),
                        a, b, a_roi, b_roi, kernel_size );
    }
  }
}

// Evaluates new random search
void evaluate_new_search( ImageView<uint8> const& a, ImageView<uint8> const& b,
                          BBox2i const& a_roi, BBox2i const& b_roi,
                          BBox2f const& search_range, Vector2i const& kernel_size, int iteration,
                          boost::variate_generator<boost::rand48, boost::random::uniform_01<> >& random_source,
                          ImageView<DispT>& ab_disparity,
                          ImageView<float>& ab_cost ) {

  Vector2f search_range_size = search_range.size();
  float scaling_size = 1.0/pow(2.0,iteration);
  search_range_size *= scaling_size;
  Vector2f search_range_size_half = search_range_size / 2.0;
  search_range_size_half[0] = std::max(0.5f, search_range_size_half[0]);
  search_range_size_half[1] = std::max(0.5f, search_range_size_half[1]);

  std::cout << search_range_size_half << std::endl;

  // TODO: This could iterate by pixel accessor using ab_disparity
  for ( int j = 0; j < ab_disparity.rows(); j++ ) {
    for ( int i = 0; i < ab_disparity.cols(); i++ ) {
      DispT d_new = ab_disparity(i,j);
      Vector2f loc(i,j);

      // Evaluate a new possible disparity from a local random guess
      subvector(d_new,0,2) =
        clip_to_search_range( subvector(d_new,0,2) +
                              elem_prod(Vector2f( random_source(), random_source()),
                                        search_range_size)
                              - search_range_size_half, search_range );
      float cost_new = calculate_cost( loc, d_new, a, b, a_roi, b_roi, kernel_size );
      if ( cost_new < ab_cost(i,j) ) {
        ab_cost(i,j) = cost_new;
        ab_disparity(i,j) = d_new;
      }
    }
  }
}

TEST( PatchMatch, Basic ) {
  ImageView<PixelGray<uint8> > left_image, right_image;
  read_image(left_image,"../SemiGlobalMatching/data/cones/im2.png");
  read_image(right_image,"../SemiGlobalMatching/data/cones/im6.png");

  // This are our disparity guess. The Vector2f represents a window offset
  ImageView<DispT> lr_disparity(left_image.cols(),left_image.rows()),
    rl_disparity(right_image.cols(),right_image.rows());
  boost::rand48 gen(std::rand());
  typedef boost::variate_generator<boost::rand48, boost::random::uniform_01<> > vargen_type;
  BBox2f search_range(Vector2f(-128,-1),Vector2f(0,1)); // inclusive
  vargen_type random_source(gen, boost::random::uniform_01<>());
  Vector2f search_range_size = search_range.size();
  BBox2f search_range_rl( -search_range.max(), -search_range.min() );
  Vector2i kernel_size(11,11);

  for (int j = 0; j < lr_disparity.rows(); j++ ) {
    for (int i = 0; i < lr_disparity.cols(); i++ ) {
      DispT result;
      subvector(result,0,2) = elem_prod(Vector2f(random_source(),random_source()),search_range_size) + search_range.min();
      lr_disparity(i,j) = result;
    }
  }
  for (int j = 0; j < rl_disparity.rows(); j++ ) {
    for (int i = 0; i < rl_disparity.cols(); i++ ) {
      DispT result;
      subvector(result,0,2) = elem_prod(Vector2f(random_source(),random_source()),search_range_size) - search_range.max();
      rl_disparity(i,j) = result;
    }
  }

  ImageView<float> lr_cost( lr_disparity.cols(), lr_disparity.rows() ),
    rl_cost( rl_disparity.cols(), rl_disparity.rows() );
  BBox2i left_expanded_roi = bounding_box( left_image );
  BBox2i right_expanded_roi = bounding_box( right_image );
  left_expanded_roi.min() -= kernel_size/2;      // Expand by kernel size
  left_expanded_roi.max() += kernel_size/2;
  right_expanded_roi.min() -= kernel_size/2;
  right_expanded_roi.max() += kernel_size/2;
  left_expanded_roi.min() -= search_range.max(); // Search range
  left_expanded_roi.max() -= search_range.min();
  right_expanded_roi.min() += search_range.min();
  right_expanded_roi.max() += search_range.max();
  left_expanded_roi.expand( BilinearInterpolation::pixel_buffer );
  right_expanded_roi.expand( BilinearInterpolation::pixel_buffer );
  ImageView<uint8> left_expanded( crop(edge_extend(left_image), left_expanded_roi ) ),
    right_expanded( crop(edge_extend(right_image), right_expanded_roi ) );

  // Evaluate the first cost
  evaluate_disparity( left_expanded, right_expanded,
                      left_expanded_roi, right_expanded_roi,
                      kernel_size, lr_disparity, lr_cost );
  evaluate_disparity( right_expanded, left_expanded,
                      right_expanded_roi, left_expanded_roi,
                      kernel_size, rl_disparity, rl_cost );

  for ( int iteration = 0; iteration < 6; iteration++ ) {
    if ( iteration > 0 ) {
      evaluate_new_search( left_expanded, right_expanded,
                           left_expanded_roi, right_expanded_roi,
                           search_range, kernel_size, iteration,
                           random_source, lr_disparity, lr_cost );
      evaluate_new_search( right_expanded, left_expanded,
                           right_expanded_roi, left_expanded_roi,
                           search_range_rl, kernel_size, iteration,
                           random_source, rl_disparity, rl_cost );
    }
    if ( iteration % 2 ) {
      evaluate_even_iteration( left_expanded, right_expanded,
                               left_expanded_roi, right_expanded_roi,
                               kernel_size,
                               lr_disparity, rl_disparity, lr_cost );
      evaluate_even_iteration( right_expanded, left_expanded,
                               right_expanded_roi, left_expanded_roi,
                               kernel_size,
                               rl_disparity, lr_disparity, rl_cost );
    } else {
      evaluate_odd_iteration( left_expanded, right_expanded,
                              left_expanded_roi, right_expanded_roi,
                              kernel_size,
                              lr_disparity, rl_disparity, lr_cost );
      evaluate_odd_iteration( right_expanded, left_expanded,
                              right_expanded_roi, left_expanded_roi,
                              kernel_size,
                              rl_disparity, lr_disparity, rl_cost );
    }

  }

  // Write out the final trusted disparity
  ImageView<PixelMask<Vector2f> > final_disparity = lr_disparity;
  write_image("lr_disparity.tif", ImageView<PixelMask<Vector2f> >(lr_disparity) );
  write_image("rl_disparity.tif", ImageView<PixelMask<Vector2f> >(rl_disparity) );

  //    per_pixel_filter( lr_disparity, CastVec2fFunc() );
  stereo::cross_corr_consistency_check( final_disparity,
                                        rl_disparity, 1.0, true );
  write_image("final_disparity.tif", final_disparity );
}

int main( int argc, char **argv ) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
