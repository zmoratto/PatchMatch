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

typedef Vector4f DispT;

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
float calculate_cost( Vector2f const& a_loc, Vector2f const& disparity,
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

// Propogates Left, Above, and against opposite disparity
//
// Remember a and ab_disparity don't have the same dimensions. A has
// been expanded so we don't have to do an edge extension. The same
// applies to b and ba_disparity.
void evaluate_even_iteration( ImageView<uint8> const& a, ImageView<uint8> const& b,
                              BBox2i const& a_roi, BBox2i const& b_roi,
                              Vector2i const& kernel_size,
                              ImageView<DispT>& ab_disparity,
                              ImageView<DispT>& ba_disparity ) {

  BBox2i b_disp_size = bounding_box(ba_disparity);

  // TODO: This could iterate by pixel accessor using ab_disparity
  for ( size_t j = 0; j < ab_disparity.rows(); j++ ) {
    for ( size_t i = 0; i < ab_disparity.cols(); i++ ) {
      float cost_new;
      DispT d_new = ab_disparity(i,j);
      Vector2f loc(i,j);

      // TODO: This could be cached!
      float curr_cost =
        calculate_cost( loc, d_new, a, b, a_roi, b_roi, kernel_size );

      // Comparing left
      if ( i > 0 ) {
        d_new = ab_disparity(i-i,j);
        cost_new = calculate_cost( loc, d_new, a, b, a_roi, b_roi, kernel_size );
        if ( cost_new < curr_cost ) {
          curr_cost = cost_new;
          ab_disparity(i,j) = d_new;
        }
      }
      // Comparing top
      if ( j > 0 ) {
        d_new = ab_disparity(i,j-1);
        cost_new = calculate_cost( loc, d_new, a, b, a_roi, b_roi, kernel_size );
        if ( cost_new < curr_cost ) {
          curr_cost = cost_new;
          ab_disparity(i,j) = d_new;
        }
      }

      // Comparing against RL
      Vector2f d = subvector(ab_disparity(i,j),0,2);
      if ( b_disp_size.contains( d + loc ) ) {
        d_new = ba_disparity(i+d[0],j+d[1]);
        subvector(d_new,0,2) = -subvector(d_new,0,2);
        cost_new = calculate_cost( loc, d_new, a, b, a_roi, b_roi, kernel_size );
        if ( cost_new < curr_cost ) {
          curr_cost = cost_new;
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
                             ImageView<DispT>& ba_disparity ) {
  BBox2i b_disp_size = bounding_box(ba_disparity);

  // TODO: This could iterate by pixel accessor using ab_disparity
  for ( size_t j = ab_disparity.rows()-1; j < ab_disparity.rows(); j-- ) {
    for ( size_t i = ab_disparity.cols()-1; i < ab_disparity.cols(); i-- ) {
      float cost_new;
      DispT d_new = ab_disparity(i,j);
      Vector2f loc(i,j);

      // TODO: This could be cached!
      float curr_cost =
        calculate_cost( loc, d_new, a, b, a_roi, b_roi, kernel_size );

      // Comparing right
      if ( i < ab_disparity.cols()-1 ) {
        d_new = ab_disparity(i+1,j);
        cost_new = calculate_cost( loc, d_new, a, b, a_roi, b_roi, kernel_size );
        if ( cost_new < curr_cost ) {
          curr_cost = cost_new;
          ab_disparity(i,j) = d_new;
        }
      }
      // Comparing bottom
      if ( j < ab_disparity.rows()-1 ) {
        d_new = ab_disparity(i,j+1);
        cost_new = calculate_cost( loc, d_new, a, b, a_roi, b_roi, kernel_size );
        if ( cost_new < curr_cost ) {
          curr_cost = cost_new;
          ab_disparity(i,j) = d_new;
        }
      }

      // Comparing against RL
      Vector2f d = subvector(ab_disparity(i,j),0,2);
      if ( b_disp_size.contains( d + loc ) ) {
        d_new = ba_disparity(i+d[0],j+d[1]);
        subvector(d_new,0,2) = -subvector(d_new,0,2);
        cost_new = calculate_cost( loc, d_new, a, b, a_roi, b_roi, kernel_size );
        if ( cost_new < curr_cost ) {
          curr_cost = cost_new;
          ab_disparity(i,j) = d_new;
        }
      }

    }
  }
}

// Evaluates new random search
void evaluate_new_search( ImageView<uint8> const& a, ImageView<uint8> const& b,
                          BBox2i const& a_roi, BBox2i const& b_roi,
                          BBox2f const& search_range, Vector2i const& kernel_size, int iteration,
                          boost::variate_generator<boost::rand48, boost::random::uniform_01<> >& random_source,
                          ImageView<DispT>& ab_disparity ) {

  Vector2f search_range_size = search_range.size();
  float scaling_size = 1.0/pow(2.0,iteration);
  search_range_size *= scaling_size;
  Vector2f search_range_size_half = search_range_size / 2.0;

  std::cout << search_range_size_half << std::endl;

  // TODO: This could iterate by pixel accessor using ab_disparity
  for ( size_t j = 0; j < ab_disparity.rows(); j++ ) {
    for ( size_t i = 0; i < ab_disparity.cols(); i++ ) {
      float cost_new;
      DispT d_new = ab_disparity(i,j);
      Vector2f loc(i,j);

      // TODO: This could be cached!
      float curr_cost =
        calculate_cost( loc, d_new, a, b, a_roi, b_roi, kernel_size );

      // Evaluate a new possible disparity from a local random guess
      Vector2f translation =
        clip_to_search_range( subvector(d_new,0,2) + elem_prod(Vector2f( random_source(), random_source()),search_range_size)
                              - search_range_size_half, search_range );
      subvector(d_new,0,2) = translation;
      d_new[2] += random_source()*scaling_size - scaling_size/2;
      d_new[3] += random_source()*scaling_size - scaling_size/2;
      d_new[2] = std::min(1.0f,std::max(0.01f,d_new[2]));
      d_new[3] = std::min(1.0f,std::max(0.01f,d_new[3]));
      cost_new = calculate_cost( loc, d_new, a, b, a_roi, b_roi, kernel_size );
      if ( cost_new < curr_cost ) {
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
  ImageView<DispT> lr_disparity(left_image.cols(),left_image.rows()), rl_disparity(right_image.cols(),right_image.rows());
  boost::rand48 gen(std::rand());
  typedef boost::variate_generator<boost::rand48, boost::random::uniform_01<> > vargen_type;
  BBox2f search_range(Vector2f(-128,-1),Vector2f(0,1)); // inclusive
  Vector2f iteration_search_size = Vector2f(search_range.size())/4.0;
  vargen_type random_source(gen, boost::random::uniform_01<>());
  Vector2f search_range_size = search_range.size();
  BBox2f search_range_rl( -search_range.max(), -search_range.min() );
  Vector2i kernel_size(11,11);

  for (size_t j = 0; j < lr_disparity.rows(); j++ ) {
    for (size_t i = 0; i < lr_disparity.cols(); i++ ) {
      DispT result;
      subvector(result,0,2) = elem_prod(Vector2f(random_source(),random_source()),search_range_size) + search_range.min();
      result[2] = random_source();
      result[3] = random_source();
      lr_disparity(i,j) = result;
    }
  }
  for (size_t j = 0; j < rl_disparity.rows(); j++ ) {
    for (size_t i = 0; i < rl_disparity.cols(); i++ ) {
      DispT result;
      subvector(result,0,2) = elem_prod(Vector2f(random_source(),random_source()),search_range_size) - search_range.max();
      result[2] = random_source();
      result[3] = random_source();
      rl_disparity(i,j) = result;
    }
  }

  write_image("lr_0.tif",lr_disparity);
  write_image("rl_0.tif",rl_disparity);

  ImageView<float> lr_costs( lr_disparity.cols(), lr_disparity.rows() ),
    rl_costs( rl_disparity.cols(), rl_disparity.rows() );
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

  for ( int iteration = 0; iteration < 20; iteration++ ) {
    if ( iteration > 0 ) {
      evaluate_new_search( left_expanded, right_expanded,
                           left_expanded_roi, right_expanded_roi,
                           search_range, kernel_size, iteration,
                           random_source, lr_disparity );
      evaluate_new_search( right_expanded, left_expanded,
                           right_expanded_roi, left_expanded_roi,
                           search_range_rl, kernel_size, iteration,
                           random_source, rl_disparity );
    }
    if ( iteration % 2 ) {
      evaluate_even_iteration( left_expanded, right_expanded,
                               left_expanded_roi, right_expanded_roi,
                               kernel_size,
                               lr_disparity, rl_disparity );
      evaluate_even_iteration( right_expanded, left_expanded,
                               right_expanded_roi, left_expanded_roi,
                               kernel_size,
                               rl_disparity, lr_disparity );
    } else {
      evaluate_odd_iteration( left_expanded, right_expanded,
                              left_expanded_roi, right_expanded_roi,
                              kernel_size,
                              lr_disparity, rl_disparity );
      evaluate_odd_iteration( right_expanded, left_expanded,
                              right_expanded_roi, left_expanded_roi,
                              kernel_size,
                              rl_disparity, lr_disparity );
    }

    std::string index_str = boost::lexical_cast<std::string>(iteration+1);
    write_image("lr_"+index_str+".tif",lr_disparity);
    write_image("rl_"+index_str+".tif",rl_disparity);
  }

  // Write out the final trusted disparity
  ImageView<PixelMask<Vector2f> > final_disparity =
    per_pixel_filter( lr_disparity, CastVec2fFunc() );
  stereo::cross_corr_consistency_check( final_disparity,
                                        rl_disparity, 1.0, true );
  write_image("final_disparity.tif", final_disparity );
}

int main( int argc, char **argv ) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
