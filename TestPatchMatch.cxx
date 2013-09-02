#include <vw/Core.h>
#include <vw/Image.h>
#include <vw/FileIO.h>
#include <vw/Stereo/DisparityMap.h>
#include <vw/Stereo/Correlate.h>
#include <gtest/gtest.h>

#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_01.hpp>

namespace vw {
  template<> struct PixelFormatID<Vector2f> { static const PixelFormatEnum value = VW_PIXEL_GENERIC_2_CHANNEL; };
}

using namespace vw;

// Possibly replace with something that keep resampling when out of bounds?
inline Vector2f
clip_to_search_range( Vector2f& in, BBox2f const& search_range ) {
  if ( in.x() < search_range.min().x() ) in.x() = search_range.min().x();
  if ( in.x() > search_range.max().x() ) in.x() = search_range.max().x();
  if ( in.y() < search_range.min().y() ) in.y() = search_range.min().y();
  if ( in.y() > search_range.max().y() ) in.y() = search_range.max().y();
  return in;
}

template <int KX, int KY>
float calculate_cost( MemoryStridingPixelAccessor<uint8> first,
                      MemoryStridingPixelAccessor<uint8> second ) {
  first.advance(-KX/2,-KY/2);
  second.advance(-KX/2,-KY/2);
  float sum = 0;
  for ( int y = 0; y < KY; y++ ) {
    for ( int x = 0; x < KX; x++ ) {
      sum += abs(*first - *second);
      first.next_col();
      second.next_col();
    }
    first.advance(-KX,1);
    second.advance(-KX,1);
  }
  return sum;
}

template <int KX, int KY>
float calculate_cost( Vector2f const& a_loc, Vector2f const& b_loc,
                      ImageView<uint8> const& a, ImageView<uint8> const& b,
                      BBox2i const& a_roi, BBox2i const& b_roi ) {
  ImageView<uint8>::pixel_accessor aacc = a.origin(), bacc = b.origin();
  aacc.advance( -a_roi.min().x() + a_loc.x(),
                -a_roi.min().y() + a_loc.y() );
  bacc.advance( -b_roi.min().x() + b_loc.x(),
                -b_roi.min().y() + b_loc.y() );
  return calculate_cost<KX,KY>(aacc,bacc);
}

// Propogates Left, Above, and against opposite disparity
//
// Remember a and ab_disparity don't have the same dimensions. A has
// been expanded so we don't have to do an edge extension. The same
// applies to b and ba_disparity.
void evaluate_even_iteration( ImageView<uint8> const& a, ImageView<uint8> const& b,
                              BBox2i const& a_roi, BBox2i const& b_roi,
                              ImageView<Vector2f>& ab_disparity,
                              ImageView<Vector2f>& ba_disparity,
                              BBox2f const& search_range ) {

  BBox2i b_disp_size = bounding_box(ba_disparity);

  // TODO: This could iterate by pixel accessor using ab_disparity
  for ( size_t j = 0; j < ab_disparity.rows(); j++ ) {
    for ( size_t i = 0; i < ab_disparity.cols(); i++ ) {
      float cost_new;
      Vector2f d_new = ab_disparity(i,j);
      Vector2f loc(i,j);

      // TODO: This could be cached!
      float curr_cost =
        calculate_cost<15,15>( loc, loc+d_new, a, b, a_roi, b_roi );

      // Comparing left
      if ( i > 0 ) {
        d_new = ab_disparity(i-i,j);
        cost_new = calculate_cost<15,15>( loc, loc+d_new, a, b, a_roi, b_roi );
        if ( cost_new < curr_cost ) {
          curr_cost = cost_new;
          ab_disparity(i,j) = d_new;
        }
      }
      // Comparing top
      if ( j > 0 ) {
        d_new = ab_disparity(i,j-1);
        cost_new = calculate_cost<15,15>( loc, loc+d_new, a, b, a_roi, b_roi );
        if ( cost_new < curr_cost ) {
          curr_cost = cost_new;
          ab_disparity(i,j) = d_new;
        }
      }

      // Comparing against RL
      Vector2f d = ab_disparity(i,j);
      if ( b_disp_size.contains( d + Vector2f(i,j) ) ) {
        d_new = -ba_disparity(i+d[0],j+d[1]);
        cost_new = calculate_cost<15,15>( loc, loc+d_new, a, b, a_roi, b_roi );
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
                             ImageView<Vector2f>& ab_disparity,
                             ImageView<Vector2f>& ba_disparity,
                             BBox2i const& search_range ) {
  BBox2i b_disp_size = bounding_box(ba_disparity);

  // TODO: This could iterate by pixel accessor using ab_disparity
  for ( size_t j = ab_disparity.rows()-1; j < ab_disparity.rows(); j-- ) {
    for ( size_t i = ab_disparity.cols()-1; i < ab_disparity.cols(); i-- ) {
      float cost_new;
      Vector2f d_new = ab_disparity(i,j);
      Vector2f loc(i,j);

      // TODO: This could be cached!
      float curr_cost =
        calculate_cost<15,15>( loc, loc+d_new, a, b, a_roi, b_roi );

      // Comparing right
      if ( i < ab_disparity.cols()-1 ) {
        d_new = ab_disparity(i+1,j);
        cost_new = calculate_cost<15,15>( loc, loc+d_new, a, b, a_roi, b_roi );
        if ( cost_new < curr_cost ) {
          curr_cost = cost_new;
          ab_disparity(i,j) = d_new;
        }
      }
      // Comparing bottom
      if ( j < ab_disparity.rows()-1 ) {
        d_new = ab_disparity(i,j+1);
        cost_new = calculate_cost<15,15>( loc, loc+d_new, a, b, a_roi, b_roi );
        if ( cost_new < curr_cost ) {
          curr_cost = cost_new;
          ab_disparity(i,j) = d_new;
        }
      }

      // Comparing against RL
      Vector2f d = ab_disparity(i,j);
      if ( b_disp_size.contains( d + Vector2f(i,j) ) ) {
        d_new = -ba_disparity(i+d[0],j+d[1]);
        cost_new = calculate_cost<15,15>( loc, loc+d_new, a, b, a_roi, b_roi );
        if ( cost_new < curr_cost ) {
          curr_cost = cost_new;
          ab_disparity(i,j) = d_new;
        }
      }

    }
  }
}

TEST( PatchMatch, Basic ) {
  ImageView<PixelGray<uint8> > left_image, right_image;
  read_image(left_image,"../SemiGlobalMatching/data/cones/im2.png");
  read_image(right_image,"../SemiGlobalMatching/data/cones/im6.png");

  // This are our disparity guess. The Vector2f represents a window offset
  ImageView<Vector2f> lr_disparity(left_image.cols(),left_image.rows()), rl_disparity(right_image.cols(),right_image.rows());
  boost::rand48 gen(std::rand());
  typedef boost::variate_generator<boost::rand48, boost::random::uniform_01<> > vargen_type;
  BBox2f search_range(Vector2f(-128,-1),Vector2f(0,1)); // inclusive
  Vector2f iteration_search_size = Vector2f(search_range.size())/4.0;
  vargen_type random_source(gen, boost::random::uniform_01<>());
  Vector2f search_range_size = search_range.size();
  BBox2f search_range_rl( -search_range.max(), -search_range.min() );

  for (size_t j = 0; j < lr_disparity.rows(); j++ ) {
    for (size_t i = 0; i < lr_disparity.cols(); i++ ) {
      lr_disparity(i,j) = elem_prod(Vector2f(random_source(),random_source()),search_range_size) + search_range.min();
    }
  }
  for (size_t j = 0; j < rl_disparity.rows(); j++ ) {
    for (size_t i = 0; i < rl_disparity.cols(); i++ ) {
      rl_disparity(i,j) = elem_prod(Vector2f(random_source(),random_source()),search_range_size) - search_range.max();
    }
  }

  write_image("lr_0.tif",lr_disparity);
  write_image("rl_0.tif",rl_disparity);

  ImageView<float> lr_costs( lr_disparity.cols(), lr_disparity.rows() ),
    rl_costs( rl_disparity.cols(), rl_disparity.rows() );
  BBox2i left_expanded_roi = bounding_box( left_image );
  BBox2i right_expanded_roi = bounding_box( right_image );
  left_expanded_roi.expand(7); // Half kernel size
  right_expanded_roi.expand(7);
  left_expanded_roi.min() -= search_range.max(); // Search range
  left_expanded_roi.max() -= search_range.min();
  right_expanded_roi.min() += search_range.min();
  right_expanded_roi.max() += search_range.max();
  ImageView<uint8> left_expanded( crop(edge_extend(left_image), left_expanded_roi ) ),
    right_expanded( crop(edge_extend(right_image), right_expanded_roi ) );


  // // Apply new search
  // d_new =
  //   clip_to_search_range(lr_disparity(i,j) + Vector2f(horizontal_search_noise(),vertical_search_noise()),search_range);;
  // cost_new = calculate_cost<15,15>( loc, loc+d_new, left_expanded, right_expanded,
  //                                   left_expanded_roi, right_expanded_roi );
  // if ( cost_new < curr_cost ) {
  //   curr_cost = cost_new;
  //   lr_disparity(i,j) = d_new;
  // }

  evaluate_even_iteration( left_expanded, right_expanded,
                           left_expanded_roi, right_expanded_roi,
                           lr_disparity, rl_disparity, search_range );
  evaluate_even_iteration( right_expanded, left_expanded,
                           right_expanded_roi, left_expanded_roi,
                           rl_disparity, lr_disparity, search_range_rl );

  write_image("lr_1.tif",lr_disparity);
  write_image("rl_1.tif",rl_disparity);

  evaluate_odd_iteration( left_expanded, right_expanded,
                          left_expanded_roi, right_expanded_roi,
                          lr_disparity, rl_disparity, search_range );
  evaluate_odd_iteration( right_expanded, left_expanded,
                          right_expanded_roi, left_expanded_roi,
                          rl_disparity, lr_disparity, search_range_rl );

  write_image("lr_2.tif",lr_disparity);
  write_image("rl_2.tif",rl_disparity);

  iteration_search_size /= 2;

  // Write out the final trusted disparity
  ImageView<PixelMask<Vector2f> > final_disparity =
    pixel_cast<PixelMask<Vector2f> >( lr_disparity );
  stereo::cross_corr_consistency_check( final_disparity,
                                        rl_disparity, 1.0, true );
  write_image("final_disparity.tif", final_disparity );
}

int main( int argc, char **argv ) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
