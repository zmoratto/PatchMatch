#include <PatchMatch2.h>
#include <vw/Core/Exception.h>
#include <vw/Math/BBox.h>
#include <vw/Math/Vector.h>
#include <vw/Image/ImageView.h>

#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/linear_congruential.hpp>

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

void
stereo::PatchMatchBase::add_uniform_noise(BBox2i const& range_of_noise_to_add, // Inclusive
                                          BBox2i const& max_search_range, // Inclusive
                                          BBox2i const& other_image_bbox, // Exclusive
                                          ImageView<stereo::PatchMatchBase::DispT>& disparity ) const {
  typedef boost::random::uniform_01<float> DistributionT;
  typedef boost::variate_generator<GenT, DistributionT > vargen_type;
  vargen_type random_source(GenT(0), DistributionT());

  BBox2i local_search_range_bounds, local_search_range;
  for (int j = 0; j < disparity.rows(); j++) {
    local_search_range_bounds.min().y() =
      std::max(max_search_range.min().y(),
               other_image_bbox.min().y() - j);
    local_search_range_bounds.max().y() =
      std::min(max_search_range.max().y(),
               other_image_bbox.max().y() - j - 1);
    for (int i = 0; i < disparity.cols(); i++) {
      local_search_range_bounds.min().x() =
        std::max(max_search_range.min().x(),
                 other_image_bbox.min().x() - i);
      local_search_range_bounds.max().x() =
        std::min(max_search_range.max().x(),
                 other_image_bbox.max().x() - i - 1);
      local_search_range = range_of_noise_to_add;
      local_search_range.min() += disparity(i, j);
      local_search_range.max() += disparity(i, j);
      local_search_range.crop(local_search_range_bounds);

#ifdef DEBUG
      DispT d =
        elem_prod(Vector2f(random_source(),random_source()),
                  local_search_range.size()) + local_search_range.min();
      VW_DEBUG_ASSERT(other_image_bbox.contains(Vector2i(i,j) + d),
                      MathErr() << "Modified disparity points outside of other image.");
      disparity(i,j) = d;
#else
      disparity(i,j) =
        elem_prod(Vector2f(random_source(),random_source()),
                  local_search_range.size()) + local_search_range.min();

#endif
    }
  }
}

// Simple square kernels
float stereo::PatchMatchBase::calculate_cost( Vector2i const& a_loc, Vector2i const& disparity,
                                              ImageView<float> const& a, ImageView<float> const& b,
                                              BBox2i const& a_roi, BBox2i const& b_roi,
                                              BBox2i const& kernel_roi) const {
  float result =
    sum_of_pixel_values
    (per_pixel_filter
     (crop( a, kernel_roi + a_loc - a_roi.min() ),
      crop( b, kernel_roi + a_loc + disparity - b_roi.min() ),
      AbsDiffFunc<float>() ));
  return result;
}

// Evaluates current disparity and writes its cost
void stereo::PatchMatchBase::evaluate_disparity( ImageView<float> const& a, ImageView<float> const& b,
                                                 BBox2i const& a_roi, BBox2i const& b_roi,
                                                 ImageView<DispT>& ab_disparity,
                                                 ImageView<float>& ab_cost ) const {
  for ( int j = 0; j < ab_disparity.rows(); j++ ) {
    for ( int i = 0; i < ab_disparity.cols(); i++ ) {
      ab_cost(i,j) =
        calculate_cost( Vector2i(i,j),
                        ab_disparity(i,j),
                        a, b, a_roi, b_roi,
                        m_kernel_roi);
    }
  }
}

void stereo::PatchMatchBase::keep_lowest_cost( ImageView<DispT>& dest_disp,
                                               ImageView<float>& dest_cost,
                                               ImageView<DispT> const& src_disp,
                                               ImageView<float> const& src_cost ) const {
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
void stereo::PatchMatchBase::evaluate_8_connected( ImageView<float> const& a,
                                                   ImageView<float> const& b,
                                                   BBox2i const& a_roi, BBox2i const& b_roi,
                                                   ImageView<DispT> const& ba_disparity,
                                                   BBox2i const& ba_roi,
                                                   ImageView<DispT>& ab_disparity,
                                                   ImageView<float>& ab_cost) const {
  float cost;
  DispT d_new;
  for ( int j = 0; j < ab_disparity.rows(); j++ ) {
    for ( int i = 0; i < ab_disparity.cols(); i++ ) {
      DispT loc(i,j);

      float curr_best_cost = ab_cost(i,j);
      DispT curr_best_d = ab_disparity(i, j);

      if ( i > 0 ) {
        // Compare left
        d_new = ab_disparity(i-1,j);
        if ( ba_roi.contains(d_new + loc) && curr_best_d != d_new) {
          cost = ab_cost(i - 1, j)
            + calculate_cost(loc, d_new, a, b, a_roi, b_roi,
                             m_kernel_roi_left_p)
            - calculate_cost(loc, d_new, a, b, a_roi, b_roi,
                             m_kernel_roi_left_n);
#ifdef DEBUG
          VW_DEBUG_ASSERT(fabs(cost - calculate_cost(loc, d_new, a, b, a_roi, b_roi,
                                                     m_kernel_roi)) < 1e-3,
                          MathErr() << "Bug");
#endif
          if (cost < curr_best_cost) {
            curr_best_cost = cost;
            curr_best_d = d_new;
          }
        }
      }
      if ( j > 0 ) {
        // Compare up
        d_new = ab_disparity(i,j - 1);
        if ( ba_roi.contains(d_new + loc) && curr_best_d != d_new) {
          cost = ab_cost(i, j - 1)
            + calculate_cost(loc, d_new, a, b, a_roi, b_roi, m_kernel_roi_top_p)
            - calculate_cost(loc, d_new, a, b, a_roi, b_roi, m_kernel_roi_top_n);
#ifdef DEBUG
          VW_DEBUG_ASSERT(fabs(cost - calculate_cost(loc, d_new, a, b, a_roi, b_roi,
                                                     m_kernel_roi)) < 1e-3,
                          MathErr() << "Bug");
#endif
          if (cost < curr_best_cost) {
            curr_best_cost = cost;
            curr_best_d = d_new;
          }
        }
      }
      if ( i > 0 && j > 0 ) {
        // Compare upper left
        d_new = ab_disparity(i-1,j-1);
        if ( ba_roi.contains(d_new + loc) && curr_best_d != d_new) {
          cost = calculate_cost(loc, d_new, a, b, a_roi, b_roi, m_kernel_roi);
          if (cost < curr_best_cost) {
            curr_best_cost = cost;
            curr_best_d = d_new;
          }
        }
      }
      if ( i < ab_disparity.cols() - 1) {
        // Compare right
        d_new = ab_disparity(i+1,j);
        if ( ba_roi.contains(d_new + loc) && curr_best_d != d_new) {
          cost = ab_cost(i + 1, j)
            + calculate_cost(loc, d_new, a, b, a_roi, b_roi,
                             m_kernel_roi_right_p)
            - calculate_cost(loc, d_new, a, b, a_roi, b_roi,
                             m_kernel_roi_right_n);
#ifdef DEBUG
          VW_DEBUG_ASSERT(fabs(cost - calculate_cost(loc, d_new, a, b, a_roi, b_roi,
                                                     m_kernel_roi)) < 1e-3,
                          MathErr() << "Bug");
#endif
          if (cost < curr_best_cost) {
            curr_best_cost = cost;
            curr_best_d = d_new;
          }
        }
      }
      if ( i < ab_disparity.cols() - 1 && j > 0 ) {
        // Compare upper right
        d_new = ab_disparity(i+1,j-1);
        if ( ba_roi.contains(d_new + loc) && curr_best_d != d_new) {
          cost = calculate_cost(loc, d_new, a, b, a_roi, b_roi, m_kernel_roi);
          if (cost < curr_best_cost) {
            curr_best_cost = cost;
            curr_best_d = d_new;
          }
        }
      }
      if ( i < ab_disparity.cols() - 1 && j < ab_disparity.rows() - 1 ) {
        // Compare lower right
        d_new = ab_disparity(i+1,j+1);
        if ( ba_roi.contains(d_new + loc) && curr_best_d != d_new) {
          cost = calculate_cost(loc, d_new, a, b, a_roi, b_roi, m_kernel_roi);
          if (cost < curr_best_cost) {
            curr_best_cost = cost;
            curr_best_d = d_new;
          }
        }

      }
      if ( j < ab_disparity.rows() - 1 ) {
        // Compare lower
        d_new = ab_disparity(i,j+1);
        if ( ba_roi.contains(d_new + loc) && curr_best_d != d_new) {
          cost = ab_cost(i, j + 1)
            + calculate_cost(loc, d_new, a, b, a_roi, b_roi, m_kernel_roi_bottom_p)
            - calculate_cost(loc, d_new, a, b, a_roi, b_roi, m_kernel_roi_bottom_n);
#ifdef DEBUG
          VW_DEBUG_ASSERT(fabs(cost - calculate_cost(loc, d_new, a, b, a_roi, b_roi,
                                                     m_kernel_roi)) < 1e-3,
                          MathErr() << "Bug");
#endif
          if (cost < curr_best_cost) {
            curr_best_cost = cost;
            curr_best_d = d_new;
          }
        }
      }
      if ( i > 0 && j < ab_disparity.rows() - 1 ) {
        // Compare lower left
        d_new = ab_disparity(i-1,j+1);
        if ( ba_roi.contains(d_new + loc) && curr_best_d != d_new) {
          cost = calculate_cost(loc, d_new, a, b, a_roi, b_roi, m_kernel_roi);
          if (cost < curr_best_cost) {
            curr_best_cost = cost;
            curr_best_d = d_new;
          }
        }
      }
      {
        // Compare LR alternative
        DispT d = ab_disparity(i,j);
        d_new = -ba_disparity(i + d[0] - ba_roi.min().x(),
                              j + d[1] - ba_roi.min().y());
        if ( ba_roi.contains(d_new + loc) && curr_best_d != d_new) {
          cost = calculate_cost(loc, d_new, a, b, a_roi, b_roi, m_kernel_roi);
          if (cost < curr_best_cost) {
            curr_best_cost = cost;
            curr_best_d = d_new;
          }
        }
      }

      ab_cost(i,j) = curr_best_cost;
      ab_disparity(i,j) = curr_best_d;
    }
  }
}

stereo::PatchMatchBase::PatchMatchBase( BBox2i const& search_region, Vector2i const& kernel,
                                        float consistency_threshold,
                                        int32 max_iterations) :
  m_search_region( search_region ),
  m_search_region_rl( -search_region.max(), -search_region.min() ),
  m_kernel_size( kernel ),
  m_consistency_threshold( consistency_threshold ),
  m_max_iterations(max_iterations) {
  m_expansion = m_kernel_size / 2;
  m_expansion +=
    Vector2i( BilinearInterpolation::pixel_buffer,
              BilinearInterpolation::pixel_buffer );

  m_kernel_roi = BBox2i(-m_kernel_size/2, m_kernel_size/2 + Vector2i(1,1) );
  m_kernel_roi_left_p = BBox2i(Vector2i(m_kernel_size.x()/2, -m_kernel_size.y()/2),
                               Vector2i(m_kernel_size.x()/2 + 1, m_kernel_size.y()/2 + 1));
  m_kernel_roi_left_n = BBox2i(Vector2i(-m_kernel_size.x()/2 - 1, -m_kernel_size.y()/2),
                               Vector2i(-m_kernel_size.x()/2, m_kernel_size.y()/2 + 1));
  m_kernel_roi_right_p = BBox2i(Vector2i(-m_kernel_size.x()/2, -m_kernel_size.y()/2),
                                Vector2i(-m_kernel_size.x()/2 + 1, m_kernel_size.y()/2 + 1));
  m_kernel_roi_right_n = BBox2i(Vector2i(m_kernel_size.x()/2 + 1, -m_kernel_size.y()/2),
                                Vector2i(m_kernel_size.x()/2 + 2, m_kernel_size.y()/2 + 1));
  m_kernel_roi_bottom_p = BBox2i(Vector2i(-m_kernel_size.x()/2, -m_kernel_size.y()/2),
                                 Vector2i(m_kernel_size.x()/2+1, -m_kernel_size.y()/2+1));
  m_kernel_roi_bottom_n = BBox2i(Vector2i(-m_kernel_size.x()/2, m_kernel_size.y()/2 + 1),
                                 Vector2i(m_kernel_size.x()/2+1, m_kernel_size.y()/2 + 2));
  m_kernel_roi_top_p = BBox2i(Vector2i(-m_kernel_size.x()/2, m_kernel_size.y()/2),
                              Vector2i(m_kernel_size.x()/2+1, m_kernel_size.y()/2 + 1));
  m_kernel_roi_top_n = BBox2i(Vector2i(-m_kernel_size.x()/2, -m_kernel_size.y()/2 - 1),
                              Vector2i(m_kernel_size.x()/2+1, -m_kernel_size.y()/2));
  }
