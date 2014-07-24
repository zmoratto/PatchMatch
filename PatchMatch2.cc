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
  typedef ImageView<float>::pixel_accessor CAccT;
  typedef ImageView<DispT>::pixel_accessor DAccT;
  CAccT cost_row = ab_cost.origin();
  DAccT disp_row = ab_disparity.origin();
  Vector2i loc;
  for ( loc[1] = 0; loc[1] < ab_disparity.rows(); loc[1]++ ) {
    CAccT cost_col = cost_row;
    DAccT disp_col = disp_row;
    for ( loc[0] = 0; loc[0] < ab_disparity.cols(); loc[0]++ ) {
      *cost_col =
        calculate_cost( loc, *disp_col,
                        a, b, a_roi, b_roi, m_kernel_roi);
      cost_col.next_col();
      disp_col.next_col();
    }
    cost_row.next_row();
    disp_row.next_row();
  }
}

// Evaluates current disparity and writes its cost
void stereo::PatchMatchBase::evaluate_disparity( ImageView<float> const& a, ImageView<float> const& b,
                                                 BBox2i const& a_roi, BBox2i const& b_roi,
                                                 ImageView<DispT> const& ab_disparity_prior,
                                                 ImageView<float> const& ab_cost_prior,
                                                 ImageView<DispT>& ab_disparity,
                                                 ImageView<float>& ab_cost ) const {
  for ( int j = 0; j < ab_disparity.rows(); j++ ) {
    for ( int i = 0; i < ab_disparity.cols(); i++ ) {
      if (ab_disparity(i,j) == ab_disparity_prior(i,j)) {
        // Most common case first
        // This means we are using the prior
        ab_cost(i,j) = ab_cost_prior(i,j);
      } else {
        ab_cost(i,j) =
          calculate_cost( Vector2i(i,j), ab_disparity(i,j),
                          a, b, a_roi, b_roi, m_kernel_roi);
      }
    }
  }
}

void stereo::PatchMatchBase::keep_lowest_cost( ImageView<DispT> const& src_disp,
                                               ImageView<float> const& src_cost,
                                               ImageView<DispT>& dest_disp,
                                               ImageView<float>& dest_cost ) const {
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
      DispT starting_d = ab_disparity(i, j);

      if ( i > 0 ) {
        // Compare left
        d_new = ab_disparity(i-1,j);
        if ( ba_roi.contains(d_new + loc) && curr_best_d != d_new && starting_d != d_new) {
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
        if ( ba_roi.contains(d_new + loc) && curr_best_d != d_new && starting_d != d_new) {
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
      /*
      if ( i > 0 && j > 0 && false) {
        // Compare upper left
        d_new = ab_disparity(i-1,j-1);
        if ( ba_roi.contains(d_new + loc) && curr_best_d != d_new && starting_d != d_new) {
          cost = calculate_cost(loc, d_new, a, b, a_roi, b_roi,
                                m_kernel_roi);
#ifdef DEBUG
          float hold = calculate_cost(loc, d_new, a, b, a_roi, b_roi,
                                      m_kernel_roi);
          VW_DEBUG_ASSERT(fabs(cost - hold) < 1e-3,
                          MathErr() << "Bug");
#endif
          if (cost < curr_best_cost) {
            curr_best_cost = cost;
            curr_best_d = d_new;
          }
        }
      }
      */
      if ( i < ab_disparity.cols() - 1) {
        // Compare right
        d_new = ab_disparity(i+1,j);
        if ( ba_roi.contains(d_new + loc) && curr_best_d != d_new && starting_d != d_new) {
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
      /*
      if ( i < ab_disparity.cols() - 1 && j > 0 && false ) {
        // Compare upper right
        d_new = ab_disparity(i+1,j-1);
        if ( ba_roi.contains(d_new + loc) && curr_best_d != d_new && starting_d != d_new) {
          cost = calculate_cost(loc, d_new, a, b, a_roi, b_roi, m_kernel_roi);
          if (cost < curr_best_cost) {
            curr_best_cost = cost;
            curr_best_d = d_new;
          }
        }
      }
      if ( i < ab_disparity.cols() - 1 && j < ab_disparity.rows() - 1 && false ) {
        // Compare lower right
        d_new = ab_disparity(i+1,j+1);
        if ( ba_roi.contains(d_new + loc) && curr_best_d != d_new && starting_d != d_new) {
          cost = calculate_cost(loc, d_new, a, b, a_roi, b_roi, m_kernel_roi);
          if (cost < curr_best_cost) {
            curr_best_cost = cost;
            curr_best_d = d_new;
          }
        }

      }
      */
      if ( j < ab_disparity.rows() - 1 ) {
        // Compare lower
        d_new = ab_disparity(i,j+1);
        if ( ba_roi.contains(d_new + loc) && curr_best_d != d_new && starting_d != d_new) {
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
      /*
      if ( i > 0 && j < ab_disparity.rows() - 1 && false ) {
        // Compare lower left
        d_new = ab_disparity(i-1,j+1);
        if ( ba_roi.contains(d_new + loc) && curr_best_d != d_new && starting_d != d_new) {
          cost = calculate_cost(loc, d_new, a, b, a_roi, b_roi, m_kernel_roi);
          if (cost < curr_best_cost) {
            curr_best_cost = cost;
            curr_best_d = d_new;
          }
        }
      }
      */
#ifndef DISABLE_RL
      {
        // Compare LR alternative
        DispT d = ab_disparity(i,j);
        d_new = -ba_disparity(i + d[0] - ba_roi.min().x(),
                              j + d[1] - ba_roi.min().y());
        if ( ba_roi.contains(d_new + loc) && curr_best_d != d_new && starting_d != d_new) {
          cost = calculate_cost(loc, d_new, a, b, a_roi, b_roi, m_kernel_roi);
          if (cost < curr_best_cost) {
            curr_best_cost = cost;
            curr_best_d = d_new;
          }
        }
      }
#endif

      ab_cost(i,j) = curr_best_cost;
      ab_disparity(i,j) = curr_best_d;
    }
  }
}

void
stereo::PatchMatchBase::cross_corr_consistency_check(ImageView<DispT> const& ab_disparity,
                                                     ImageView<DispT> const& ba_disparity,
                                                     BBox2i const& ab_roi,
                                                     BBox2i const& ba_roi,
                                                     ImageView<PixelMask<DispT> >& ab_masked_disp) const {
  if (m_consistency_threshold < 0) {
    ab_masked_disp = ab_disparity;
    return;
  }

  for ( int j = 0; j < ab_disparity.rows(); j++ ) {
    for ( int i = 0; i < ab_disparity.cols(); i++ ) {
      Vector2i other =
        Vector2i(i, j) + Vector2i(ab_disparity(i, j))
        + ab_roi.min() - ba_roi.min();
      if (norm_2(Vector2f(ab_disparity(i, j)) +
                 Vector2f(ba_disparity(other.x(), other.y())))  < m_consistency_threshold) {
        // It is good
        ab_masked_disp(i, j) = ab_disparity(i, j);
      } else {
        ab_masked_disp(i, j) = PixelMask<DispT>();
      }
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

  Vector2i khalf = m_kernel_size/2;
  m_kernel_roi = BBox2i(-khalf, khalf + Vector2i(1,1) );
  m_kernel_roi_left_p =
    BBox2i(Vector2i(khalf.x(), -khalf.y()),
           Vector2i(khalf.x() + 1, khalf.y() + 1));
  m_kernel_roi_left_n =
    BBox2i(Vector2i(-khalf.x() - 1, -khalf.y()),
           Vector2i(-khalf.x(), khalf.y() + 1));
  m_kernel_roi_right_p =
    BBox2i(Vector2i(-khalf.x(), -khalf.y()),
           Vector2i(-khalf.x() + 1, khalf.y() + 1));
  m_kernel_roi_right_n =
    BBox2i(Vector2i(khalf.x() + 1, -khalf.y()),
           Vector2i(khalf.x() + 2, khalf.y() + 1));
  m_kernel_roi_bottom_p =
    BBox2i(Vector2i(-khalf.x(), -khalf.y()),
           Vector2i(khalf.x()+1, -khalf.y()+1));
  m_kernel_roi_bottom_n =
    BBox2i(Vector2i(-khalf.x(), khalf.y() + 1),
           Vector2i(khalf.x()+1, khalf.y() + 2));
  m_kernel_roi_top_p =
    BBox2i(Vector2i(-khalf.x(), khalf.y()),
           Vector2i(khalf.x()+1, khalf.y() + 1));
  m_kernel_roi_top_n =
    BBox2i(Vector2i(-khalf.x(), -khalf.y() - 1),
           Vector2i(khalf.x()+1, -khalf.y()));
  m_kernel_roi_tl_p1 =
    BBox2i(Vector2i(-khalf.x(),khalf.y()),
           Vector2i(khalf.x()+1,khalf.y()+1));
  m_kernel_roi_tl_p2 =
    BBox2i(khalf.x(), -khalf.y(), 1, m_kernel_size.y()-1);
  m_kernel_roi_tl_n1 =
    BBox2i(-khalf.x()-1, -khalf.y()-1, 1, m_kernel_size.y());
  m_kernel_roi_tl_n2 =
    BBox2i(-khalf.x(), -khalf.y()-1, m_kernel_size.x()-1, 1);
  }
