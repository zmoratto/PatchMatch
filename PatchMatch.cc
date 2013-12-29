#include "PatchMatch.h"
#include <vw/Core/Exception.h>
#include <vw/Math/BBox.h>
#include <vw/Math/Vector.h>
#include <vw/Image/ImageView.h>

#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/linear_congruential.hpp>

using namespace vw;

void
stereo::PatchMatchBase::add_uniform_noise( Vector2f const& lo,
                                           Vector2f const& hi,
                                           ImageView<stereo::PatchMatchBase::DispT>& disp ) const {
  typedef boost::random::uniform_real_distribution<float> DistributionT;
  typedef boost::variate_generator<GenT, DistributionT > vargen_type;
  vargen_type source_x(GenT(std::rand()), DistributionT(lo[0],hi[0])),
    source_y(GenT(std::rand()), DistributionT(lo[1],hi[1]));

  typedef ImageView<DispT> ImageT;
  typename ImageT::pixel_accessor row = disp.origin();
  for ( int j = disp.rows(); j; --j ) {
    typename ImageT::pixel_accessor col = row;
    for ( int i = disp.cols(); i; --i ) {
      (*col)[0] += source_x();
      (*col)[1] += source_y();
      (*col)[0] = std::max(0.0f,std::min(m_search_size_f[0],(*col)[0]));
      (*col)[1] = std::max(0.0f,std::min(m_search_size_f[1],(*col)[1]));
      col.next_col();
    }
    row.next_row();
  }
}

void
stereo::PatchMatchBase::keep_best_disparity( ImageView<DispT>& dest_disp,
                                             ImageView<float>& dest_cost,
                                             ImageView<DispT> const& src_disp,
                                             ImageView<float> const& src_cost ) const {
  typedef ImageView<DispT> IDispT;
  typedef ImageView<float> CostT;

#ifdef DEBUG
  size_t improve_cnt = 0;
#endif

  typename IDispT::pixel_accessor dest_disp_row = dest_disp.origin();
  typename IDispT::pixel_accessor src_disp_row = src_disp.origin();
  typename CostT::pixel_accessor dest_cost_row = dest_cost.origin();
  typename CostT::pixel_accessor src_cost_row = src_cost.origin();
  for ( int j = dest_disp.rows(); j; --j ) {
    typename IDispT::pixel_accessor dest_disp_col = dest_disp_row;
    typename IDispT::pixel_accessor src_disp_col = src_disp_row;
    typename CostT::pixel_accessor dest_cost_col = dest_cost_row;
    typename CostT::pixel_accessor src_cost_col = src_cost_row;
    for ( int i = dest_disp.cols(); i; --i ) {
      if ( *dest_cost_col > *src_cost_col ) {
        *dest_disp_col = *src_disp_col;
        *dest_cost_col = *src_cost_col;
#ifdef DEBUG
        improve_cnt++;
#endif
      }

      dest_disp_col.next_col();
      src_disp_col.next_col();
      dest_cost_col.next_col();
      src_cost_col.next_col();
    }
    dest_disp_row.next_row();
    src_disp_row.next_row();
    dest_cost_row.next_row();
    src_cost_row.next_row();
  }

#ifdef DEBUG
  std::cout << "Pixels improved: " << improve_cnt << std::endl;
#endif
}

// Used to propogate be LR and RL disparities
void
stereo::PatchMatchBase::transfer_disparity( ImageView<DispT>& dest_disp,
                                            Vector2i const& dest_offset,
                                            ImageView<DispT> const& src_disp,
                                            Vector2i const& src_offset ) const {
  typedef ImageView<DispT> IDispT;

  typename IDispT::pixel_accessor dest_disp_row = dest_disp.origin();
  for ( int j = 0; j < dest_disp.rows(); j++ ) {
    typename IDispT::pixel_accessor dest_disp_col = dest_disp_row;
    for ( int i = 0; i < dest_disp.cols(); i++ ) {
      Vector2i src_idx = Vector2i(i,j) + *dest_disp_col + m_expansion - src_offset;
      if ( src_idx[0] >= 0 && src_idx[1] >= 0 &&
           src_idx[0] < src_disp.cols() && src_idx[1] < src_disp.rows() ) {
        *dest_disp_col =
          -src_disp( src_idx.x(), src_idx.y() ) + DispT(dest_offset) +
          DispT(src_offset) - 2*DispT(m_expansion);
      }

      dest_disp_col.next_col();
    }
    dest_disp_row.next_row();
  }
}

// Used to propogate be LR and RL disparities
void
stereo::PatchMatchBase::transfer_disparity_mark_invalid(

                                                        ImageView<DispT>& dest_disp,
                                                        Vector2i const& dest_offset,
                                                        ImageView<DispT> const& src_disp,
                                                        Vector2i const& src_offset ) const {
  typedef ImageView<DispT> IDispT;

  typename IDispT::pixel_accessor dest_disp_row = dest_disp.origin();
  for ( int j = 0; j < dest_disp.rows(); j++ ) {
    typename IDispT::pixel_accessor dest_disp_col = dest_disp_row;
    for ( int i = 0; i < dest_disp.cols(); i++ ) {
      Vector2i src_idx = Vector2i(i,j) + *dest_disp_col + m_expansion - src_offset;
      if ( src_idx[0] >= 0 && src_idx[1] >= 0 &&
           src_idx[0] < src_disp.cols() && src_idx[1] < src_disp.rows() ) {
        *dest_disp_col =
          -src_disp( src_idx.x(), src_idx.y() ) + DispT(dest_offset) +
          DispT(src_offset) - 2*DispT(m_expansion);
      } else {
        *dest_disp_col = DispT(-1,-1);
      }

      dest_disp_col.next_col();
    }
    dest_disp_row.next_row();
  }
}

// Used for cross consistency checking
void
stereo::PatchMatchBase::mark_invalid( ImageView<PixelMask<DispT> >& dest_disp,
                                      ImageView<DispT> const& comp_disp ) const {
  typedef ImageView<PixelMask<DispT> > IMDispT;
  typedef ImageView<DispT> IDispT;

  VW_ASSERT( dest_disp.cols() == comp_disp.cols() &&
             dest_disp.rows() == comp_disp.rows(),
             ArgumentErr() << "Input arguments are not the same size" );

  typename IMDispT::pixel_accessor dest_disp_row = dest_disp.origin();
  typename IDispT::pixel_accessor comp_disp_row = comp_disp.origin();
  for ( int j = 0; j < dest_disp.rows(); j++ ) {
    typename IMDispT::pixel_accessor dest_disp_col = dest_disp_row;
    typename IDispT::pixel_accessor comp_disp_col = comp_disp_row;
    for ( int i = 0; i < dest_disp.cols(); i++ ) {
      if ( norm_2((*dest_disp_col).child() - *comp_disp_col) < m_consistency_threshold ) {
        (*dest_disp_col).validate();
      } else {
        (*dest_disp_col).invalidate();
      }
      dest_disp_col.next_col();
      comp_disp_col.next_col();
    }
    dest_disp_row.next_row();
    comp_disp_row.next_row();
  }
}

stereo::PatchMatchBase::PatchMatchBase( BBox2i const& search_region, Vector2i const& kernel,
                                        float consistency_threshold ) :
  m_search_region( search_region ), m_kernel_size( kernel ),
  m_consistency_threshold( consistency_threshold ) {
  m_search_size = m_search_region.size();
  m_search_size_f = m_search_size;
  m_expansion = m_kernel_size / 2;
  m_expansion +=
    Vector2i( BilinearInterpolation::pixel_buffer,
              BilinearInterpolation::pixel_buffer );
}
