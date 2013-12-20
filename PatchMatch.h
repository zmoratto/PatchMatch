#ifndef __VW_STEREO_PATCHMATCH_H__
#define __VW_STEREO_PATCHMATCH_H__

#include <vw/Core/Thread.h>
#include <vw/Image/ImageView.h>

namespace vw {
  namespace stereo {

    template <class Image1T, class Image2T>
    class PatchMatchView : public ImageViewBase<PatchMatchView<Image1T, Image2T> > {
      Image1T m_left_image;
      Image2T m_right_image;
      BBox2i m_search_region;
      Vector2i m_kernel_size;
      Vector2i m_search_size;
      Vector2f m_search_size_f;
      float m_consistency_threshold;

      // Types to help me when program
      typedef typename Image1T::pixel_type Pixel1T;
      typedef typename Image2T::pixel_type Pixel2T;
      typedef Vector2f DispT;
      typedef boost::rand48 GenT;

      template <class ImageT, class TransformT>
      TransformView<vw::InterpolationView<ImageT, BilinearInterpolation>, TransformT>
      inline transform_no_edge( ImageViewBase<ImageT> const& v,
                                TransformT const& transform_func ) const {
        return TransformView<InterpolationView<ImageT, BilinearInterpolation>, TransformT>( InterpolationView<ImageT, BilinearInterpolation>( v.impl() ), transform_func );
      }

      // To avoid casting higher for uint8 subtraction
      template <class PixelT>
      struct AbsDiffFunc : public vw::ReturnFixedType<PixelT> {
        inline PixelT operator()( PixelT const& a, PixelT const& b ) const {
          return abs( a - b );
        }
      };

      // Floating point square cost functor that uses adaptive support weights
      float calculate_cost( Vector2f const& a_loc, Vector2f const& disparity,
                            ImageView<Pixel1T> const& a, ImageView<Pixel2T> const& b,
                            BBox2i const& a_roi, BBox2i const& b_roi ) const {
        BBox2i kernel_roi( -m_kernel_size/2, m_kernel_size/2 + Vector2i(1,1) );

        ImageView<float> left_kernel
          = crop(a, kernel_roi + a_loc - a_roi.min() );
        ImageView<float> right_kernel
          = crop( transform_no_edge(b,
                                    TranslateTransform(-(a_loc.x() + disparity[0] - float(b_roi.min().x())),
                                                       -(a_loc.y() + disparity[1] - float(b_roi.min().y())))),
                  kernel_roi );

        // Calculate support weights for left and right
        ImageView<float>
          weight(m_kernel_size.x(),m_kernel_size.y());

        Vector2f center_index = m_kernel_size/2;
        float left_color = left_kernel(center_index[0],center_index[1]),
          right_color = right_kernel(center_index[0],center_index[1]);
        float kernel_diag = norm_2(m_kernel_size);
        float sum = 0;
        for ( int j = 0; j < m_kernel_size.y(); j++ ) {
          for ( int i = 0; i < m_kernel_size.x(); i++ ) {
            float dist = norm_2( Vector2f(i,j) - center_index )/kernel_diag;
            //float dist = 0;
            float lcdist = fabs( left_kernel(i,j) - left_color );
            float rcdist = fabs( right_kernel(i,j) - right_color );
            sum += weight(i,j) = exp(-lcdist/14 - dist) * exp(-rcdist/14 - dist);
          }
        }

        return sum_of_pixel_values( weight * per_pixel_filter(left_kernel,right_kernel,AbsDiffFunc<float>())) / sum;
      }

      // Takes a disparity and writes to cost
      template <int D_OFFSET_X,  int D_OFFSET_Y>
      void evaluate_disparity( ImageView<Pixel1T> const& a,
                               ImageView<Pixel2T> const& b,
                               Vector2i const& a_offset,  // b_offset assumed zero
                               ImageView<DispT>& ab_disp, // Gets modified on non-zero D_OFFSET
                               ImageView<float>& ab_cost  // Always modified
                               ) {
        BBox2i kernel_roi( -kernel_size/2, kernel_size/2 + Vector2i(1,1) );

        // My hope is that these conditionals collapse during
        // compiling because they are switching based on template
        // parameters.
        const int DIR_X = D_OFFSET_X < 1 ? 1 : -1;
        const int DIR_Y = D_OFFSET_Y < 1 ? 1 : -1;
        const int START_X = D_OFFSET_X < 1 ? 0 - D_OFFSET_X : ab_disp.cols() - D_OFFSET_X;
        const int START_Y = D_OFFSET_Y < 1 ? 0 - D_OFFSET_Y : ab_disp.rows() - D_OFFSET_Y;
        const int LOWER_X = D_OFFSET_X < 0 ? 1 : 0;
        const int LOWER_Y = D_OFFSET_Y < 0 ? 1 : 0;
        const int UPPER_X = D_OFFSET_X < 1 ? ab_disp.cols() : ab_disp.cols() - 1;
        const int UPPER_Y = D_OFFSET_Y < 1 ? ab_disp.rows() : ab_disp.rows() - 1;
        for ( int j = START_Y; j >= LOWER_Y && j < UPPER_Y; j += DIR_Y ) {
          for ( int i = START_X; i >= LOWER_X && i < UPPER_Y; i += DIR_X ) {
            Vector2f a_index =
              Vector2f(i,j) + a_offset;
            Vector2f b_index =
              Vector2f(i,j) + ab_disp(i+D_OFFSET_X,j+D_OFFSET_Y);

            // This is the basic
            float result =
              sum_of_pixel_values
              (per_pixel_filter
               (crop( a, kernel_roi + a_index ),
                crop( transform_no_edge(b, TranslateTransform(-b_index[0], -b_index[1])),
                      kernel_roi ), AbsDiffFunc<uint8>() ));
            if ( D_OFFSET_X == 0 && D_OFFSET_Y == 0 ) {
              ab_cost(i,j) = result;
            } else {
              if ( result < ab_cost(i,j) ) {
                ab_cost(i,j) = result;
                ab_disp(i,j) = ab_disp(i+D_OFFSET_X,j+D_OFFSET_Y);
              }
            }
          }
        }
      }

      void add_uniform_noise( Vector2f const& lo,
                              Vector2f const& hi,
                              ImageView<DispT>& disp,
                              GenT& gen ) {
        typedef boost::random::uniform_real_distribution<float> DistributionT;
        typedef boost::variate_generator<GenT, DistributionT > vargen_type;
        vargen_type source_x(gen, DistributionT(lo[0],hi[0])),
          source_y(gen, DistributionT(lo[1],hi[1]));

        typedef ImageView<DispT> ImageT;
        typename ImageT::pixel_accessor row = disp.origin();
        for ( int j = disp.rows(); j; --j ) {
          typename ImageT::pixel_accessor col = row;
          for ( int i = disp.cols(); i; --i ) {
            (*col)[0] += source_x();
            (*col)[1] += source_y();
            (*col)[0] = std::max(0,std::min(m_search_size_f[0],(*col)[0]));
            (*col)[1] = std::max(0,std::min(m_search_size_f[1],(*col)[1]));
            col.next_col();
          }
          row.next_row();
        }
      }

      // There can probably be an optimized version of this wrriten
      // for SSE4 using BLEND insutrctions.
      void keep_best_disparity( ImageView<DispT>& dest_disp,
                                ImageView<float>& dest_cost,
                                ImageView<DispT> const& src_disp,
                                ImageView<float> const& src_cost ) {
        typedef ImageView<DispT> DispT;
        typedef ImageView<float> CostT;

        typename DispT::pixel_accessor dest_disp_row = dest_disp.origin();
        typename DispT::pixel_accessor src_disp_row = src_disp.origin();
        typename CostT::pixel_accessor dest_cost_row = dest_cost.origin();
        typename CostT::pixel_accessor src_cost_row = src_cost.origin();
        for ( int j = dest_disp.rows(); j; --j ) {
          typename DispT::pixel_accessor dest_disp_col = dest_disp_row;
          typename DispT::pixel_accessor src_disp_col = src_disp_row;
          typename CostT::pixel_accessor dest_cost_col = dest_cost_row;
          typename CostT::pixel_accessor src_cost_col = src_cost_row;
          for ( int i = dest_disp.cols(); i; --i ) {
            if ( *dest_cost > *src_cost ) {
              *dest_disp = *src_disp;
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
      }

    public:
      typedef PixelMask<Vector2f> pixel_type;
      typedef pixel_type result_type;
      typedef ProceduralPixelAccessor<PatchMatchView> pixel_accessor;

      // Search region values are inclusive.
      PatchMatchView( ImageViewBase<Image1T> const& left,
                      ImageViewBase<Image2T> const& right,
                      BBox2i const& search_region, Vector2i const& kernel_size,
                      float consistency_threshold = 1 ) :
        m_left_image(left.impl()), m_right_image(right.impl()),
        m_search_region(search_region), m_kernel_size(kernel_size),
        m_consistency_threshold(consistency_threshold) {
        m_search_size = m_search_region.size();
        m_search_size_f = m_search_size;
      }

      // Standard required ImageView interfaces
      inline int32 cols() const { return m_left_image.cols(); }
      inline int32 rows() const { return m_left_image.rows(); }
      inline int32 planes() const { return 1; }

      inline pixel_accessor origin() const { return pixel_accessor( *this, 0, 0 ); }
      inline pixel_type operator()( int32 /*i*/, int32 /*j*/, int32 /*p*/ = 0) const {
        vw_throw( NoImplErr() << "PatchMatchView::operator()(....) has not been implemented." );
        return pixel_type();
      }

      // Block rasterization section that does actual work
      typedef CropView<ImageView<pixel_type> > prerasterize_type;
      inline prerasterize_type prerasterize(BBox2i const& bbox) const {
        // 1. Define Left ROI.
        BBox2i l_roi = bbox;

        // 2. Define Right ROI.
        BBox2i r_roi = l_roiOB;
        r_roi.min() += m_search_region.min();
        r_roi.max() += m_search_region.max();
        // Crop by the image bounds as we don't want to be calculating
        // disparities for interpolated regions.
        r_roi.crop( bounding_box( m_right_image ) );

        // 3. Define Left Expanded ROI. This is where all the possible
        // places we might make a pixel access to in the left.
        BBox2i l_exp_roi = r_roi;
        l_exp_roi.min() -= m_search_region.max();
        l_exp_roi.max() -= m_search_region.min();

        // 4. Define Right Expanded ROI.
        BBox2i r_exp_roi = l_roi;
        r_exp_roi.min() += m_search_region.min();
        r_exp_roi.max() += m_search_region.max();

        // 5. Expand the Expanded ROI by the kernel size and the space
        // need for interpolation.
        Vector2i half_kernel_size = m_kernel_size / 2;
        l_exp_roi.min() -= half_kernel_size;
        l_exp_roi.max() += half_kernel_size;
        r_exp_roi.min() -= half_kernel_size;
        r_exp_roi.max() += half_kernel_size;
        l_exp_roi.expand( BilinearInterpolation::pixel_buffer );
        r_exp_roi.expand( BilinearInterpolation::pixel_buffer );

        // 6. Allocate buffers
        ImageView<Pixel1T> l_exp( crop( edge_extend(m_left_image), l_exp_roi ) );
        ImageView<Pixel2T> r_exp( crop( edge_extend(m_right_image), r_exp_roi ) );
        ImageView<DispT>
          l_disp( l_roi.width(), l_roi.height() ), l_disp_f( l_roi.width(), l_roi.height() ),
          r_disp( r_roi.width(), r_roi.height() ), r_disp_f( r_roi.width(), r_roi.height() );
        ImageView<float>
          l_cost( l_roi.width(), l_roi.height() ), l_cost_f( l_roi.width(), l_roi.height() )
          r_cost( r_roi.width(), r_roi.height() ), r_cost_f( r_roi.width(), r_roi.height() );
        fill( l_disp, DispT() ); // TODO Is this needed?
        fill( r_disp, DispT() );

        // 7. Write uniform noise
        boost::rand48 gen(std::rand());
        vargen_type random_source(gen, boost::random::uniform_01<float>());
        add_uniform_noise( Vector2f(0,0), m_search_size_f,
                           l_disp, gen );
        add_uniform_noise( Vector2f(0,0), m_search_size_f,
                           r_disp, gen );

        // 8. Evaluate the current disparities
        evaluate_disparity<0,0>();
        evaluate_disparity<0,0>();

        // 9. Implement iterative search.
        for ( int iterations = 0; i < 6; i++ ) {
          if ( iteration && 1 ) {
            // 9.1 Compare to Left
            evaluate_disparity<-1,0>();
            evaluate_disparity<-1,0>();

            // 9.2 Compare to Above
            evaluate_disparity<0,-1>();
          } else {
            // 9.3 Compare to Right
            evaluate_disparity<1,0>();

            // 9.4 Compare to Bottom
            evaluate_disparity<0,1>();
          }

          // 9.5 Compare across LR

          // 9.6 Compare across RL

          // 9.7 Add noise and evaluate disparity
          Vector2f half_search =
            m_search_size_f * 0.25f / pow(2.0f,iterations);
          add_uniform_noise( -half_search, half_search,
                             l_disp, gen );
          add_uniform_noise( -half_search, half_search,
                             r_disp, gen );
          evaluate_disparity<0,0>();
          evaluate_disparity<0,0>();
          
        }
      }

      template <class DestT>
      inline void rasterize(DestT const& dest, BBox2i const& bbox) const {
        vw::rasterize(prerasterize(bbox), dest, bbox);
      }
    };

    template <class Image1T, class Image2T>
    PatchMatchView<Image1T,Image2T>
    patch_match( ImageViewBase<Image1T> const& left,
                 ImageViewBase<Image2T> const& right,
                 BBox2i const& search_region, Vector2i const& kernel_size,
                 float consistency_threshold = 1 ) {
      typedef PatchMatchView<Image1T,Image2T> result_type;
      return result_type( left.impl(), right.impl(), search_region,
                          kernel_size, consistency_threshold );
    }
  }} // vw::stereo

#endif//__VW_STEREO_PATCHMATCH_H__
