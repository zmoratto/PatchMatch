#ifndef __VW_STEREO_PATCHMATCH_H__
#define __VW_STEREO_PATCHMATCH_H__

#include <vw/Math/BBox.h>
#include <vw/Math/Vector.h>
#include <vw/Image/Filter.h>
#include <vw/Image/ImageMath.h>
#include <vw/Image/ImageView.h>
#include <vw/Image/ImageViewBase.h>
#include <vw/Image/Manipulation.h>
#include <vw/Image/Statistics.h>
#include <vw/Image/Transform.h>
#include <vw/Stereo/Correlate.h>
#include <vw/Stereo/CostFunctions.h>

#ifdef DEBUG
#include <iomanip>

#include <vw/FileIO.h>
#endif

namespace boost {
  namespace random {
    class rand48;
  }
}

namespace vw {
  namespace stereo {

    class PatchMatchBase {
    protected:
      BBox2i m_search_region;
      Vector2i m_kernel_size;
      Vector2i m_search_size;
      Vector2f m_search_size_f;
      Vector2i m_expansion;
      float m_consistency_threshold;
      ImageView<float> m_weight;

      typedef Vector2f DispT;
      typedef boost::random::rand48 GenT;

      void add_uniform_noise( Vector2f const& lo,
                              Vector2f const& hi,
                              ImageView<DispT>& disp ) const;

      // There can probably be an optimized version of this wrriten
      // for SSE4 using BLEND insutrctions.
      void keep_best_disparity( ImageView<DispT>& dest_disp,
                                ImageView<float>& dest_cost,
                                ImageView<DispT> const& src_disp,
                                ImageView<float> const& src_cost ) const;

      // Used to propogate be LR and RL disparities
      void transfer_disparity( ImageView<DispT>& dest_disp,
                               Vector2i const& dest_offset,
                               ImageView<DispT> const& src_disp,
                               Vector2i const& src_offset ) const;

      // Used for cross consistency checking
      void transfer_disparity_mark_invalid( ImageView<DispT>& dest_disp,
                                            Vector2i const& dest_offset,
                                            ImageView<DispT> const& src_disp,
                                            Vector2i const& src_offset ) const;

      // Used for cross consistency checking
      void mark_invalid( ImageView<PixelMask<DispT> >& dest_disp,
                         ImageView<DispT> const& comp_disp ) const;

    public:
      PatchMatchBase( BBox2i const& bbox, Vector2i const& kernel,
                      float consistency_threshold );
    };

    template <class Image1T, class Image2T>
    class PatchMatchView : public ImageViewBase<PatchMatchView<Image1T, Image2T> >, PatchMatchBase {
      Image1T m_left_image;
      Image2T m_right_image;

      // Types to help me when program
      typedef typename Image1T::pixel_type Pixel1T;
      typedef typename Image2T::pixel_type Pixel2T;

      template <class ImageT, class TransformT>
      TransformView<vw::InterpolationView<ImageT, BilinearInterpolation>, TransformT>
      inline transform_no_edge( ImageViewBase<ImageT> const& v,
                                TransformT const& transform_func ) const {
        return TransformView<InterpolationView<ImageT, BilinearInterpolation>, TransformT>( InterpolationView<ImageT, BilinearInterpolation>( v.impl() ), transform_func );
      }

      // Takes a disparity and writes to cost
      template <int D_OFFSET_X,  int D_OFFSET_Y>
      void evaluate_disparity( ImageView<Pixel1T> const& a,
                               ImageView<Pixel2T> const& b,
                               Vector2i const& a_offset,  // b_offset assumed zero
                               ImageView<DispT>& ab_disp, // Gets modified on non-zero D_OFFSET
                               ImageView<float>& ab_cost  // Always modified
                               ) const {
        BBox2i kernel_roi( -m_kernel_size/2, m_kernel_size/2 + Vector2i(1,1) );
        // Possibly there should be one for the left and one for the right?
        const float color_weight = 0.0549f * float(ChannelRange<typename CompoundChannelType<Pixel1T>::type>::max() - ChannelRange<typename CompoundChannelType<Pixel1T>::type>::min());

#ifdef DEBUG
        size_t improve_cnt = 0;
#endif

        ImageView<float> lcrop, rcrop;

        // My hope is that these conditionals collapse during
        // compiling because they are switching based on template
        // parameters.
        const int DIR_X = D_OFFSET_X < 1 ? 1 : -1;
        const int DIR_Y = D_OFFSET_Y < 1 ? 1 : -1;
        const int START_X = D_OFFSET_X < 1 ? 0 - D_OFFSET_X : ab_disp.cols() - 2;
        const int START_Y = D_OFFSET_Y < 1 ? 0 - D_OFFSET_Y : ab_disp.rows() - 2;
        const int LOWER_X = D_OFFSET_X < 0 ? 1 : 0;
        const int LOWER_Y = D_OFFSET_Y < 0 ? 1 : 0;
        const int UPPER_X = D_OFFSET_X < 1 ? ab_disp.cols() : ab_disp.cols() - 1;
        const int UPPER_Y = D_OFFSET_Y < 1 ? ab_disp.rows() : ab_disp.rows() - 1;
        for ( int j = START_Y; j >= LOWER_Y && j < UPPER_Y; j += DIR_Y ) {
          for ( int i = START_X; i >= LOWER_X && i < UPPER_X; i += DIR_X ) {
            Vector2f a_index =
              Vector2f(i,j) + a_offset;
            Vector2f b_index =
              Vector2f(i,j) + ab_disp(i+D_OFFSET_X,j+D_OFFSET_Y) + m_expansion;

            // Rasterize before hand for calculating masks
            lcrop = crop( a, kernel_roi + a_index );
            rcrop = crop( transform_no_edge(b, TranslateTransform(-b_index[0], -b_index[1])),
                          kernel_roi );

#if 0
            // This is simple NCC
            double result =
              1 - sum_of_pixel_values(lcrop*rcrop)/sqrt(sum_of_pixel_values(square(lcrop))*sum_of_pixel_values(square(rcrop)));

            // This is simple SAD
            // double result = sum_of_pixel_values(abs(lcrop-rcrop));
#else
            // This is NCC with adaptive support weights
            ImageView<float> weight = copy(m_weight);
            float left_color = lcrop(m_kernel_size.x()/2,m_kernel_size.y()/2),
              right_color = rcrop(m_kernel_size.x()/2,m_kernel_size.y()/2);
            float sum = 0;
            for ( int jk = 0; jk < m_kernel_size.y(); jk++ ) {
              for ( int ik = 0; ik < m_kernel_size.x(); ik++ ) {
                // Left side could be precomputed
                weight(ik,jk) *= exp(-fabs( lcrop(ik,jk) - left_color )/color_weight -
                                     fabs( rcrop(ik,jk) - right_color )/color_weight);
                sum += weight(ik,jk);
              }
            }

            // NCC weights
            float result =
              1 - sum_of_pixel_values(lcrop*rcrop*weight) /
              sqrt(sum_of_pixel_values(weight*square(lcrop))*sum_of_pixel_values(weight*square(rcrop)));
            // ZNCC weights
            // double result =
            //   1 - sum_of_pixel_values((lcrop - a_mv_l.x()) * weight *
            //                           (rcrop - b_mv_l.x()) ) / // / (a_mv_l.y() * b_mv_l.y()) ) /
            //   ( sum );
#endif

            if ( D_OFFSET_X == 0 && D_OFFSET_Y == 0 ) {
              // The caller is expected to call
              // 'keep_best_disparity'. Keep disparity at a later date
              // can be rewritten to use vector instructions to run
              // much faster.
              ab_cost(i,j) = result;
            } else {
              if ( result < ab_cost(i,j) ) {
#ifdef DEBUG
                improve_cnt++;
#endif
                ab_cost(i,j) = result;
                ab_disp(i,j) = ab_disp(i+D_OFFSET_X,j+D_OFFSET_Y);
              }
            }
          }
        }
#ifdef DEBUG
        std::cout << "Evaluate improved: " << improve_cnt << std::endl;
#endif
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
        PatchMatchBase(search_region, kernel_size,
                       consistency_threshold),
        m_left_image(left.impl()), m_right_image(right.impl()) {
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
        // TODO: Check search range to see if correlation is even
        // possible given the values of search range. Also consider
        // cropping search range based on inputs.

        // 1. Define Left ROI.
        BBox2i l_roi = bbox;

        // 2. Define Right ROI.
        BBox2i r_roi = l_roi;
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
        l_exp_roi.min() -= m_expansion;
        l_exp_roi.max() += m_expansion;
        r_exp_roi.min() -= m_expansion;
        r_exp_roi.max() += m_expansion;

#ifdef DEBUG
        std::cout << "Search: " << m_search_region << " Exp: "
                  << m_expansion << " Right: " << bounding_box(m_right_image)
                  << std::endl;
        std::cout << "L_ROI: " << l_roi << std::endl;
        std::cout << "R_ROI: " << r_roi << std::endl;
        std::cout << "L_EXP_ROI: " << l_exp_roi << std::endl;
        std::cout << "R_EXP_ROI: " << r_exp_roi << std::endl;
#endif

        // 6. Allocate buffers
        ImageView<Pixel1T> l_exp( crop( edge_extend(m_left_image), l_exp_roi ) );
        ImageView<Pixel2T> r_exp( crop( edge_extend(m_right_image), r_exp_roi ) );
        ImageView<DispT>
          l_disp( l_roi.width(), l_roi.height() ), l_disp_f( l_roi.width(), l_roi.height() ),
          r_disp( r_roi.width(), r_roi.height() ), r_disp_f( r_roi.width(), r_roi.height() );
        ImageView<float>
          l_cost( l_roi.width(), l_roi.height() ), l_cost_f( l_roi.width(), l_roi.height() ),
          r_cost( r_roi.width(), r_roi.height() ), r_cost_f( r_roi.width(), r_roi.height() );
        fill( l_disp, DispT() ); // TODO Is this needed?
        fill( r_disp, DispT() );

        // 7. Write uniform noise
        add_uniform_noise( Vector2f(0,0), m_search_size_f,
                           l_disp );
        add_uniform_noise( Vector2f(0,0), m_search_size_f,
                           r_disp );

        // 8. Evaluate the current disparities
        evaluate_disparity<0,0>(l_exp, r_exp,
                                l_roi.min() - l_exp_roi.min(),
                                l_disp, l_cost);
        evaluate_disparity<0,0>(r_exp, l_exp,
                                r_roi.min() - r_exp_roi.min(),
                                r_disp, r_cost);
#ifdef DEBUG
        std::cout << std::setprecision(10)
                  << "Starting cost:\t" << sum_of_pixel_values(l_cost) << std::endl;
#endif

        // 9. Implement iterative search.
        for ( int iterations = 0; iterations < 6; iterations++ ) {
#ifdef DEBUG
          std::ostringstream ostr;
          ostr << bbox.min()[0] << "_" << bbox.min()[1] << "_" << iterations << "_";
          if ( iterations == 0 ) {
            write_image( ostr.str() + "lexp.tif", l_exp );
            write_image( ostr.str() + "rexp.tif", r_exp );
          }
#endif
          if ( iterations & 1 ) {
            // 9.1 Compare to Left
            evaluate_disparity<-1,0>(l_exp, r_exp,
                                     l_roi.min() - l_exp_roi.min(),
                                     l_disp, l_cost);
#ifdef DEBUG
            write_image( ostr.str() + "0-D.tif", l_disp );
            std::cout << "After left:\t" << sum_of_pixel_values(l_cost) << std::endl;
#endif
            evaluate_disparity<-1,0>(r_exp, l_exp,
                                     r_roi.min() - r_exp_roi.min(),
                                     r_disp, r_cost);

            // 9.2 Compare to Above
            evaluate_disparity<0,-1>(l_exp, r_exp,
                                     l_roi.min() - l_exp_roi.min(),
                                     l_disp, l_cost);
#ifdef DEBUG
            write_image( ostr.str() + "1-D.tif", l_disp );
            std::cout << "After Above:\t" << sum_of_pixel_values(l_cost) << std::endl;
#endif
            evaluate_disparity<0,-1>(r_exp, l_exp,
                                     r_roi.min() - r_exp_roi.min(),
                                     r_disp, r_cost);
          } else {
            // 9.3 Compare to Right
            evaluate_disparity<1,0>(l_exp, r_exp,
                                    l_roi.min() - l_exp_roi.min(), l_disp, l_cost);
#ifdef DEBUG
            write_image( ostr.str() + "2-D.tif", l_disp );
            std::cout << "After right:\t" << sum_of_pixel_values(l_cost) << std::endl;
#endif
            evaluate_disparity<1,0>(r_exp, l_exp,
                                    r_roi.min() - r_exp_roi.min(), r_disp, r_cost);

            // 9.4 Compare to Bottom
            evaluate_disparity<0,1>(l_exp, r_exp,
                                    l_roi.min() - l_exp_roi.min(), l_disp, l_cost);
#ifdef DEBUG
            write_image( ostr.str() + "3-D.tif", l_disp );
            std::cout << "After bottom:\t" << sum_of_pixel_values(l_cost) << std::endl;
#endif
            evaluate_disparity<0,1>(r_exp, l_exp,
                                    r_roi.min() - r_exp_roi.min(), r_disp, r_cost);
          }

          // 9.5 Compare LR against RL
          std::copy( l_disp.data(), l_disp.data() + prod(l_roi.size()), l_disp_f.data() );
          transfer_disparity( l_disp_f, l_roi.min() - l_exp_roi.min(),
                              r_disp, r_roi.min() - r_exp_roi.min() );
          evaluate_disparity<0,0>(l_exp, r_exp,
                                  l_roi.min() - l_exp_roi.min(), l_disp_f, l_cost_f);
          keep_best_disparity(l_disp, l_cost, l_disp_f, l_cost_f );
#ifdef DEBUG
          write_image( ostr.str() + "4-D.tif", l_disp );
          std::cout << "After prop:\t" << sum_of_pixel_values(l_cost) << std::endl;
#endif

          // 9.6 Compare RL against LR
          std::copy( r_disp.data(), r_disp.data() + prod(r_roi.size()), r_disp_f.data() );
          transfer_disparity( r_disp_f, r_roi.min() - r_exp_roi.min(),
                              l_disp, l_roi.min() - l_exp_roi.min() );
          evaluate_disparity<0,0>(r_exp, l_exp,
                                  r_roi.min() - r_exp_roi.min(), r_disp_f, r_cost_f);
          keep_best_disparity(r_disp, r_cost, r_disp_f, r_cost_f );

          // 9.7 Add noise and evaluate disparity
          Vector2f half_search =
            m_search_size_f * 0.25f / pow(2.0f,iterations);
          std::copy( l_disp.data(), l_disp.data() + prod(l_roi.size()), l_disp_f.data() );
          std::copy( r_disp.data(), r_disp.data() + prod(r_roi.size()), r_disp_f.data() );
          add_uniform_noise( -half_search, half_search, l_disp_f );
          add_uniform_noise( -half_search, half_search, r_disp_f );
          evaluate_disparity<0,0>(l_exp, r_exp,
                                  l_roi.min() - l_exp_roi.min(), l_disp_f, l_cost_f);
          evaluate_disparity<0,0>(r_exp, l_exp,
                                  r_roi.min() - r_exp_roi.min(), r_disp_f, r_cost_f);
          keep_best_disparity(l_disp, l_cost, l_disp_f, l_cost_f);
          keep_best_disparity(r_disp, r_cost, r_disp_f, r_cost_f);
#ifdef DEBUG
          write_image( ostr.str() + "5-D.tif", l_disp );
          std::cout << "After reseed:\t" << sum_of_pixel_values(l_cost) << std::endl;
#endif
        }

        // Perform L R check
        std::copy( l_disp.data(), l_disp.data() + prod(l_roi.size()),
                   l_disp_f.data() );
        transfer_disparity_mark_invalid( l_disp_f, l_roi.min() - l_exp_roi.min(),
                                         r_disp, r_roi.min() - r_exp_roi.min() );
        ImageView<pixel_type> output = pixel_cast<pixel_type>(l_disp);
        mark_invalid( output, l_disp_f );

        return prerasterize_type(output + pixel_type(m_search_region.min()),
                        -bbox.min().x(), -bbox.min().y(),
                        cols(), rows());
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
