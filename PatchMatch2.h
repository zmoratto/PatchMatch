#ifndef __VW_STEREO_PATCHMATCH2_H__
#define __VW_STEREO_PATCHMATCH2_H__

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
      BBox2i m_search_region_rl;
      Vector2i m_kernel_size;
      Vector2i m_expansion;
      float m_consistency_threshold;
      int32 m_max_iterations;

      typedef Vector2f DispT;
      typedef boost::random::rand48 GenT;

      void add_uniform_noise(BBox2f const& range_of_noise_to_add,
                             BBox2f const& max_search_range,
                             BBox2f const& other_image_bbox,
                             ImageView<DispT>& disparity ) const;


      // Simple square kernels
      float calculate_cost( Vector2f const& a_loc, Vector2f const& disparity,
                            ImageView<float> const& a, ImageView<float> const& b,
                            BBox2i const& a_roi, BBox2i const& b_roi ) const;

      // Evaluates current disparity and writes its cost
      void evaluate_disparity( ImageView<float> const& a, ImageView<float> const& b,
                               BBox2i const& a_roi, BBox2i const& b_roi,
                               ImageView<DispT>& ab_disparity,
                               ImageView<float>& ab_cost ) const;

      void keep_lowest_cost( ImageView<Vector2f>& dest_disp,
                             ImageView<float>& dest_cost,
                             ImageView<DispT> const& src_disp,
                             ImageView<float> const& src_cost ) const;

      // Propogates from the 3x3 neighbor hood
      void evaluate_8_connected( ImageView<float> const& a,
                                 ImageView<float> const& b,
                                 BBox2i const& a_roi, BBox2i const& b_roi,
                                 ImageView<Vector2f> const& ba_disparity,
                                 ImageView<Vector2f> const& ab_disparity_in,
                                 ImageView<float> const& ab_cost_in,
                                 ImageView<Vector2f>& ab_disparity_out,
                                 ImageView<float>& ab_cost_out ) const;

    public:
      PatchMatchBase( BBox2i const& bbox, Vector2i const& kernel,
                      float consistency_threshold, int32 max_iterations );
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

    public:
      typedef PixelMask<Vector2f> pixel_type;
      typedef pixel_type result_type;
      typedef ProceduralPixelAccessor<PatchMatchView> pixel_accessor;

      // Search region values are inclusive.
      PatchMatchView( ImageViewBase<Image1T> const& left,
                      ImageViewBase<Image2T> const& right,
                      BBox2i const& search_region, Vector2i const& kernel_size,
                      float consistency_threshold = 1,
                      int32 max_iterations = 6) :
        PatchMatchBase(search_region, kernel_size,
                       consistency_threshold,
                       max_iterations),
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
        BBox2i l_exp_roi = l_roi;
        l_exp_roi.min() -= m_expansion;
        l_exp_roi.max() += m_expansion;

        // 4. Define Right Expanded ROI.
        BBox2i r_exp_roi = r_roi;
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
        ImageView<float> l_exp( crop( edge_extend(m_left_image), l_exp_roi ) );
        ImageView<float> r_exp( crop( edge_extend(m_right_image), r_exp_roi ) );
        ImageView<DispT>
          l_disp( l_roi.width(), l_roi.height() ), l_disp_cpy( l_roi.width(), l_roi.height() ),
          r_disp( r_roi.width(), r_roi.height() ), r_disp_cpy( r_roi.width(), r_roi.height() );
        ImageView<float>
          l_cost( l_roi.width(), l_roi.height() ), l_cost_cpy( l_roi.width(), l_roi.height() ),
          r_cost( r_roi.width(), r_roi.height() ), r_cost_cpy( r_roi.width(), r_roi.height() );
        fill( l_disp, DispT() ); // TODO Is this needed?
        fill( r_disp, DispT() );

        // 7. Write uniform noise
        add_uniform_noise(m_search_region, m_search_region,
                          r_roi - l_roi.min(), l_disp);
        add_uniform_noise(m_search_region_rl, m_search_region_rl,
                          l_roi - r_roi.min(), r_disp );

#ifdef DEBUG
        {
          std::ostringstream ostr;
          ostr << bbox.min()[0] << "_" << bbox.min()[1] << "_";
          write_image( ostr.str() + "noisel-D.tif", l_disp );
          write_image( ostr.str() + "noiser-D.tif", r_disp );
        }
#endif

#ifdef DEBUG
        {
          std::ostringstream ostr;
          ostr << bbox.min()[0] << "_" << bbox.min()[1] << "_";
          write_image( ostr.str() + "lexp.tif", l_exp );
          write_image( ostr.str() + "rexp.tif", r_exp );
        }
#endif


        // 9. Implement iterative search.
        for ( int iteration = 0; iteration < m_max_iterations; iteration++ ) {

          // 8. Evaluate the current disparities
          evaluate_disparity(l_exp, r_exp,
                             l_exp_roi - l_roi.min(),
                             r_exp_roi - r_roi.min(),
                             l_disp, l_cost);
          evaluate_disparity(r_exp, l_exp,
                             r_exp_roi - r_roi.min(),
                             l_exp_roi - l_roi.min(),
                             r_disp, r_cost);

#ifdef DEBUG
          std::cout << std::setprecision(10)
                    << "Starting cost:\t" << sum_of_pixel_values(l_cost) << std::endl;
#endif


          { // Propogate
            evaluate_8_connected(l_exp, r_exp,
                                 l_exp_roi - l_roi.min(),
                                 r_exp_roi - r_roi.min(),
                                 r_disp, l_disp,
                                 l_cost, l_disp, l_cost);
            evaluate_8_connected(r_exp, l_exp,
                                 r_exp_roi - r_roi.min(),
                                 l_exp_roi - l_roi.min(),
                                 l_disp, r_disp,
                                 r_cost, r_disp, r_cost);
          }

          { // Add noise
            l_disp_cpy = copy(l_disp);
            r_disp_cpy = copy(r_disp);

            float scaling_size = 1.0 / pow(2.0, iteration + 1);
            Vector2f search_size_half =
              scaling_size * Vector2f(m_search_region.size());

            add_uniform_noise(BBox2f(-search_size_half, search_size_half),
                              m_search_region,
                              bounding_box(r_exp), l_disp_cpy);
            add_uniform_noise(BBox2f(-search_size_half, search_size_half),
                              m_search_region_rl,
                              bounding_box(l_exp), r_disp_cpy);

            evaluate_disparity(l_exp, r_exp,
                               l_exp_roi - l_roi.min(),
                               r_exp_roi - r_roi.min(),
                               l_disp_cpy, l_cost_cpy);
            evaluate_disparity(r_exp, l_exp,
                               r_exp_roi - r_roi.min(),
                               l_exp_roi - l_roi.min(),
                               r_disp_cpy, r_cost_cpy);

            keep_lowest_cost(l_disp, l_cost,
                             l_disp_cpy, l_cost_cpy);
            keep_lowest_cost(r_disp, r_cost,
                             r_disp_cpy, r_cost_cpy);
          }
        } // end of iterations

        ImageView<PixelMask<Vector2f> > final_disparity = l_disp;
        stereo::cross_corr_consistency_check(final_disparity,
                                             r_disp, m_consistency_threshold, true);

        /*
        // Perform L R check
        std::copy( l_disp.data(), l_disp.data() + prod(l_roi.size()),
                   l_disp_f.data() );
        transfer_disparity_mark_invalid( l_disp_f, l_roi.min() - l_exp_roi.min(),
                                         r_disp, r_roi.min() - r_exp_roi.min() );
        ImageView<pixel_type> output = pixel_cast<pixel_type>(l_disp);
        mark_invalid( output, l_disp_f );
        */

        return prerasterize_type(final_disparity,
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
                 float consistency_threshold = 1,
                 int32 max_iterations = 6 ) {
      typedef PatchMatchView<Image1T,Image2T> result_type;
      return result_type( left.impl(), right.impl(), search_region,
                          kernel_size, consistency_threshold,
                          max_iterations);
    }
  }} // vw::stereo

#endif//__VW_STEREO_PATCHMATCH2_H__
