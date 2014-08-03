#ifndef __VW_STEREO_PATCHMATCH2HEISE_H__
#define __VW_STEREO_PATCHMATCH2HEISE_H__

#include <PatchMatch2NCC.h>
#include <TVMin3.h>
#include <SurfaceFitView.h>
#include <vw/Image/MaskViews.h>

#ifdef DEBUG
#include <vw/FileIO.h>
#endif

namespace vw {
  namespace stereo {

    class PMHeiseBase : public PMNCCBase {
    public:
      void evaluate_8_connect_smooth( ImageView<float> const& a,
                                      ImageView<float> const& b,
                                      BBox2i const& a_roi, BBox2i const& b_roi,
                                      ImageView<DispT> const& ba_disparity,
                                      BBox2i const& ba_roi,
                                      ImageView<DispT> const& ab_disparity_smooth,
                                      float theta, // defines how close we need to be to smooth
                                      float lambda, // thumb on the scale to support E_data
                                      ImageView<DispT>& ab_disparity,
                                      ImageView<float>& ab_cost) const;

      void evaluate_disparity_smooth( ImageView<float> const& a, ImageView<float> const& b,
                                      BBox2i const& a_roi, BBox2i const& b_roi,
                                      ImageView<DispT> const& ab_disparity_smooth,
                                      ImageView<DispT> const& ab_disparity,
                                      float theta, // defines how close we need to be to smooth
                                      float lambda, // thumb on the scale to support E_data
                                      ImageView<float>& ab_cost ) const;

      void solve_smooth(ImageView<DispT> const& ab_disparity_noisy,
                        ImageView<float> const& ab_weight,
                        float theta_sigma_d,
                        ImageView<float> & p_x_dx, // Holding buffers for the hidden variable
                        ImageView<float> & p_x_dy,
                        ImageView<float> & p_y_dx,
                        ImageView<float> & p_y_dy,
                        ImageView<DispT> & ab_disparity_smooth) const;

      void solve_gradient_weight(ImageView<float> const& a_exp,
                                 BBox2i const& a_exp_roi,
                                 BBox2i const& a_roi,
                                 ImageView<float> & weight) const;

      void copy_valid_pixels(ImageView<PixelMask<Vector2i> > const& input,
                             ImageView<Vector2i> & output) const;

      PMHeiseBase(BBox2i const& bbox, Vector2i const& kernel,
                  float consistency_threshold, int32 max_iterations) :
        PMNCCBase(bbox, kernel, consistency_threshold, max_iterations) {}
    };

    template <class Image1T, class Image2T>
    class PMHeiseView : public ImageViewBase<PMHeiseView<Image1T, Image2T> >, PMHeiseBase {
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
      typedef PixelMask<DispT> pixel_type;
      typedef pixel_type result_type;
      typedef ProceduralPixelAccessor<PMHeiseView> pixel_accessor;

      // Search region values are inclusive.
      PMHeiseView( ImageViewBase<Image1T> const& left,
                      ImageViewBase<Image2T> const& right,
                      BBox2i const& search_region, Vector2i const& kernel_size,
                      float consistency_threshold = 1,
                      int32 max_iterations = 6) :
        PMHeiseBase(search_region, kernel_size,
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
        vw_throw( NoImplErr() << "PMHeiseView::operator()(....) has not been implemented." );
        return pixel_type();
      }

      // Block rasterization section that does actual work
      typedef CropView<ImageView<pixel_type> > prerasterize_type;
      inline prerasterize_type prerasterize(BBox2i const& bbox) const {
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
        std::ostringstream prefix;
        prefix << bbox.min()[0] << "_" << bbox.min()[1] << "_";
#endif

        // 6. Allocate buffers
        ImageView<float> l_exp( crop( edge_extend(m_left_image), l_exp_roi ) );
        ImageView<float> r_exp( crop( edge_extend(m_right_image), r_exp_roi ) );
        ImageView<DispT>
          l_disp( l_roi.width(), l_roi.height() ), l_disp_cpy( l_roi.width(), l_roi.height() ),
          r_disp( r_roi.width(), r_roi.height() ), r_disp_cpy( r_roi.width(), r_roi.height() ),
          l_disp_smooth( l_roi.width(), l_roi.height() ),
          r_disp_smooth( r_roi.width(), r_roi.height() );
        ImageView<float>
          l_weight( l_roi.width(), l_roi.height() ),
          l_p_x_dx( l_roi.width(), l_roi.height() ),
          l_p_x_dy( l_roi.width(), l_roi.height() ),
          l_p_y_dx( l_roi.width(), l_roi.height() ),
          l_p_y_dy( l_roi.width(), l_roi.height() ),
          r_weight( r_roi.width(), r_roi.height() ),
          r_p_x_dx( r_roi.width(), r_roi.height() ),
          r_p_y_dx( r_roi.width(), r_roi.height() ),
          r_p_x_dy( r_roi.width(), r_roi.height() ),
          r_p_y_dy( r_roi.width(), r_roi.height() );
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

        // Solve for the weights .. ie deweight the smoothness
        // constraint on color transitions.
        solve_gradient_weight(l_exp, l_exp_roi, l_roi, l_weight);
        solve_gradient_weight(r_exp, r_exp_roi, r_roi, r_weight);

#ifdef DEBUG
        write_image(prefix.str() + "l_weight.tif", l_weight);
        write_image(prefix.str() + "r_weight.tif", r_weight);
#endif

        // This is a combination of theta [0, 1] and the disparity
        // sigma_d. The paper says it should vary between 0 and 50 /
        // max_disp.
        float theta = 0;
        // Lambda is essentially how important the data term is over
        // being smooth.
        float lambda = 500;
        float sigma_d = lambda / norm_2(Vector2f(m_search_region.size()));
        std::cout << "Lambda: " << lambda << " Sigma_d: " << sigma_d << std::endl;

        // Initialize the l_p_x, and l_p_y
        stereo::gradient(select_channel(l_disp, 0), l_p_x_dx, l_p_x_dy);
        stereo::gradient(select_channel(l_disp, 1), l_p_y_dx, l_p_y_dy);
        stereo::gradient(select_channel(r_disp, 0), r_p_x_dx, r_p_x_dy);
        stereo::gradient(select_channel(r_disp, 1), r_p_y_dx, r_p_y_dy);

          // Re-evaluate the cost since smooth has been refined.
        evaluate_disparity_smooth(l_exp, r_exp,
                                  l_exp_roi - l_roi.min(),
                                  r_exp_roi - l_roi.min(),
                                  l_disp_smooth, l_disp,
                                  theta, lambda,
                                  l_cost);
#ifndef DISABLE_RL
        evaluate_disparity_smooth(r_exp, l_exp,
                                  r_exp_roi - r_roi.min(),
                                  l_exp_roi - r_roi.min(),
                                  r_disp_smooth, r_disp,
                                  theta, lambda,
                                  r_cost);
#endif
        std::cout << prefix.str() << " starting cost: " << sum_of_pixel_values(l_cost) << std::endl;

        for (int iteration = 0; iteration < m_max_iterations; iteration++ ) {
#ifdef DEBUG
          std::ostringstream iprefix;
          iprefix << prefix.str() << iteration << "_";
#endif

          // Propagate the best
          evaluate_8_connect_smooth(l_exp, r_exp,
                                    l_exp_roi - l_roi.min(),
                                    r_exp_roi - l_roi.min(),
                                    r_disp,
                                    r_roi - l_roi.min(),
                                    l_disp_smooth,
                                    theta, lambda,
                                    l_disp, l_cost);
          evaluate_8_connect_smooth(r_exp, l_exp,
                                    r_exp_roi - r_roi.min(),
                                    l_exp_roi - r_roi.min(),
                                    l_disp,
                                    l_roi - r_roi.min(),
                                    r_disp_smooth,
                                    theta, lambda,
                                    r_disp, r_cost);

#ifdef DEBUG
          std::cout << iprefix.str() << " cost after 8 conn: " << sum_of_pixel_values(l_cost) << std::endl;
          write_image(iprefix.str() + "l-D.tif", pixel_cast<pixel_type>(l_disp));
          write_image(iprefix.str() + "r-D.tif", pixel_cast<pixel_type>(r_disp));
#endif

          // Start creating a filter view of l_disp and r_disp so they
          // can be smoothed without most of the outliers.
          ImageView<pixel_type > l_filtered(l_disp.cols(), l_disp.rows()),
            r_filtered(r_disp.cols(), r_disp.rows());
          cross_corr_consistency_check(l_disp, r_disp,
                                       l_roi, r_roi,
                                       l_filtered);
          cross_corr_consistency_check(r_disp, l_disp,
                                       r_roi, l_roi,
                                       r_filtered);

          // Solve for surface fitted disparity
          ImageView<Vector2i>
            l_filled = pixel_cast<Vector2i>(apply_mask(block_rasterize(stereo::surface_fit(l_filtered),
                                                                       Vector2i(64,64), 1))),
            r_filled = pixel_cast<Vector2i>(apply_mask(block_rasterize(stereo::surface_fit(r_filtered),
                                                                       Vector2i(64,64), 1)));

          // Move back in our actual measurements
          copy_valid_pixels(l_filtered, l_filled);
          copy_valid_pixels(r_filtered, r_filled);

#ifdef DEBUG
          write_image(iprefix.str() + "lfilled-D.tif", pixel_cast<pixel_type >(l_filled));
          write_image(iprefix.str() + "rfilled-D.tif", pixel_cast<pixel_type >(r_filled));
#endif

          // Increase the theta requirement between smooth and
          // non-smooth
          theta += 1.0 / float(m_max_iterations);
          //theta = exp(iteration - m_max_iterations);
          std::cout << "Theta is now: " << theta << std::endl;

          // Perform smoothing step
          solve_smooth(l_filled, l_weight,
                       theta * sigma_d,
                       l_p_x_dx, l_p_x_dy, l_p_y_dx, l_p_y_dy,
                       l_disp_smooth);
          solve_smooth(r_filled, r_weight,
                       theta * sigma_d,
                       r_p_x_dx, r_p_x_dy, r_p_y_dx, r_p_y_dy,
                       r_disp_smooth);

#ifdef DEBUG
          write_image(iprefix.str() + "lsmooth-D.tif", pixel_cast<pixel_type >(l_disp_smooth));
          write_image(iprefix.str() + "rsmooth-D.tif", pixel_cast<pixel_type >(r_disp_smooth));
#endif

          // Re-evaluate the cost since smooth has been refined.
          evaluate_disparity_smooth(l_exp, r_exp,
                                    l_exp_roi - l_roi.min(),
                                    r_exp_roi - l_roi.min(),
                                    l_disp_smooth, l_disp,
                                    theta, lambda,
                                    l_cost);
          evaluate_disparity_smooth(r_exp, l_exp,
                                    r_exp_roi - r_roi.min(),
                                    l_exp_roi - r_roi.min(),
                                    r_disp_smooth, r_disp,
                                    theta, lambda,
                                    r_cost);

          std::cout << iprefix.str() << " cost after smooth: " << sum_of_pixel_values(l_cost) << std::endl;

          { // Add noise
            l_disp_cpy = copy(l_disp);
            r_disp_cpy = copy(r_disp);

            float scaling_size = 1.0 / pow(2.0, iteration + 1);
            Vector2f search_size_half =
              scaling_size * Vector2f(m_search_region.size());

            add_uniform_noise(BBox2f(-search_size_half, search_size_half),
                              m_search_region,
                              r_roi - l_roi.min(), l_disp_cpy);
            add_uniform_noise(BBox2f(-search_size_half, search_size_half),
                              m_search_region_rl,
                              l_roi - r_roi.min(), r_disp_cpy);

            evaluate_disparity_smooth(l_exp, r_exp,
                                      l_exp_roi - l_roi.min(),
                                      r_exp_roi - l_roi.min(),
                                      l_disp_smooth, l_disp_cpy,
                                      theta, lambda,
                                      l_cost_cpy);
            evaluate_disparity_smooth(r_exp, l_exp,
                                      r_exp_roi - r_roi.min(),
                                      l_exp_roi - r_roi.min(),
                                      r_disp_smooth, r_disp_cpy,
                                      theta, lambda,
                                      r_cost_cpy);

            keep_lowest_cost(l_disp_cpy, l_cost_cpy,
                             l_disp, l_cost);
            keep_lowest_cost(r_disp_cpy, r_cost_cpy,
                             r_disp, r_cost);
          }

#ifdef DEBUG
          std::cout << iprefix.str() << " cost after add noise[: " << sum_of_pixel_values(l_cost) << std::endl;
          write_image(iprefix.str() + "ladd-D.tif", pixel_cast<pixel_type>(l_disp));
          write_image(iprefix.str() + "radd-D.tif", pixel_cast<pixel_type>(r_disp));
#endif
        }

        ImageView<pixel_type > final_disparity(l_disp.cols(), l_disp.rows());
        cross_corr_consistency_check(l_disp, r_disp,
                                     l_roi, r_roi,
                                     final_disparity);

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
    PMHeiseView<Image1T,Image2T>
    patch_match_heise( ImageViewBase<Image1T> const& left,
                     ImageViewBase<Image2T> const& right,
                     BBox2i const& search_region, Vector2i const& kernel_size,
                     float consistency_threshold = 1,
                     int32 max_iterations = 6 ) {
      typedef PMHeiseView<Image1T,Image2T> result_type;
      return result_type( left.impl(), right.impl(), search_region,
                          kernel_size, consistency_threshold,
                          max_iterations);
    }

  }
}

#endif //__VW_STEREO_PATCHMATCH2HEISE_H__
