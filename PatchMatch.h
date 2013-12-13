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

      typedef typename Image1T::pixel_type Pixel1T;
      typedef typename Image2T::pixel_type Pixel2T;
      typedef Vector2f DispT;

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

      // Propogates Left, Above, and against opposite disparity
      //
      // Remember a and ab_disparity don't have the same dimensions. A has
      // been expanded so we don't have to do an edge extension. The same
      // applies to b and ba_disparity.
      void evaluate_even_iteration( ImageView<Pixel1T> const& a,
                                    ImageView<Pixel2T> const& b,
                                    BBox2i const& a_roi, BBox2i const& b_roi,
                                    ImageView<DispT>& ab_disparity,
                                    ImageView<DispT>& ba_disparity,
                                    ImageView<float>& ab_cost ) const {

        BBox2i b_disp_size = bounding_box(ba_disparity);

        // TODO: This could iterate by pixel accessor using ab_disparity
        for ( int j = 0; j < ab_disparity.rows(); j++ ) {
          // Compare to the left
          for ( int i = 1; i < ab_disparity.cols(); i++ ) {
            Vector2f loc(i,j);

            DispT d_new = ab_disparity(i-i,j);
            float cost_new = calculate_cost( loc, d_new, a, b, a_roi, b_roi );
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
              float cost_new = calculate_cost( loc, d_new, a, b, a_roi, b_roi );
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
              float cost_new = calculate_cost( loc, d_new, a, b, a_roi, b_roi );
              if ( cost_new < ab_cost(i,j) ) {
                ab_cost(i,j) = cost_new;
                ab_disparity(i,j) = d_new;
              }
            }
          }
        }
      }

      // Propogates Right, Below, and against opposite disparity
      void evaluate_odd_iteration( ImageView<Pixel1T> const& a,
                                   ImageView<Pixel2T> const& b,
                                   BBox2i const& a_roi, BBox2i const& b_roi,
                                   ImageView<DispT>& ab_disparity,
                                   ImageView<DispT>& ba_disparity,
                                   ImageView<float>& ab_cost ) const {
        BBox2i b_disp_size = bounding_box(ba_disparity);

        // TODO: This could iterate by pixel accessor using ab_disparity
        for ( int j = ab_disparity.rows()-1; j >= 0; j-- ) {
          // Comparing right
          for ( int i = ab_disparity.cols()-2; i >= 0; i-- ) {
            DispT d_new = ab_disparity(i+1,j);
            float cost_new = calculate_cost( Vector2f(i,j), d_new, a, b, a_roi, b_roi );
            if ( cost_new < ab_cost(i,j) ) {
              ab_cost(i,j) = cost_new;
              ab_disparity(i,j) = d_new;
            }
          }

          // Comparing bottom
          if ( j < ab_disparity.rows()-1 ) {
            for ( int i = ab_disparity.cols()-1; i >= 0; i-- ) {
              DispT d_new = ab_disparity(i,j+1);
              float cost_new = calculate_cost( Vector2f(i,j), d_new, a, b, a_roi, b_roi );
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
              float cost_new = calculate_cost( Vector2f(i,j), d_new, a, b, a_roi, b_roi );
              if ( cost_new < ab_cost(i,j) ) {
                ab_cost(i,j) = cost_new;
                ab_disparity(i,j) = d_new;
              }
            }

          }
        }
      }

      // Evaluates current disparity and writes its cost
      void evaluate_disparity( ImageView<Pixel1T> const& a,
                               ImageView<Pixel2T> const& b,
                               BBox2i const& a_roi, BBox2i const& b_roi,
                               ImageView<DispT>& ab_disparity,
                               ImageView<float>& ab_cost ) const {
        for ( int j = 0; j < ab_disparity.rows(); j++ ) {
          for ( int i = 0; i < ab_disparity.cols(); i++ ) {
            ab_cost(i,j) =
              calculate_cost( Vector2f(i,j),
                              ab_disparity(i,j),
                              a, b, a_roi, b_roi );
          }
        }
      }

      // Evaluates new random search
      template <bool ForwardT>
      void evaluate_new_search( ImageView<Pixel1T> const& a,
                                ImageView<Pixel2T> const& b,
                                BBox2i const& a_roi, BBox2i const& b_roi,
                                int iteration,
                                boost::variate_generator<boost::rand48, boost::random::uniform_01<> >& random_source,
                                ImageView<DispT>& ab_disparity,
                                ImageView<float>& ab_cost ) const {

        Vector2f search_range_size = m_search_size_f / pow(2.0f,iteration);
        search_range_size[0] = std::max(0.5f, search_range_size[0]);
        search_range_size[1] = std::max(0.5f, search_range_size[1]);

        std::cout << search_range_size << std::endl;

        // TODO: This could iterate by pixel accessor using ab_disparity
        for ( int j = 0; j < ab_disparity.rows(); j++ ) {
          for ( int i = 0; i < ab_disparity.cols(); i++ ) {
            DispT d_new = ab_disparity(i,j);
            Vector2f loc(i,j);

            // Add noise
            subvector(d_new,0,2) +=
              elem_prod(Vector2f( random_source()-0.5, random_source()-0.5),
                        search_range_size );

            // Clip Noise to Search Range so we don't exceed prerasters
            if ( ForwardT ) {
              d_new[0] = std::max(0.f,std::min(d_new[0],m_search_size_f[0]));
              d_new[1] = std::max(0.f,std::min(d_new[1],m_search_size_f[1]));
            } else {
              d_new[0] = std::max(-m_search_size_f[0],std::min(0.f,d_new[0]));
              d_new[1] = std::max(-m_search_size_f[1],std::min(0.f,d_new[1]));
            }

            float cost_new = calculate_cost( loc, d_new, a, b, a_roi, b_roi );
            if ( cost_new < ab_cost(i,j) ) {
              ab_cost(i,j) = cost_new;
              ab_disparity(i,j) = d_new;
            }
          }
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

        std::cout << "-------------------------------\n";
        std::cout << bbox << std::endl;

        // 0.) Define the left and right regions
        BBox2i left_region = bbox;
        BBox2i right_region = left_region + m_search_region.min();
        right_region.max() += m_search_size;
        right_region.max() += Vector2i(1,1); // Because search range is inclusive

        // 1.) Expand the left raster region by the kernel size.
        Vector2i half_kernel = m_kernel_size/2;
        BBox2i left_expanded_roi = left_region;
        left_expanded_roi.min() -= half_kernel;
        left_expanded_roi.max() += half_kernel;
        left_expanded_roi.expand( BilinearInterpolation::pixel_buffer );

        // 2.) Calculate the region of the right image that we're using.
        BBox2i right_expanded_roi = right_region;
        right_expanded_roi.min() -= half_kernel;
        right_expanded_roi.max() += half_kernel;
        right_expanded_roi.expand( BilinearInterpolation::pixel_buffer );

        // 3.) Rasterize copies of the input imagery that we
        // need. Allocate space for disparity and cost images.
        ImageView<Pixel1T> left_expanded =
          crop(edge_extend(m_left_image), left_expanded_roi),
          right_expanded =
          crop(edge_extend(m_right_image), right_expanded_roi);
        ImageView<DispT>
          lr_disparity( left_region.width(), left_region.height() ),
          rl_disparity( right_region.width(), right_region.height() );
        ImageView<float> lr_cost(lr_disparity.cols(), lr_disparity.rows()),
          rl_cost( rl_disparity.cols(), rl_disparity.rows() );

        // DEBUG
        write_image( "input-L.tif", left_expanded );
        write_image( "input-R.tif", right_expanded );

        // 4.) Fill disparity with noise
        boost::rand48 gen(std::rand());
        typedef boost::variate_generator<boost::rand48, boost::random::uniform_01<> > vargen_type;
        vargen_type random_source(gen, boost::random::uniform_01<>());
        for (int j = 0; j < lr_disparity.rows(); j++ ) {
          for (int i = 0; i < lr_disparity.cols(); i++ ) {
            subvector(lr_disparity(i,j),0,2) =
              elem_prod(Vector2f(random_source(),random_source()),m_search_size_f);
          }
        }
        for (int j = 0; j < rl_disparity.rows(); j++ ) {
          for (int i = 0; i < rl_disparity.cols(); i++ ) {
            subvector(rl_disparity(i,j),0,2) =
              elem_prod(Vector2f(random_source(),random_source()),-m_search_size_f);
          }
        }
        std::cout << "Search Range Size: " << m_search_size << std::endl;
        std::cout << "Search Region: " << m_search_region << std::endl;
        std::cout << "Left Expanded ROI: " << left_expanded_roi << std::endl;
        std::cout << "Right Expanded ROI: " << right_expanded_roi << std::endl;

        // Left and Right expanded roi need to be transformed to read
        // relative to their objective roi.
        left_expanded_roi -= left_region.min();
        right_expanded_roi -= right_region.min();

        // Evaluate the current disparity guess so that current costs
        // are evaluated
        evaluate_disparity( left_expanded, right_expanded,
                            left_expanded_roi, right_expanded_roi,
                            lr_disparity, lr_cost );
        evaluate_disparity( right_expanded, left_expanded,
                            right_expanded_roi, left_expanded_roi,
                            rl_disparity, rl_cost );

        // Iterate to converge on solution
        for ( int iteration = 0; iteration < 6; iteration++ ) {
          if ( iteration > 0 ) {
            evaluate_new_search<true>( left_expanded, right_expanded,
                                       left_expanded_roi, right_expanded_roi,
                                       iteration, random_source, lr_disparity, lr_cost );
            evaluate_new_search<false>( right_expanded, left_expanded,
                                        right_expanded_roi, left_expanded_roi,
                                        iteration, random_source, rl_disparity, rl_cost );
          }
          if ( iteration % 2 ) {
            evaluate_even_iteration( left_expanded, right_expanded,
                                     left_expanded_roi, right_expanded_roi,
                                     lr_disparity, rl_disparity, lr_cost );
            evaluate_even_iteration( right_expanded, left_expanded,
                                     right_expanded_roi, left_expanded_roi,
                                     rl_disparity, lr_disparity, rl_cost );
          } else {
            evaluate_odd_iteration( left_expanded, right_expanded,
                                    left_expanded_roi, right_expanded_roi,
                                    lr_disparity, rl_disparity, lr_cost );
            evaluate_odd_iteration( right_expanded, left_expanded,
                                    right_expanded_roi, left_expanded_roi,
                                    rl_disparity, lr_disparity, rl_cost );
          }
        }

        // DEBUG
        write_image("lr_disp_view-D.tif", ImageView<pixel_type>(lr_disparity));
        write_image("rl_disp_view-D.tif", ImageView<pixel_type>(rl_disparity));

        ImageView<pixel_type> result = lr_disparity;
        stereo::cross_corr_consistency_check( result,
                                              rl_disparity, m_consistency_threshold, true );
        result += pixel_type(m_search_region.min());
        return prerasterize_type( result, -bbox.min().x(), -bbox.min().y(), cols(), rows() );
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
