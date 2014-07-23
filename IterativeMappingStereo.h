#ifndef __VW_ITERATIVE_MAPPING_STEREO_H__
#define __VW_ITERATIVE_MAPPING_STEREO_H__

#include <vw/Image/ImageView.h>
#include <vw/Image/ImageViewBase.h>
#include <vw/Image/Manipulation.h>
#include <vw/Stereo/Correlate.h>
#include <vw/Stereo/CostFunctions.h>

namespace vw {
  namespace stereo {

    template <class Image1T, class Image2T, class DispT>
    class IterativeMapStereoView : public ImageViewBase<IterativeMapStereoView<Image1T, Image2T, DispT> > {
      Image1T m_left_image;
      Image2T m_right_image;
      DispT m_disparity_image;
      int m_iterations;

    public:
      typedef PixelMask<Vector2f> pixel_type;
      typedef pixel_type result_type;
      typedef ProceduralPixelAccessor<IterativeMapStereoView> pixel_accessor;

      IterativeMapStereoView( ImageViewBase<Image1T> const& left,
                              ImageViewBase<Image2T> const& right,
                              ImageViewBase<DispT> const& disparity,
                              int iterations ) :
        m_left_image(left.impl()), m_right_image(right.impl()),
        m_disparity_image(disparity.impl()), m_iterations(iterations) {}

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

        // Make a copy of the left and render a transformed right and
        // make a copy of the disparity.
        ImageView<PixelMask<Vector2f> > disparity =
          crop(m_disparity_image, bbox);
        ImageView<float> left = crop(m_left_image, bbox);
        ImageView<float> t_right =
          crop(transform(m_right_image,
                         stereo::DisparityTransform
                         (prerasterize_type(disparity,
                                            -bbox.min().x(), -bbox.min().y(),
                                            cols(), rows()))),
               bbox);

        // Find a delta disparity to refine our polynomial fit disparity map
        ImageView<PixelMask<Vector2i> > delta_disparity =
          stereo::correlate(left, t_right, stereo::NullOperation(),
                            BBox2i(Vector2i(-15,-15), Vector2i(15, 15)),
                            Vector2i(15, 15),
                            stereo::CROSS_CORRELATION, 2);

        // Create a combined disparity and then smooth it again for mapping
        ImageView<pixel_type > combined_disparity = disparity + delta_disparity;

        for (int iterations = m_iterations; iterations > 0; iterations--) {
          fill(disparity, pixel_type(Vector2f()));
          select_channel(disparity, 0) = gaussian_filter(select_channel(combined_disparity,0),2);
          select_channel(disparity, 1) = gaussian_filter(select_channel(combined_disparity,1),2);

          // Do a better warping of the right image to the left
          t_right =
            crop(transform(m_right_image,
                           stereo::DisparityTransform
                           (prerasterize_type(disparity,
                                              -bbox.min().x(), -bbox.min().y(),
                                              cols(), rows()))),
                 bbox);

          // Again calculate a disparity using this newly refined image
          delta_disparity =
            stereo::correlate(left, t_right, stereo::NullOperation(),
                              BBox2i(Vector2i(-iterations,-iterations),
                                     Vector2i(iterations, iterations)),
                              Vector2i(3 + 2 *iterations, 3 + 2 * iterations),
                              stereo::CROSS_CORRELATION, 2);
          combined_disparity = disparity + delta_disparity;
        }


        return prerasterize_type(combined_disparity,
                                 -bbox.min().x(), -bbox.min().y(),
                                 cols(), rows());
      }

      template <class DestT>
      inline void rasterize(DestT const& dest, BBox2i const& bbox) const {
        vw::rasterize(prerasterize(bbox), dest, bbox);
      }
    };

    template <class Image1T, class Image2T, class DispT>
    IterativeMapStereoView<Image1T, Image2T, DispT>
    iterative_mapping_stereo( ImageViewBase<Image1T> const& left,
                              ImageViewBase<Image2T> const& right,
                              ImageViewBase<DispT> const& disparity,
                              int iterations = 1) {
      typedef IterativeMapStereoView<Image1T, Image2T, DispT> result_type;
      return result_type( left.impl(),
                          right.impl(),
                          disparity.impl(), iterations );
    }
  }
}

#endif // __VW_ITERATIVE_MAPPING_STEREO_H__
