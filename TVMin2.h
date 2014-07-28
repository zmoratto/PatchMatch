#ifndef __VW_STEREO_TVMIN2_H__
#define __VW_STEREO_TVMIN2_H__

#include <vw/Math/BBox.h>
#include <vw/Math/Vector.h>
#include <vw/Math/Matrix.h>
#include <vw/Image/ImageMath.h>
#include <vw/Image/ImageView.h>
#include <vw/Image/ImageViewBase.h>
#include <vw/Image/Manipulation.h>

namespace vw {
  namespace stereo {

    void imROF( ImageView<float> const& input,
                float lambda, int iterations,
                ImageView<float> & output );

    template <class ImageT>
    class TVMinView : public ImageViewBase<TVMinView<ImageT> > {
      ImageT m_input;

    public:
      typedef PixelMask<Vector2f> pixel_type;
      typedef pixel_type result_type;
      typedef ProceduralPixelAccessor<TVMinView> pixel_accessor;

      TVMinView( ImageViewBase<ImageT> const& input ) : m_input(input.impl()) {}

      inline int32 cols() const { return m_input.cols(); }
      inline int32 rows() const { return m_input.rows(); }
      inline int32 planes() const { return 1; }

      inline pixel_accessor origin() const { return pixel_accessor( *this, 0, 0 ); }
      inline pixel_type operator()( int32 /*i*/, int32 /*j*/, int32 /*p*/ = 0) const {
        vw_throw( NoImplErr() << "PatchMatchView::operator()(....) has not been implemented." );
        return pixel_type();
      }

      // Block rasterization section that does actual work
      typedef CropView<ImageView<pixel_type> > prerasterize_type;
      inline prerasterize_type prerasterize(BBox2i const& bbox) const {
        BBox2i exp_bbox(0, 0, bbox.width(), bbox.height());
        int expansion = 32;
        exp_bbox.expand(expansion);

        ImageView<Vector2f> copy =
          crop(edge_extend(apply_mask(crop(m_input, bbox))), exp_bbox);
        ImageView<pixel_type> output_mask(copy.cols(),copy.rows());
        fill(output_mask, pixel_type(Vector2f()));

        for (int i = 0; i < 2; i++ ) {
          ImageView<float> output_n = select_channel(copy,i);
          ImageView<float> input_n = output_n;
          imROF(input_n, 100, 10000, output_n);
          select_channel(output_mask,i) = select_channel(output_n,i);
        }

        return prerasterize_type(output_mask,
                                 -bbox.min().x() + expansion,
                                 -bbox.min().y() + expansion,
                                 cols(), rows());
      }

      template <class DestT>
      inline void rasterize(DestT const& dest, BBox2i const& bbox) const {
        vw::rasterize(prerasterize(bbox), dest, bbox);
      }
    };

    template <class ImageT>
    TVMinView<ImageT>
    tvmin_fit(ImageViewBase<ImageT> const& input) {
      typedef TVMinView<ImageT> result_type;
      return result_type(input.impl());
    }

  }
}

#endif
