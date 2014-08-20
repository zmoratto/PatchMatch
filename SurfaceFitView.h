#ifndef __VW_SURFACE_FIT_VIEW_H__
#define __VW_SURFACE_FIT_VIEW_H__

#include <vw/Math/BBox.h>
#include <vw/Math/Vector.h>
#include <vw/Math/Matrix.h>
#include <vw/Image/ImageMath.h>
#include <vw/Image/ImageView.h>
#include <vw/Image/ImageViewBase.h>
#include <vw/Image/Manipulation.h>

namespace vw {
  namespace stereo {

    struct PolynomialSurfaceFit {
      PolynomialSurfaceFit(double observed, double x, double y) :
        observed(observed), x(x), y(y) {}

      template <typename T>
      bool operator()(const T* const polynomial,
                      T* residuals) const {
        residuals[0] = T(observed) -
          (polynomial[0] +
           polynomial[1] * T(x) +
           polynomial[2] * T(x) * T(x) +
           polynomial[3] * T(y) +
           polynomial[4] * T(y) * T(x) +
           polynomial[5] * T(y) * T(x) * T(x) +
           polynomial[6] * T(y) * T(y) +
           polynomial[7] * T(y) * T(y) * T(x) +
           polynomial[8] * T(y) * T(y) * T(x) * T(x)
           );
        return true;
      }

      double observed, x, y;
    };

    class SurfaceFitViewBase {
    protected:
      bool fit_2d_polynomial_surface( ImageView<PixelMask<Vector2i> > const& input,
                                      Matrix3x3* output_h, Matrix3x3* output_v,
                                      Vector2* xscaling, Vector2* yscaling) const;

      // The size of output image will define how big of a render we
      // perform. This function will automatically scale the indices of the
      // output image to be between 0 -> 1 for polynomial evaluation.
      void render_polynomial_surface(Matrix3x3 const& polynomial_coeff,
                                     ImageView<float>* output ) const;
    };

    template <class ImageT>
    class SurfaceFitView : public ImageViewBase<SurfaceFitView<ImageT> >, SurfaceFitViewBase {
      ImageT m_input;

    public:
      typedef PixelMask<Vector2f> pixel_type;
      typedef pixel_type result_type;
      typedef ProceduralPixelAccessor<SurfaceFitView> pixel_accessor;

      SurfaceFitView( ImageViewBase<ImageT> const& input ) : m_input(input.impl()) {}

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
        BBox2i exp_bbox = bbox;
        exp_bbox.expand(16);

        Matrix3x3 polynomial_h, polynomial_v;
        Vector2 xscaling, yscaling;
        ImageView<PixelMask<Vector2i> > copy =
          crop(edge_extend(m_input), exp_bbox);
        ImageView<pixel_type> smoothed_disparity(exp_bbox.width(),
                                                 exp_bbox.height());
        
        bool ans = fit_2d_polynomial_surface(copy,
                                             &polynomial_h, &polynomial_v,
                                             &xscaling, &yscaling);
        if (ans){
          ImageView<float> fitted_h(exp_bbox.width(), exp_bbox.height()),
            fitted_v(exp_bbox.width(), exp_bbox.height());
          render_polynomial_surface(polynomial_h, &fitted_h);
          render_polynomial_surface(polynomial_v, &fitted_v);
          
          fill(smoothed_disparity, pixel_type(Vector2f()));
          select_channel(smoothed_disparity, 0) = fitted_h;
          select_channel(smoothed_disparity, 1) = fitted_v;
        }else{
          // Could not fit a surface, return the original disparity
          smoothed_disparity = copy;
        }
        
        return prerasterize_type(smoothed_disparity,
                                 -exp_bbox.min().x(), -exp_bbox.min().y(),
                                 cols(), rows());
      }

      template <class DestT>
      inline void rasterize(DestT const& dest, BBox2i const& bbox) const {
        vw::rasterize(prerasterize(bbox), dest, bbox);
      }

    };


    template <class ImageT>
    SurfaceFitView<ImageT>
    surface_fit( ImageViewBase<ImageT> const& input ) {
      typedef SurfaceFitView<ImageT> result_type;
      return result_type(input.impl());
    }
  }
}

#endif // __VW_SURFACE_FIT_VIEW_H__
