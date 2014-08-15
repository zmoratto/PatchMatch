#ifndef __VW_STEREO_ARAPDATATERM_H__
#define __VW_STEREO_ARAPDATATERM_H__

#include <vw/Image/ImageView.h>
#include <vw/Math/Vector.h>
#include <vw/Math/BBox.h>

namespace vw {
  namespace stereo {
    double gradient_cost_metric(ImageView<float> const& a,
                                ImageView<float> const& b,
                                float alpha = 0.85, // These constants come from the eccv14 paper
                                float t_col = 1,
                                float t_grad = 1);
                                //                                float t_col = 0.0784,
                                //float t_grad = 0.0157);

    double evaluate_superpixel(ImageView<float> const& a,
                               ImageView<float> const& b,
                               BBox2i const& a_superpixel,
                               Vector2 const& a_barycenter,
                               Vector<double, 10> const& surface);

    double evaluate_intermediate_term(double theta,
                                      ImageView<Vector2f> const& u,
                                      BBox2i const& a_superpixel,
                                      Vector2 const& a_barycenter,
                                      Vector<double, 10> const& surface);

    void fit_surface_superpixel(ImageView<PixelMask<Vector2i> > const& a_disp,
                                BBox2i const& a_subpixel,
                                Vector2 const& a_barycenter,
                                Vector<double, 10> & surface);

    void define_superpixels(ImageView<PixelMask<Vector2i> > const& a_disp,
                            std::vector<std::pair<BBox2i, Vector2> > & superpixels,
                            std::vector<Vector<double, 10> > & superpixel_surfaces);

    void render_disparity_image(std::vector<std::pair<BBox2i, Vector2> > const& superpixels,
                                std::vector<Vector<double, 10> > const& superpixel_surfaces,
                                ImageView<PixelMask<Vector2f> > & disp);

    struct IndiceFinder {
      int width_, num_indices_;
      IndiceFinder(int w, int num) : width_(w), num_indices_(num) {}

      inline int l(int s) {
        return std::max(0, s - 1);
      }

      inline int r(int s) {
        return std::min(num_indices_, s + 1);
      }

      inline int tl(int s) {
        return std::max(0, s - 1 - width_);
      }

      inline int t(int s) {
        return std::max(0, s - width_);
      }

      inline int tr(int s) {
        return std::max(0, s - width_ + 1);
      }

      inline int bl(int s) {
        return std::min(num_indices_, s + width_ - 1);
      }

      inline int b(int s) {
        return std::min(num_indices_, s + width_);
      }

      inline int br(int s) {
        return std::min(num_indices_, s + width_ + 1);
      }
    };

    struct NMFunctor {
      ImageView<float> const& left, right;
      ImageView<Vector2f> const& u;
      std::pair<BBox2i, Vector2> const& superpixel;
      double theta;

      NMFunctor(ImageView<float> const& a,
                ImageView<float> const& b,
                ImageView<Vector2f> const& u,
                std::pair<BBox2i, Vector2> const& s,
                double t) :
        left(a), right(b), u(u), superpixel(s), theta(t) {}

      double operator()(Vector<double, 10> const& surface) const {
        return stereo::evaluate_superpixel(left, right,
                                           superpixel.first,
                                           superpixel.second,
                                           surface) +
          stereo::evaluate_intermediate_term(theta, u,
                                             superpixel.first,
                                             superpixel.second,
                                             surface);
      }
    };
  }
}

#endif  //__VW_STEREO_ARAPDATATERM_H__
