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
                                float t_col = 0.0784,
                                float t_grad = 0.0157);

    double evaluate_superpixel(ImageView<float> const& a,
                               ImageView<float> const& b,
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
  }
}

#endif  //__VW_STEREO_ARAPDATATERM_H__
