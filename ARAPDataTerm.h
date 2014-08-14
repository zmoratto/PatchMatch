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
                               Vector<double, 5> const& surface_dx,
                               Vector<double, 5> const& surface_dy);

    void fit_surface_superpixel(ImageView<PixelMask<Vector2i> > const& a_disp,
                                BBox2i const& a_subpixel,
                                Vector2 const& a_barycenter,
                                Vector<double, 5> & surface_dx,
                                Vector<double, 5> & surface_dy);

    void define_superpixels(ImageView<PixelMask<Vector2i> > const& a_disp,
                            std::vector<std::pair<BBox2i, Vector2> > & superpixels,
                            std::vector<Vector<double, 5> > & superpixel_surfaces_x,
                            std::vector<Vector<double, 5> > & superpixel_surfaces_y);
  }
}

#endif  //__VW_STEREO_ARAPDATATERM_H__
