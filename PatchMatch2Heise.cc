#include <PatchMatch2Heise.h>

using namespace vw;

void stereo::PMHeiseBase::evaluate_8_connect_smooth( ImageView<float> const& a,
                                                     ImageView<float> const& b,
                                                     BBox2i const& a_roi, BBox2i const& b_roi,
                                                     ImageView<DispT> const& ba_disparity,
                                                     BBox2i const& ba_roi,
                                                     ImageView<DispT> const& ab_disparity_smooth,
                                                     ImageView<DispT>& ab_disparity,
                                                     ImageView<float>& ab_cost) const {
}

void stereo::PMHeiseBase::evaluate_disparity_smooth( ImageView<float> const& a, ImageView<float> const& b,
                                                     BBox2i const& a_roi, BBox2i const& b_roi,
                                                     ImageView<DispT> const& ab_disparity_smooth,
                                                     ImageView<DispT>& ab_disparity,
                                                     ImageView<float>& ab_cost ) const {
}

void stereo::PMHeiseBase::solve_smooth(ImageView<DispT> const& ab_disparity_noisy,
                                       ImageView<float> const& ab_weight,
                                       ImageView<float> & p_x_dx, // Holding buffers for the hidden variable
                                       ImageView<float> & p_x_dy,
                                       ImageView<float> & p_y_dx,
                                       ImageView<float> & p_y_dy,
                                       ImageView<DispT> & ab_disparity_smooth) const {
}

struct ExponentFunc : public vw::ReturnFixedType<float> {
  float operator()(float const& p) const {
    return exp(p);
  }
};

void stereo::PMHeiseBase::solve_gradient_weight(ImageView<float> const& a_exp,
                                                BBox2i const& a_exp_roi,
                                                BBox2i const& a_roi,
                                                ImageView<float> & weight) const {
  const float constant = 3;
  const float power = 0.8;
  // This is found in section 2.6 of the Heise 2013 paper.
  ImageView<float> delx = derivative_filter(a_exp, 1, 0);
  ImageView<float> dely = derivative_filter(a_exp, 0, 1);
  weight =
    crop(per_pixel_filter(-1 * constant * pow(sqrt(delx*delx + dely * dely), power), ExponentFunc()),
         a_roi - a_exp_roi.min());
}

