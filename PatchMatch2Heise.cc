#include <PatchMatch2Heise.h>
#include <TVMin3.h>

using namespace vw;

void stereo::PMHeiseBase::evaluate_8_connect_smooth( ImageView<float> const& a,
                                                     ImageView<float> const& b,
                                                     BBox2i const& a_roi, BBox2i const& b_roi,
                                                     ImageView<DispT> const& ba_disparity,
                                                     BBox2i const& ba_roi,
                                                     ImageView<DispT> const& ab_disparity_smooth,
                                                     float theta, // defines how close we need to be to smooth
                                                     float lambda, // thumb on the scale to support E_data
                                                     ImageView<DispT>& ab_disparity,
                                                     ImageView<float>& ab_cost) const {
  float cost;
  DispT d_new;
  for ( int j = 0; j < ab_disparity.rows(); j++ ) {
    for ( int i = 0; i < ab_disparity.cols(); i++ ) {
      DispT loc(i,j);

      float curr_best_cost = ab_cost(i,j);
      DispT curr_best_d = ab_disparity(i, j);
      DispT starting_d = ab_disparity(i, j);

#define EVALUATE_AND_KEEP_BEST                                          \
      if ( ba_roi.contains(d_new + loc) && curr_best_d != d_new && starting_d != d_new) { \
        cost =                                                          \
          lambda * calculate_cost(loc, d_new, a, b, a_roi, b_roi,       \
                                  m_kernel_roi) +                       \
          (theta / 2.0) * norm_2_sqr(Vector2f(d_new) - Vector2f(ab_disparity_smooth(i,j))); \
        if (cost < curr_best_cost) {                                    \
          curr_best_cost = cost;                                        \
          curr_best_d = d_new;                                          \
        }                                                               \
      }


      if ( i > 0 ) {
        // Compare left
        d_new = ab_disparity(i-1,j);
        EVALUATE_AND_KEEP_BEST;
      }
      if ( j > 0 ) {
        // Compare up
        d_new = ab_disparity(i,j-1);
        EVALUATE_AND_KEEP_BEST;
      }
      if ( i < ab_disparity.cols() - 1) {
        // Compare right
        d_new = ab_disparity(i+1,j);
        EVALUATE_AND_KEEP_BEST;
      }
      if ( j < ab_disparity.rows() - 1 ) {
        // Compare lower
        d_new = ab_disparity(i,j+1);
        EVALUATE_AND_KEEP_BEST;
      }
      {
        // Compare LR alternative
        DispT d = ab_disparity(i,j);
        d_new = -ba_disparity(i + d[0] - ba_roi.min().x(),
                              j + d[1] - ba_roi.min().y());
        EVALUATE_AND_KEEP_BEST;
      }

      ab_cost(i,j) = curr_best_cost;
      ab_disparity(i,j) = curr_best_d;
  }
}
#undef EVALUATE_AND_KEEP_BEST
}

void stereo::PMHeiseBase::evaluate_disparity_smooth( ImageView<float> const& a, ImageView<float> const& b,
                                                       BBox2i const& a_roi, BBox2i const& b_roi,
                                                       ImageView<DispT> const& ab_disparity_smooth,
                                                       ImageView<DispT> const& ab_disparity,
                                                       float theta, // defines how close we need to be to smooth
                                                       float lambda, // thumb on the scale to support E_data
                                                       ImageView<float>& ab_cost ) const {
  typedef ImageView<float>::pixel_accessor CAccT;
  typedef ImageView<DispT>::pixel_accessor DAccT;
  CAccT cost_row = ab_cost.origin();
  DAccT disp_row = ab_disparity.origin();
  Vector2i loc;
  for ( loc[1] = 0; loc[1] < ab_disparity.rows(); loc[1]++ ) {
    CAccT cost_col = cost_row;
    DAccT disp_col = disp_row;
    for ( loc[0] = 0; loc[0] < ab_disparity.cols(); loc[0]++ ) {
      *cost_col =
        lambda * calculate_cost( loc, *disp_col,
                                 a, b, a_roi, b_roi, m_kernel_roi) +
        (theta / 2.0) * norm_2_sqr(Vector2f(*disp_col) - Vector2f(ab_disparity_smooth(loc[0],loc[1])));
      cost_col.next_col();
      disp_col.next_col();
    }
    cost_row.next_row();
    disp_row.next_row();
  }
}

float calc_energy( ImageView<float> const& input,
                   ImageView<float> const& ref,
                   ImageView<float> const& weight,
                   float theta_sigma_d) {
  ImageView<float> dx, dy;
  stereo::gradient(input, dx, dy);
  float e_reg = sum_of_pixel_values(weight * sqrt(dx * dx + dy * dy));
  float e_data = 0.5 * theta_sigma_d * sum_of_pixel_values((input - ref)*(input -ref));
  return e_reg + e_data;
}

void PMHuberROF( ImageView<float> const& input,
                   ImageView<float> const& weight,
                   int iterations,
                   float alpha, // Huber threshold coeff,
                   float sigma, float tau, // Gradient step sizes
                   float theta_sigma_d, // Essentially lambda
                   ImageView<float> & p_x,
                   ImageView<float> & p_y,
                   ImageView<float> & output ) {
  // Allocate space for p, our hidden variable and u our output.
  ImageView<float> grad_u_x, grad_u_y;
  ImageView<float> div_p;
  for ( int i = 0; i < iterations; i++ ) {

    // Eqn 29
    stereo::gradient(output, grad_u_x, grad_u_y);
    p_x += sigma * weight * grad_u_x;
    p_y += sigma * weight * grad_u_y;
    p_x /= (1 + sigma * alpha * (1 / weight));
    p_y /= (1 + sigma * alpha * (1 / weight));

    // Eqn 29
    for (int j = 0; j < p_x.rows(); j++ ) {
      for (int i = 0; i < p_x.cols(); i++ ) {
        float mag =
          std::max(1.0, sqrt(p_x(i,j)*p_x(i,j) +
                             p_y(i,j)*p_y(i,j)));
        p_x(i,j) /= mag;
        p_y(i,j) /= mag;
      }
    }

    // Eqn 30
    stereo::divergence(p_x, p_y, div_p);
    output += tau * (theta_sigma_d * input + weight * div_p);
    output /= (1 + tau * theta_sigma_d);

    // DEBUG, determine are we actually reducing our own cost?
    if (!(i % 10)) {
      std::cout << i << " -> energy -> " << calc_energy(output, input, weight, theta_sigma_d) << std::endl;
    }
  }
}

void stereo::PMHeiseBase::solve_smooth(ImageView<DispT> const& ab_disparity_noisy,
                                         ImageView<float> const& ab_weight,
                                         float theta_sigma_d,
                                         ImageView<float> & p_x_dx, // Holding buffers for the hidden variable
                                         ImageView<float> & p_x_dy,
                                         ImageView<float> & p_y_dx,
                                         ImageView<float> & p_y_dy,
                                         ImageView<DispT> & ab_disparity_smooth) const {
  const float L2 = 8.0;
  const float tau = 0.04;
  const float sigma = 1.0 / (L2 * tau);
  const float huber_coeff = 0.001;

  // This implements equations 29 and 30 in the heise paper
  //  const int MAX_ITERATIONS = 101;
  const int MAX_ITERATIONS = 41;

  ImageView<float> buffer0(ab_disparity_noisy.cols(),
                           ab_disparity_noisy.rows()),
    buffer1(ab_disparity_noisy.cols(),
            ab_disparity_noisy.rows());

  // Solve for smooth x disparity
  buffer0 = select_channel(ab_disparity_noisy, 0);
  buffer1 = select_channel(ab_disparity_smooth, 0);
  PMHuberROF( buffer0, ab_weight, MAX_ITERATIONS,
              huber_coeff, sigma, tau, theta_sigma_d,
              p_x_dx, p_x_dy, buffer1);
  select_channel(ab_disparity_smooth, 0) = buffer1;

  // Solve for smooth y disparity
  buffer0 = select_channel(ab_disparity_noisy, 1);
  buffer1 = select_channel(ab_disparity_smooth, 1);
  PMHuberROF( buffer0, ab_weight, MAX_ITERATIONS,
              huber_coeff, sigma, tau, theta_sigma_d,
              p_y_dx, p_y_dy, buffer1);
  select_channel(ab_disparity_smooth, 1) = buffer1;
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

void stereo::PMHeiseBase::copy_valid_pixels(ImageView<PixelMask<Vector2i> > const& input,
                                            ImageView<Vector2i> & output) const {
  for (int j = 0; j < input.rows(); j++ ) {
    for (int i = 0; i < input.cols(); i++ ) {
      if (is_valid(input(i,j))) {
        output(i,j) = input(i,j).child();
      }
    }
  }
}
