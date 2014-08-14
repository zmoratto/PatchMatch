#include <ARAPDataTerm.h>

#include <vw/Image/Filter.h>
#include <vw/Image/ImageView.h>
#include <vw/Image/Transform.h>

#include <ceres/ceres.h>

using namespace vw;

double vw::stereo::gradient_cost_metric(ImageView<float> const& a,
                                        ImageView<float> const& b,
                                        float alpha, float t_col, float t_grad ) {
  ImageView<float> agrad =
    sqrt(square(derivative_filter(a, 1, 0)) + square(derivative_filter(a, 0, 1)));
  ImageView<float> bgrad =
    sqrt(square(derivative_filter(b, 1, 0)) + square(derivative_filter(b, 0, 1)));

  double sum_of_error = 0;
  float nalpha = 1.0 - alpha;
  for (int j = 0; j < a.rows(); j++) {
    for (int i = 0; i < a.cols(); i++) {
      sum_of_error +=
        nalpha * std::min(fabsf(a(i,j) - b(i,j)), t_col) +
        alpha * std::min(fabsf(agrad(i,j) - bgrad(i,j)), t_grad);
    }
  }

  return sum_of_error;
}

class DisparitySurfaceTransform : public TransformBase<DisparitySurfaceTransform> {
  Vector<double, 5> const& surface_dx, surface_dy;
  Vector2 center;
public:
  DisparitySurfaceTransform(Vector<double, 5> const& dx,
                            Vector<double, 5> const& dy,
                            Vector2 const& c) :
    surface_dx(dx), surface_dy(dy), center(c) {}

  inline Vector2 reverse(const Vector2 &p ) const {
    // Give a destination pixel ... return the pixel that should be the source
    Vector2 dp = p - center;
    Vector2 dp2 = elem_prod(dp, dp);
    return p +
      Vector2(surface_dx[0] +
              surface_dx[1] * dp[0] +
              surface_dx[2] * dp[1] +
              surface_dx[3] * dp2[0] +
              surface_dx[4] * dp2[1],
              surface_dy[0] +
              surface_dy[1] * dp[0] +
              surface_dy[2] * dp[1] +
              surface_dy[3] * dp2[0] +
              surface_dy[4] * dp2[1]);
  }
};

double vw::stereo::evaluate_superpixel(ImageView<float> const& a,
                                       ImageView<float> const& b,
                                       BBox2i const& a_superpixel,
                                       Vector2 const& a_barycenter,
                                       Vector<double, 5> const& surface_dx,
                                       Vector<double, 5> const& surface_dy) {
  ImageView<float> a_crop = crop(a, a_superpixel);
  ImageView<float> b_crop =
    crop(transform(b, DisparitySurfaceTransform(surface_dx,
                                                surface_dy,
                                                a_barycenter)),
         a_superpixel);
  return stereo::gradient_cost_metric(a_crop, b_crop);
}

struct QuadraticSurfaceFit {
  QuadraticSurfaceFit(double obs_dx, double obs_dy,
                      double x, double y) :
    obs_dx_(obs_dx), obs_dy_(obs_dy), x_(x), y_(y), xx_(x*x), yy_(y*y) {}

  template <typename T>
  bool operator()(const T* const polynomial_dx,
                  const T* const polynomial_dy,
                  T* residuals) const {
    residuals[0] = T(obs_dx_) -
      (polynomial_dx[0] +
       polynomial_dx[1] * T(x_) +
       polynomial_dx[2] * T(y_) +
       polynomial_dx[3] * T(xx_) +
       polynomial_dx[4] * T(yy_)
       );
    residuals[1] = T(obs_dy_) -
      (polynomial_dy[0] +
       polynomial_dy[1] * T(x_) +
       polynomial_dy[2] * T(y_) +
       polynomial_dy[3] * T(xx_) +
       polynomial_dy[4] * T(yy_)
       );
    return true;
  }

  double obs_dx_, obs_dy_, x_, y_, xx_, yy_;
};

void vw::stereo::fit_surface_superpixel(ImageView<PixelMask<Vector2i> > const& a_disp,
                                        BBox2i const& a_subpixel,
                                        Vector2 const& a_barycenter,
                                        Vector<double, 5> & surface_dx,
                                        Vector<double, 5> & surface_dy) {
  ceres::Problem problem;
  for (int j = a_subpixel.min()[1]; j < a_subpixel.max()[1]; j+=2) {
    for (int i = a_subpixel.min()[0]; i < a_subpixel.max()[0]; i++) {
      if (is_valid(a_disp(i,j))) {
        problem.AddResidualBlock
          (new ceres::AutoDiffCostFunction<QuadraticSurfaceFit, 2, 5, 5>
           (new QuadraticSurfaceFit
            (a_disp(i,j).child()[0], a_disp(i,j).child()[1],
             double(i) - a_barycenter[0],
             double(j) - a_barycenter[1])),
           new ceres::CauchyLoss(3),
           &surface_dx[0], &surface_dy[0]);
      }
    }
  }

  ceres::Solver::Options options;
  options.max_num_iterations = 300;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
}

void vw::stereo::define_superpixels(ImageView<PixelMask<Vector2i> > const& a_disp,
                                    std::vector<std::pair<BBox2i, Vector2> > & superpixels,
                                    std::vector<Vector<double, 5> > & superpixel_surfaces_x,
                                    std::vector<Vector<double, 5> > & superpixel_surfaces_y) {
  superpixel_surfaces_x.resize(superpixels.size());
  superpixel_surfaces_y.resize(superpixels.size());

  for (size_t i = 0; i < superpixels.size(); i++ ) {
    fit_surface_superpixel(a_disp,
                           superpixels[i].first,
                           superpixels[i].second,
                           superpixel_surfaces_x[i],
                           superpixel_surfaces_y[i]);
  }
}
