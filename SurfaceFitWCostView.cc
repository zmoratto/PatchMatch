#include <SurfaceFitWCostView.h>

#include <NelderMead.h>

#include <vw/Math/BBox.h>
#include <vw/Image/Transform.h>
#include <vw/Image/ImageMath.h>
#include <vw/Image/Statistics.h>
#include <vw/Stereo/DisparityMap.h>
#include <ceres/ceres.h>

using namespace vw;

struct Polynomial2DSurfaceFit {
  Polynomial2DSurfaceFit(double obs_dx, double obs_dy,
                         double x, double y) :
    obs_dx_(obs_dx), obs_dy_(obs_dy), x_(x), y_(y), xx_(x*x), yy_(y*y) {}

  template <typename T>
  bool operator()(const T* const polynomial_dx,
                  const T* const polynomial_dy,
                  T* residuals) const {
    residuals[0] = T(obs_dx_) -
      (polynomial_dx[0] +
       polynomial_dx[1] * T(x_) +
       polynomial_dx[2] * T(xx_) +
       polynomial_dx[3] * T(y_) +
       polynomial_dx[4] * T(y_) * T(x_) +
       polynomial_dx[5] * T(y_) * T(xx_) +
       polynomial_dx[6] * T(yy_) +
       polynomial_dx[7] * T(yy_) * T(x_) +
       polynomial_dx[8] * T(yy_) * T(xx_)
       );
    residuals[1] = T(obs_dy_) -
      (polynomial_dy[0] +
       polynomial_dy[1] * T(x_) +
       polynomial_dy[2] * T(xx_) +
       polynomial_dy[3] * T(y_) +
       polynomial_dy[4] * T(y_) * T(x_) +
       polynomial_dy[5] * T(y_) * T(xx_) +
       polynomial_dy[6] * T(yy_) +
       polynomial_dy[7] * T(yy_) * T(x_) +
       polynomial_dy[8] * T(yy_) * T(xx_)
       );

    return true;
  }

  double obs_dx_, obs_dy_, x_, y_, xx_, yy_;
};

void fit_surface_superpixel(ImageView<PixelMask<Vector2i> > const& a_disp,
                            BBox2i const& a_subpixel,
                            Vector2 const& a_barycenter,
                            Vector<double, 18> & surface) {
  ceres::Problem problem;
  for (int j = a_subpixel.min()[1]; j < a_subpixel.max()[1]; j++) {
    for (int i = a_subpixel.min()[0]; i < a_subpixel.max()[0]; i++) {
      if (is_valid(a_disp(i,j))) {
        problem.AddResidualBlock
          (new ceres::AutoDiffCostFunction<Polynomial2DSurfaceFit, 2, 9, 9>
           (new Polynomial2DSurfaceFit
            (a_disp(i,j).child()[0], a_disp(i,j).child()[1],
             double(i) - a_barycenter[0],
             double(j) - a_barycenter[1])),
           new ceres::CauchyLoss(3),
           &surface[0], &surface[9]);
      }
    }
  }

  ceres::Solver::Options options;
  options.max_num_iterations = 300;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  if (summary.termination_type == ceres::NO_CONVERGENCE) {
    std::fill(surface.begin(), surface.end(), 0);
  }
}

void define_superpixels(ImageView<PixelMask<Vector2i> > const& a_disp,
                        std::vector<std::pair<BBox2i, Vector2> > & superpixels,
                        std::vector<Vector<double, 18> > & superpixel_surfaces) {
  superpixel_surfaces.resize(superpixels.size());

  for (size_t i = 0; i < superpixels.size(); i++ ) {
    fit_surface_superpixel(a_disp,
                           superpixels[i].first,
                           superpixels[i].second,
                           superpixel_surfaces[i]);
  }
}

class DisparityQuadSurfaceTransform : public TransformBase<DisparityQuadSurfaceTransform> {
  Vector<double, 18> const& surface;
  Vector2 center;
public:
  DisparityQuadSurfaceTransform(Vector<double, 18> const& s,
                                Vector2 const& c) :
    surface(s), center(c) {}

  inline Vector2 reverse(const Vector2 &p ) const {
    // Give a destination pixel ... return the pixel that should be the source
    Vector2 dp = p - center;
    Vector2 dp2 = elem_prod(dp, dp);
    return p +
      Vector2(surface[0] +
              surface[1] * dp.x() +
              surface[2] * dp2.x() +
              surface[3] * dp.y() +
              surface[4] * dp.y() * dp.x() +
              surface[5] * dp.y() * dp2.x() +
              surface[6] * dp2.y() +
              surface[7] * dp2.y() * dp.x() +
              surface[8] * dp2.y() * dp2.x(),
              surface[9] +
              surface[10] * dp.x() +
              surface[11] * dp2.x() +
              surface[12] * dp.y() +
              surface[13] * dp.y() * dp.x() +
              surface[14] * dp.y() * dp2.x() +
              surface[15] * dp2.y() +
              surface[16] * dp2.y() * dp.x() +
              surface[17] * dp2.y() * dp2.x());
  }
};

struct NCCQuadraticFunctor {
  ImageView<float> const& left, right;
  std::pair<BBox2i, Vector2> const& superpixel;

  NCCQuadraticFunctor(ImageView<float> const& a,
                      ImageView<float> const& b,
                      std::pair<BBox2i, Vector2> const& s ) :
    left(a), right(b), superpixel(s) {}

  double operator()(Vector<double, 18> const& surface) const {
    ImageView<float> left_kernel = crop(left, superpixel.first);
    ImageView<float> right_kernel =
      crop(transform(right, DisparityQuadSurfaceTransform(surface, superpixel.second)),
           superpixel.first);

    float cov_lr =
      sum_of_pixel_values
      (left_kernel * right_kernel);
    float cov_ll =
      sum_of_pixel_values
      (left_kernel * left_kernel);
    float cov_rr =
      sum_of_pixel_values
      (right_kernel * right_kernel);

    // NCC Cost function here
    return 1 - cov_lr / sqrt(cov_ll * cov_rr);
  }
};

void
vw::stereo::SurfaceFitWCost(ImageView<PixelMask<Vector2f> > surface,
                            ImageView<float> left, ImageView<float> right) {
  // Define our super pixels
  std::vector<BBox2i> box_vec =
    image_blocks(surface, 64, 64);
  std::vector<std::pair<BBox2i, Vector2> > superpixels;
  superpixels.reserve(box_vec.size());
  for (std::vector<BBox2i>::iterator it =
         box_vec.begin(); it != box_vec.end(); it++) {
    superpixels.push_back
      (std::make_pair
       (*it,
        Vector2(it->min()) +
        Vector2(it->size()) / 2));
  }
  std::cout << "Number of superpixels: "
            << superpixels.size() << std::endl;

  std::vector<Vector<double, 18> > superpixel_surfaces;
  define_superpixels(surface,
                     superpixels,
                     superpixel_surfaces);

  // Look through the fitted surfaces and identify ones that seem to
  // be obvious outliers.
  BBox2i disp_range = stereo::get_disparity_range(surface);
  for (int s = 0; s < superpixels.size(); s++ ) {
    if (!disp_range.contains(Vector2i(superpixel_surfaces[s][0],
                                      superpixel_surfaces[s][9]))) {
      std::cout << "Zeroing " << s << " " << superpixel_surfaces[s] << std::endl;
      std::fill(superpixel_surfaces[s].begin(),
                superpixel_surfaces[s].end(), 0);
    }
  }

  // Iterate through the fitted surfaces and refine them so that they reduce a general NCC cost
  int width = surface.cols() / 64;
  for (int s = 0; s < superpixels.size(); s++ ) {
    std::cout << s << std::endl;
    NCCQuadraticFunctor functor(left, right,
                                superpixels[s]);
    std::cout << "starting cost: " << functor(superpixel_surfaces[s]) << std::endl;
    std::cout << superpixel_surfaces[s] << std::endl;

    Vector<double, 18> seeds[19];
    int write_index = 0;
    seeds[write_index++] = superpixel_surfaces[s];
    if (s > 0) {
      seeds[write_index++] = superpixel_surfaces[s-1];
    }
    if (s < superpixel_surfaces.size() - 1) {
      seeds[write_index++] = superpixel_surfaces[s+1];
    }
    if (s >= width + 1) {
      seeds[write_index++] = superpixel_surfaces[s - width - 1];
    }
    if (s >= width) {
      seeds[write_index++] = superpixel_surfaces[s - width];
    }
    if (s >= width - 1) {
      seeds[write_index++] = superpixel_surfaces[s - width + 1];
    }
    if (s < superpixel_surfaces.size() - width) {
      seeds[write_index++] = superpixel_surfaces[s + width - 1];
    }
    if (s < superpixel_surfaces.size() - width - 1) {
      seeds[write_index++] = superpixel_surfaces[s + width];
    }
    if (s < superpixel_surfaces.size() - width - 2) {
      seeds[write_index++] = superpixel_surfaces[s + width + 1];
    }
    if (s > 1) {
      seeds[write_index++] = superpixel_surfaces[s-2];
    }
    if (s < superpixel_surfaces.size() - 2) {
      seeds[write_index++] = superpixel_surfaces[s+2];
    }
    if (s >= 2 * width) {
      seeds[write_index++] = superpixel_surfaces[s - 2 * width];
    }
    if (s < superpixel_surfaces.size() - 2 * width - 1) {
      seeds[write_index++] = superpixel_surfaces[s + 2 * width];
    }
    // Minimium, this fills in 5 elements .. worse case it fills in 14 elements
    while (write_index < 19) {
      seeds[write_index] = superpixel_surfaces[s];
      seeds[write_index][18 - write_index] += 0.1;
      write_index++;
    }

    stereo::Amoeba<18> amoeba(1e-4);
    superpixel_surfaces[s] =
      amoeba.minimize(seeds, functor);

    std::cout << "ending cost: " << functor(superpixel_surfaces[s]) << std::endl;
    std::cout << superpixel_surfaces[s] << std::endl;
  }

  // Render back out to our input so that it has our surface fit
  fill(surface, PixelMask<Vector2f>(Vector2f()));
  for (size_t s = 0; s < superpixel_surfaces.size(); s++ ) {
    for (int j = superpixels[s].first.min()[1];
         j < superpixels[s].first.max()[1]; j++ ) {
      for (int i = superpixels[s].first.min()[0];
           i < superpixels[s].first.max()[0]; i++ ) {
        Vector2 dp = Vector2(i,j) - superpixels[s].second;
        Vector2 dp2 = elem_prod(dp, dp);
        surface(i, j)[0] =
          superpixel_surfaces[s][0] +
          superpixel_surfaces[s][1] * dp.x() +
          superpixel_surfaces[s][2] * dp2.x() +
          superpixel_surfaces[s][3] * dp.y() +
          superpixel_surfaces[s][4] * dp.y() * dp.x() +
          superpixel_surfaces[s][5] * dp.y() * dp2.x() +
          superpixel_surfaces[s][6] * dp2.y() +
          superpixel_surfaces[s][7] * dp2.y() * dp.x() +
          superpixel_surfaces[s][8] * dp2.y() * dp2.x();
        surface(i, j)[1] =
          superpixel_surfaces[s][9] +
          superpixel_surfaces[s][10] * dp.x() +
          superpixel_surfaces[s][11] * dp2.x() +
          superpixel_surfaces[s][12] * dp.y() +
          superpixel_surfaces[s][13] * dp.y() * dp.x() +
          superpixel_surfaces[s][14] * dp.y() * dp2.x() +
          superpixel_surfaces[s][15] * dp2.y() +
          superpixel_surfaces[s][16] * dp2.y() * dp.x() +
          superpixel_surfaces[s][17] * dp2.y() * dp2.x();
      }
    }
  }
}
