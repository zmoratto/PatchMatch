#include <iostream>
#include <vw/Core.h>
#include <vw/Image.h>
#include <vw/FileIO.h>
#include <vw/Math/Vector.h>
#include <vw/Math/Matrix.h>
#include <vw/Stereo/PreFilter.h>
#include <vw/Stereo/CorrelationView.h>

#include <ceres/ceres.h>

using namespace vw;

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

void fit_2d_polynomial_surface( ImageView<PixelMask<Vector2i> > const& input,
                                Matrix3x3* output_h, Matrix3x3* output_v,
                                Vector2* xscaling, Vector2* yscaling) {
  // Figure out what our scaling parameters should be
  // output coordinates = scaling.y * (input coordinates + scaling.x)
  xscaling->x() = -double(input.cols()) / 2.0;
  yscaling->x() = -double(input.rows()) / 2.0;
  xscaling->y() = 2.0/double(input.cols());
  yscaling->y() = 2.0/double(input.rows());

  {
    // Build a ceres problem to fit a polynomial robustly
    ceres::Problem problem;
    for (int j = 0; j < input.rows(); j++) {
      for (int i = 0; i < input.cols(); i++ ) {
        if (is_valid(input(i,j)))
          problem.AddResidualBlock
            (new ceres::AutoDiffCostFunction<PolynomialSurfaceFit, 1, 9>
             (new PolynomialSurfaceFit
              (input(i,j)[0],
               (double(i) + xscaling->x()) * xscaling->y(),
               (double(j) + yscaling->x()) * yscaling->y())),
             new ceres::CauchyLoss(1),
             &(*output_h)(0,0));
      }
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 300;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
    std::cout << *output_h << std::endl;
  }

  {
    // Build a ceres problem to fit a polynomial robustly
    ceres::Problem problem;
    for (int j = 0; j < input.rows(); j++) {
      for (int i = 0; i < input.cols(); i++ ) {
        if (is_valid(input(i,j)))
          problem.AddResidualBlock
            (new ceres::AutoDiffCostFunction<PolynomialSurfaceFit, 1, 9>
             (new PolynomialSurfaceFit
              (input(i,j)[1],
               (double(i) + xscaling->x()) * xscaling->y(),
               (double(j) + yscaling->x()) * yscaling->y())),
             new ceres::CauchyLoss(1),
             &(*output_v)(0,0));
      }
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 300;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
    std::cout << *output_v << std::endl;
  }
}

// The size of output image will define how big of a render we
// perform. This function will automatically scale the indices of the
// output image to be between 0 -> 1 for polynomial evaluation.
void render_polynomial_surface(Matrix3x3 const& polynomial_coeff,
                               ImageView<float>* output ) {
  const double* polyarray = &polynomial_coeff(0,0);
  for (int j = 0; j < output->rows(); j++ ) {
    double jn = (double(j) * 2.0 - double(output->rows())) / double(output->rows());
    for (int i = 0; i < output->cols(); i++ ) {
      double in = (double(i) * 2.0 - double(output->cols())) / double(output->cols());
      (*output)(i, j) =
        polyarray[0] +
        polyarray[1] * in +
        polyarray[2] * in * in +
        polyarray[3] * jn +
        polyarray[4] * jn * in +
        polyarray[5] * jn * in * in +
        polyarray[6] * jn * jn +
        polyarray[7] * jn * jn * in +
        polyarray[8] * jn * jn * in * in;
    }
  }
}

int main(int argc, char **argv) {
  std::cout << "Hello!\n";

  // Load up an image
  ImageView<float> input_left, input_right;
  read_image(input_left, "../PatchMatch/arctic/asp_al-L.crop.8.tif");
  read_image(input_right, "../PatchMatch/arctic/asp_al-R.crop.8.tif");

  // Subsample the image twice
  std::vector<float > kernel(5);
  kernel[0] = kernel[4] = 1.0/16.0;
  kernel[1] = kernel[3] = 4.0/16.0;
  kernel[2] = 6.0/16.0;
  ImageView<float> subsample_left, subsample_right;
  subsample_left = subsample(separable_convolution_filter(subsample(separable_convolution_filter(input_left,kernel,kernel),2),kernel,kernel),2);
  subsample_right = subsample(separable_convolution_filter(subsample(separable_convolution_filter(input_right,kernel,kernel),2),kernel,kernel),2);
  write_image("subsampled-l.tif", subsample_left);
  write_image("subsampled-r.tif", subsample_right);

  // Correlate the two images
  BBox2i search_volume(Vector2i(-50,-10), Vector2i(50, 10));
  Vector2i kernel_size(5, 5);
  ImageView<PixelMask<Vector2i> > initial_disparity =
    stereo::correlate(subsample_left, subsample_right, stereo::NullOperation(),
                      search_volume, kernel_size,
                      stereo::CROSS_CORRELATION, 2);
  write_image("initial-d.tif", initial_disparity);

  // Fit a polynomial to the disparity image
  Matrix3x3 polynomial_h, polynomial_v;
  Vector2 xscaling, yscaling;
  fit_2d_polynomial_surface(initial_disparity,
                            &polynomial_h, &polynomial_v,
                            &xscaling, &yscaling);

  // Render the polynomial of the disparity
  ImageView<float> fitted_h(subsample_left.cols(), subsample_left.rows()),
    fitted_v(subsample_left.cols(), subsample_left.rows());
  render_polynomial_surface(polynomial_h, &fitted_h);
  render_polynomial_surface(polynomial_v, &fitted_v);
  write_image("fitted-h.tif", normalize(fitted_h));
  write_image("fitted-v.tif", normalize(fitted_v));

  // Warp the right image to the left image using the disparity
  ImageView<PixelMask<Vector2f> > smoothed_disparity(initial_disparity.cols(), initial_disparity.rows());
  fill(smoothed_disparity, PixelMask<Vector2f>(Vector2f()));
  select_channel(smoothed_disparity, 0) = fitted_h;
  select_channel(smoothed_disparity, 1) = fitted_v;
  std::cout << smoothed_disparity(0,0) << std::endl;
  ImageView<float> transformed_right =
    transform(subsample_right, stereo::DisparityTransform(smoothed_disparity));
  write_image("subsampled-tr.tif", transformed_right);

  // Find delta disparity to refine our polynomial fit disparity map
  ImageView<PixelMask<Vector2i> > delta_disparity =
    stereo::correlate(subsample_left, transformed_right, stereo::NullOperation(),
                      BBox2i(Vector2i(-10, -10), Vector2i(10, 10)), Vector2i(15, 15),
                      stereo::CROSS_CORRELATION, 2);
  write_image("delta-d.tif", delta_disparity);

  // Create combined disparity and then smooth it again.
  ImageView<PixelMask<Vector2f> > combined_disparity = smoothed_disparity + delta_disparity;
  write_image("intermediate-d.tif", combined_disparity);
  fill(smoothed_disparity, PixelMask<Vector2f>(Vector2f()));
  select_channel(smoothed_disparity, 0) = gaussian_filter(select_channel(combined_disparity,0),2);
  select_channel(smoothed_disparity, 1) = gaussian_filter(select_channel(combined_disparity,1),2);
  write_image("combined-smoothed-d.tif", smoothed_disparity);

  // Do a batter warping of the right image to the left
  transformed_right =
    transform(subsample_right, stereo::DisparityTransform(smoothed_disparity));
  write_image("subsampled-tr2.tif", transformed_right);

  // Again .. calculate a disparity using this newly refined image
  delta_disparity =
    stereo::correlate(subsample_left, transformed_right, stereo::NullOperation(),
                      BBox2i(Vector2i(-2, -2), Vector2i(2, 2)), Vector2i(5, 5),
                      stereo::CROSS_CORRELATION, 2);
  write_image("delta2-d.tif", delta_disparity);

  // Generate the final disparity
  combined_disparity = smoothed_disparity + delta_disparity;
  write_image("final-d.tif", combined_disparity);

  // Generate a final warping
  transformed_right =
    transform(subsample_right, stereo::DisparityTransform(combined_disparity));
  write_image("subsampled-tr3.tif", transformed_right);

  return 0;
}
