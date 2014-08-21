#include <vw/Image/ImageView.h>
#include <vw/Image/Filter.h>
#include <vw/Math/BBox.h>
#include <vw/FileIO.h>
#include <vw/Stereo/DisparityMap.h>

#include <iomanip>

using namespace vw;

int main(int argc, char **argv) {
  // DiskImageView<float>
  //   left_disk_image("arctic/asp_al-L.crop.16.tif"),
  //   right_disk_image("arctic/asp_al-R.crop.16.tif");
  DiskImageView<PixelGray<float> >
    left_disk_image("../SemiGlobalMatching/data/cones/im2.png"),
    right_disk_image("../SemiGlobalMatching/data/cones/im6.png");
  ImageView<float> left_image = left_disk_image, right_image = right_disk_image;
  write_image("left_image.tif", left_image);
  // BBox2i search_region(Vector2i(-70,-25),
  //                      Vector2i(105,46));
  BBox2i search_region(Vector2i(-64, -1), Vector2i(0, 2));
  std::cout << search_region << std::endl;

  const float GAUSS_SIGMA = 4;

  // The are a couple things that we keep around for a long time
  // ... like the mean and the gradient filters.
  ImageView<float>
    left_gradient_x = derivative_filter(left_image, 1, 0),
    right_gradient_x = derivative_filter(right_image, 1, 0),
    left_gradient_y = derivative_filter(left_image, 0, 1),
    right_gradient_y = derivative_filter(right_image, 0, 1);
  ImageView<float> left_mean = gaussian_filter(left_image, GAUSS_SIGMA);
  ImageView<float> left_corr = gaussian_filter(left_image * left_image, GAUSS_SIGMA);

  BBox2i roi = bounding_box(left_image);
  BBox2i roi_expanded = roi;
  roi_expanded.min() += search_region.min();
  roi_expanded.max() += search_region.max();

  ImageView<float> right_image_exp = crop(edge_extend(right_image, ConstantEdgeExtension()), roi_expanded);
  ImageView<float> right_gradient_x_exp = crop(edge_extend(right_gradient_x, ZeroEdgeExtension()), roi_expanded);
  ImageView<float> right_gradient_y_exp = crop(edge_extend(right_gradient_y, ZeroEdgeExtension()), roi_expanded);

  ImageView<float> right_mean_exp = gaussian_filter(right_image_exp, GAUSS_SIGMA);
  ImageView<float> right_corr_exp = gaussian_filter(right_image_exp * right_image_exp, GAUSS_SIGMA);

  ImageView<Vector2i> best_disparity(left_image.cols(), left_image.rows());
  ImageView<float> best_cost(left_image.cols(), left_image.rows());
  fill(best_cost, 1e9);

  const float tau1 = 0.3;
  const float tau2 = 0.6;
  const float alpha = 0.9;

  ImageView<float> idiff, gdiff_x, gdiff_y, cost;

  // Lets iterate over all the possible disparities and keep the best defined by the guided filter
  for (int j = search_region.min().y(); j < search_region.max().y(); j++ ) {
    std::cout << j << std::endl;
    for (int i = search_region.min().x(); i < search_region.max().x(); i++ ) {
      Vector2i disp(i, j);
      std::cout << disp << std::endl;

      std::ostringstream ostr;
      ostr << std::setw(4) << std::setfill('0') << i - search_region.min().x() << "_" << j - search_region.min().y();

      //ALT COST FUNCTION
      // Edge extension here is probably killing me
      idiff =
        abs(left_image - crop(right_image_exp, roi + disp - roi_expanded.min()));
      gdiff_x =
        abs(left_gradient_x - crop(right_gradient_x_exp, roi + disp - roi_expanded.min()));
      gdiff_y =
        abs(left_gradient_y - crop(right_gradient_y_exp, roi + disp - roi_expanded.min()));

      for ( int y = 0; y < left_image.rows(); y++ ) {
        for ( int x = 0; x < left_image.cols(); x++ ) {
          idiff(x,y) = std::min(idiff(x,y), tau1);
          gdiff_x(x,y) = std::min(gdiff_x(x,y) + gdiff_y(x,y),tau2);
        }
      }
      cost = (1 - alpha) * idiff + alpha * gdiff_x;

      // NCC cost function
      //cost = 1 - gaussian_filter(left_image * crop(right_image_exp, roi + disp - roi_expanded.min()), GAUSS_SIGMA) /
      //  sqrt(left_corr * crop(right_corr_exp, roi + disp - roi_expanded.min()));

      //cost = 1 - left_image * crop(right_image_exp, roi + disp - roi_expanded.min()) /
      //  sqrt(left_corr * crop(right_corr_exp, roi + disp - roi_expanded.min()));
      //cost = 1 - left_image * crop(right_image_exp, roi + disp - roi_expanded.min());

      // cost = 1 - left_mean * crop(right_mean_exp, roi + disp - roi_expanded.min()) /
      //   sqrt(left_corr * crop(right_corr_exp, roi + disp - roi_expanded.min()));
        //cost = gaussian_filter(cost, GAUSS_SIGMA);
      write_image("cost_" + ostr.str() + "_before.tif", cost * 4);

      // FILTER COST HERE
      ImageView<float> p_mean = gaussian_filter(cost, GAUSS_SIGMA);
      ImageView<float> ip_corr = gaussian_filter(left_image * cost, GAUSS_SIGMA);
      ImageView<float> var_i = left_corr - left_mean * left_mean;
      ImageView<float> cov_ip = ip_corr - left_mean * p_mean;
      ImageView<float> a = cov_ip / (var_i + 0.01 * 0.01);
      ImageView<float> b = p_mean - a * left_mean;
      cost = gaussian_filter(a, GAUSS_SIGMA) * left_image + gaussian_filter(b, GAUSS_SIGMA);
      write_image("cost_" + ostr.str() + "_after.tif", cost * 4);

      for (int y = 0; y < left_image.rows(); y++) {
        for (int x = 0; x < left_image.cols(); x++) {
          if (cost(x,y) < best_cost(x,y)) {
            best_cost(x,y) = cost(x,y);
            best_disparity(x,y) = disp;
          }
        }
      }
    }
  }

  write_image("volumefilter-D.tif", pixel_cast<PixelMask<Vector2i> >(best_disparity));

  return 0;
}
