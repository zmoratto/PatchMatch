#include <vw/Image/ImageView.h>
#include <vw/Image/MaskViews.h>
#include <vw/Math/BBox.h>
#include <vw/FileIO.h>
#include <vw/Stereo/DisparityMap.h>

#include <ARAPDataTerm.h>
#include <ARAPSmoothTerm.h>
#include <PatchMatch2NCC.h>
#include <NelderMead.h>

#include <Eigen/Sparse>

using namespace vw;

int main(int argc, char **argv) {

  DiskImageView<float>
    left_disk_image("arctic/asp_al-L.crop.16.tif"),
    right_disk_image("arctic/asp_al-R.crop.16.tif");
  ImageView<float> left_image = left_disk_image, right_image = right_disk_image;
  BBox2i search_region(Vector2i(-70,-25),
                       Vector2i(105,46));

  ImageView<PixelMask<Vector2i> > pm_disparity =
    stereo::patch_match_ncc(left_disk_image,
                            right_disk_image,
                            search_region/2,
                            Vector2i(15, 15), 2, 3);
  write_image("patchmatch16-D.tif", pm_disparity);

  // Defining the superpixels
  std::vector<BBox2i> box_vec =
    image_blocks(pm_disparity, 32, 32);
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

  // Define the surfaces
  std::vector<Vector<double, 10> > superpixel_surfaces;
  stereo::define_superpixels(pm_disparity,
                             superpixels,
                             superpixel_surfaces);

  // Track superpixel 100
  std::cout << "Superpixel 100:\n"
            << superpixels[100].first << "\n"
            << superpixels[100].second << "\n"
            << superpixel_surfaces[100] << std::endl;

  // Render an image of what the surfaces represent
  ImageView<PixelMask<Vector2f> > quad_disparity(pm_disparity.cols(), pm_disparity.rows());
  stereo::render_disparity_image(superpixels,
                                 superpixel_surfaces,
                                 quad_disparity);
  write_image("initial_quad16-D.tif", quad_disparity);
  ImageView<Vector2f> u = apply_mask(quad_disparity);

  // Build the symetric laplacian sum
  Eigen::SparseMatrix<float> laplacian_sqr_sum(left_image.cols() * left_image.rows(),
                                                left_image.cols() * left_image.rows());
  {
    ImageView<float> weight1(left_image.cols(), left_image.rows());
    stereo::generate_weight1(left_image, 0.1, weight1);
    write_image("weight1.tif", weight1);
    Eigen::SparseMatrix<float> l(left_image.cols() * left_image.rows(),
                                  left_image.cols() * left_image.rows());
    stereo::generate_laplacian1(weight1, l);
    laplacian_sqr_sum += l.transpose() * l;
  }
  {
    ImageView<float> weight2(left_image.cols(), left_image.rows());
    stereo::generate_weight2(left_image, 0.1, weight2);
    write_image("weight2.tif", weight2);
    Eigen::SparseMatrix<float> l(left_image.cols() * left_image.rows(),
                                  left_image.cols() * left_image.rows());
    stereo::generate_laplacian2(weight2, l);
    laplacian_sqr_sum += l.transpose() * l;
  }
  {
    ImageView<float> weight3(left_image.cols(), left_image.rows());
    stereo::generate_weight3(left_image, 0.1, weight3);
    write_image("weight3.tif", weight3);
    Eigen::SparseMatrix<float> l(left_image.cols() * left_image.rows(),
                                  left_image.cols() * left_image.rows());
    stereo::generate_laplacian3(weight3, l);
    laplacian_sqr_sum += l.transpose() * l;
  }
  {
    ImageView<float> weight4(left_image.cols(), left_image.rows());
    stereo::generate_weight4(left_image, 0.1, weight4);
    write_image("weight4.tif", weight4);
    Eigen::SparseMatrix<float> l(left_image.cols() * left_image.rows(),
                                  left_image.cols() * left_image.rows());
    stereo::generate_laplacian4(weight4, l);
    laplacian_sqr_sum += l.transpose() * l;
  }

  std::cout << "Non zero coeffients in LTL: " << laplacian_sqr_sum.nonZeros() << std::endl;

  // Perform the actual iteration
  stereo::IndiceFinder indexer(pm_disparity.cols()/32, superpixels.size());
  double theta = 0;
  for (int i = 0; i < 1; i++ ) {
    // Perform a simplex algorithm to solve for a better fitting surface
    for (size_t s = 0; s < superpixels.size(); s++ ) {
      Vector<double, 10> seeds[11];
      seeds[0] = superpixel_surfaces[s];
      seeds[1] = superpixel_surfaces[indexer.tl(s)];
      seeds[2] = superpixel_surfaces[indexer.t(s)];
      seeds[3] = superpixel_surfaces[indexer.tr(s)];
      seeds[4] = superpixel_surfaces[indexer.l(s)];
      seeds[5] = superpixel_surfaces[indexer.r(s)];
      seeds[6] = superpixel_surfaces[indexer.bl(s)];
      seeds[7] = superpixel_surfaces[indexer.b(s)];
      seeds[8] = superpixel_surfaces[indexer.br(s)];
      seeds[9] = superpixel_surfaces[std::min(indexer.num_indices_, int(s)+2)];
      seeds[10] = superpixel_surfaces[std::max(0, int(s)-2)];

      /*
      std::cout << "Starting: " << superpixel_surfaces[s] << std::endl;
      std::cout << "  cost: "
                << stereo::evaluate_superpixel(left_image,
                                               right_image,
                                               superpixels[s].first,
                                               superpixels[s].second,
                                               superpixel_surfaces[s]) << std::endl;
      */
      stereo::Amoeba<10> amoeba(1e-2);
      stereo::NMFunctor functor(left_image, right_image,
                                u, superpixels[s], theta);
      superpixel_surfaces[s] =
        amoeba.minimize(seeds, functor);
      /*
      std::cout << "Finish: " << superpixel_surfaces[s] << std::endl;
      std::cout << "  cost: "
                << stereo::evaluate_superpixel(left_image,
                                               right_image,
                                               superpixels[s].first,
                                               superpixels[s].second,
                                               superpixel_surfaces[s]) << std::endl;
      */
    }
    stereo::render_disparity_image(superpixels,
                                   superpixel_surfaces,
                                   quad_disparity);
    write_image("iteration0_quad16-D.tif", quad_disparity);

    // Perform a second order fitting for the entire image.
    double tau = .2;
    ImageView<float> v_x = select_channel(quad_disparity,0),
      v_y = select_channel(quad_disparity,1);
    Eigen::Map<Eigen::VectorXf> v_x_map(v_x.data(), v_x.cols() * v_x.rows());
    Eigen::Map<Eigen::VectorXf> v_y_map(v_y.data(), v_y.cols() * v_y.rows());
    ImageView<float> u_x = select_channel(u, 0), u_y = select_channel(u, 1);
    Eigen::Map<Eigen::VectorXf> u_x_map(u_x.data(), u_x.cols() * u_x.rows());
    Eigen::Map<Eigen::VectorXf> u_y_map(u_y.data(), u_y.cols() * u_y.rows());
    for (int j = 0; j < 10; j++ ) {
      Eigen::VectorXf es(left_image.cols() * left_image.rows());
      u_x_map -= tau * (2 * laplacian_sqr_sum * u_x_map + 2 * theta * (u_x_map - v_x_map));
      u_y_map -= tau * (2 * laplacian_sqr_sum * u_y_map + 2 * theta * (u_y_map - v_y_map));
    }
    select_channel(u, 0) = u_x;
    select_channel(u, 1) = u_y;
    write_image("iteration0_smooth16-D.tif", pixel_cast<PixelMask<Vector2f> >(u));

    // Increase theta
    theta += .1;
  }

  return 0;
}
