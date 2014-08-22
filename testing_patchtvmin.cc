#include <vw/Core.h>
#include <vw/Image.h>
#include <vw/FileIO.h>
#include <vw/Stereo/PreFilter.h>
#include <vw/Stereo/CorrelationView.h>
#include <vw/Stereo/DisparityMap.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <PatchMatch2NCC.h>
#include <PatchMatch2Heise.h>
#include <SurfaceFitView.h>
#include <TVMin2.h>
#include <TVMin3.h>

using namespace vw;

int main(int argc, char **argv) {

  DiskImageView<float>
    left_disk_image("arctic/asp_al-L.crop.16.tif"),
    right_disk_image("arctic/asp_al-R.crop.16.tif");
  BBox2i search_region(Vector2i(-70,-25),
                       Vector2i(105,46));

  {
    vw::Timer corr_timer("Correlation time");
    DiskImageView<float>
      ltile("troublesome_tile/2048_3584_initial-L.tif"), rtile("troublesome_tile/2048_3584_initial-R.tif");

    ImageView<PixelMask<Vector2i> > solution
      = stereo::patch_match_ncc(ltile, rtile,
                                  BBox2i(0, 0, 91, 25),
                                  Vector2i(15, 15), -1, 6);
    write_image("pmheisetile-D.tif", solution);
  }
  exit(1);

  // Do only a single iteration so I can see something really noisy
  ImageView<PixelMask<Vector2i> > pm_disparity;
  {
    vw::Timer corr_timer("Correlation Time");

    pm_disparity =
      block_rasterize
      (stereo::patch_match_heise((left_disk_image),
                                 (right_disk_image),
                                 search_region/2,
                                 Vector2i(15, 15), 2 , 6),
       Vector2i(512, 512));
    write_image("patchmatch16-D.tif", pm_disparity);
    write_image("patchmatch16-L.tif", left_disk_image);
    write_image("patchmatch16-R.tif", right_disk_image);
    exit(1);
  }

  // Fit a surface
  ImageView<PixelMask<Vector2f> > sf_disparity = pm_disparity;
  if (1) {
    vw::Timer surface_timer("Surface Fitting time");

    sf_disparity = block_rasterize(stereo::surface_fit(pm_disparity),
                                   Vector2i(64, 64));
    // The sf_dispaity will be used to fill holes in combined. However
    // instead will reapply the patch match disparity and then blur it
    // in.
    //
    // We don't do this at the beginning because the original
    // disparity has a lot of noise.
    for (int j = 0; j < sf_disparity.rows(); j++ ) {
      for (int i = 0; i < sf_disparity.cols(); i++ ) {
        if (is_valid(pm_disparity(i,j))) {
          sf_disparity(i,j) = pm_disparity(i,j);
        }
      }
    }
    write_image("surface16-D.tif", sf_disparity);
  }

  // Apply imROF (original code)
  float lambda = .1;
  int iterations = 1000;
  ImageView<float> buffer0, buffer1;
  buffer0.set_size(sf_disparity.cols(), sf_disparity.rows());
  buffer1.set_size(sf_disparity.cols(), sf_disparity.rows());
  if (0) {
    vw::Timer timer("Original ROF");
    ImageView<PixelMask<Vector2f> > imROF_disparity(sf_disparity.cols(), sf_disparity.rows());
    fill(imROF_disparity, PixelMask<Vector2f>(Vector2f()));
    for (int i = 0; i < 2; i++) {
      buffer0 = select_channel(sf_disparity, i);
      stereo::imROF(buffer0, lambda, iterations, buffer1);
      select_channel(imROF_disparity, i) = buffer1;
    }
    write_image("imrof16-D.tif", pixel_cast<PixelMask<Vector2i> >(apply_mask(imROF_disparity)));
  }

  DiskImageView<float> lenna("lenna/image_noise.png");
  ImageView<float> lenna_norm = lenna / 255.0;

  // Apply my code
  if (0) {
    vw::Timer timer("ROF Primal Dual");

    float L2 = 8.0;
    float tau = 0.04;
    float sigma = 1.0 / (L2 * tau);
    ImageView<PixelMask<Vector2f> > imROF_disparity(sf_disparity.cols(), sf_disparity.rows());
    fill(imROF_disparity, PixelMask<Vector2f>(Vector2f()));
    for (int i = 0; i < 2; i++) {
      buffer0 = select_channel(sf_disparity, i);
      stereo::ROF(buffer0, lambda, iterations, sigma, tau, buffer1);
      select_channel(imROF_disparity, i) = buffer1;
    }

    write_image("rofpd16-D.tif", pixel_cast<PixelMask<Vector2i> >(apply_mask(imROF_disparity)));
  }

  {
    vw::Timer timer("Huber ROF Primal Dual");

    ImageView<float > output(lenna.cols(), lenna.rows());
    float L2 = 8.0;
    float tau = 0.04;
    float sigma = 1.0 / (L2 * tau);
    float alpha = .001;
    ImageView<PixelMask<Vector2f> > imROF_disparity(sf_disparity.cols(), sf_disparity.rows());
    fill(imROF_disparity, PixelMask<Vector2f>(Vector2f()));
    for (int i = 0; i < 2; i++) {
      buffer0 = select_channel(sf_disparity, i);
      stereo::HuberROF(buffer0, lambda, iterations, alpha, sigma, tau, buffer1);
      select_channel(imROF_disparity, i) = buffer1;
    }

    write_image("huberrof16-D.tif", pixel_cast<PixelMask<Vector2i> >(apply_mask(imROF_disparity)));
  }

  {
    vw::Timer timer("ROF TV L1 Primal Dual");

    ImageView<float > output(lenna.cols(), lenna.rows());
    float L2 = 8.0;
    float tau = 0.04;
    float sigma = 1.0 / (L2 * tau);
    ImageView<PixelMask<Vector2f> > imROF_disparity(sf_disparity.cols(), sf_disparity.rows());
    fill(imROF_disparity, PixelMask<Vector2f>(Vector2f()));
    for (int i = 0; i < 2; i++) {
      buffer0 = select_channel(sf_disparity, i);
      stereo::ROF_TVL1(buffer0, lambda, iterations, sigma, tau, buffer1);
      select_channel(imROF_disparity, i) = buffer1;
    }

    write_image("tvl116-D.tif", pixel_cast<PixelMask<Vector2i> >(apply_mask(imROF_disparity)));
  }

  {
    stereo::PMHeiseBase heise(bounding_box(left_disk_image),
                      Vector2i(15, 15), 2, 1);

    vw::Timer timer("PatchMatch internal weighted HuberROF");
    ImageView<float> lweight;
    heise.solve_gradient_weight(left_disk_image,
                                bounding_box(left_disk_image),
                                bounding_box(left_disk_image),
                                lweight);
    write_image("left16-W.tif", lweight);

    ImageView<float>
      p_x_dx(left_disk_image.cols(), left_disk_image.rows()),
      p_x_dy(left_disk_image.cols(), left_disk_image.rows()),
      p_y_dx(left_disk_image.cols(), left_disk_image.rows()),
      p_y_dy(left_disk_image.cols(), left_disk_image.rows());
    ImageView<Vector2i>
      smooth(left_disk_image.cols(), left_disk_image.rows());
    heise.solve_smooth
      (pixel_cast<Vector2i>(apply_mask(sf_disparity)),
       lweight, lambda * 1/3,
       p_x_dx, p_x_dy, p_y_dx, p_y_dy, smooth);
    write_image("heise16-D.tif", pixel_cast<PixelMask<Vector2i> >(smooth));
  }

  return 0;
}
