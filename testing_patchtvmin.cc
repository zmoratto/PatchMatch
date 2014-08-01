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
    left_disk_image("arctic/asp_al-L.crop.32.tif"),
    right_disk_image("arctic/asp_al-R.crop.32.tif");
  BBox2i search_region(Vector2i(-70,-25),
                       Vector2i(105,46));

  // Do only a single iteration so I can see something really noisy
  ImageView<PixelMask<Vector2i> > pm_disparity;
  {
    vw::Timer corr_timer("Correlation Time");

    pm_disparity =
      block_rasterize
      (stereo::patch_match_heise((left_disk_image),
                                 (right_disk_image),
                                 search_region/4,
                                 Vector2i(15, 15), 2 , 10),
       Vector2i(256, 256));
    write_image("patchmatch32-D.tif", pm_disparity);
    exit(1);
  }

  // Fit a surface
  ImageView<PixelMask<Vector2f> > sf_disparity = pm_disparity;
  if (0) {
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
    write_image("surface32-D.tif", sf_disparity);
  }

  // Apply imROF (original code)
  float lambda = 1;
  int iterations = 1000;
  ImageView<float> buffer0, buffer1;
  buffer0.set_size(sf_disparity.cols(), sf_disparity.rows());
  buffer1.set_size(sf_disparity.cols(), sf_disparity.rows());
  {
    vw::Timer timer("Original ROF");
    ImageView<PixelMask<Vector2f> > imROF_disparity(sf_disparity.cols(), sf_disparity.rows());
    fill(imROF_disparity, PixelMask<Vector2f>(Vector2f()));
    for (int i = 0; i < 2; i++) {
      buffer0 = select_channel(sf_disparity, i);
      stereo::imROF(buffer0, lambda, iterations, buffer1);
      select_channel(imROF_disparity, i) = buffer1;
    }
    write_image("imrof32-D.tif", imROF_disparity);
  }

  DiskImageView<float> lenna("lenna/image_noise.png");
  ImageView<float> lenna_norm = lenna / 255.0;

  // Apply my code
  {
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

    write_image("rofpd32-D.tif", imROF_disparity);
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

    write_image("huberrof32-D.tif", imROF_disparity);
  }

  return 0;
}
