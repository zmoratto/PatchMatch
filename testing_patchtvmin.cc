#include <vw/Core.h>
#include <vw/Image.h>
#include <vw/FileIO.h>
#include <vw/Stereo/PreFilter.h>
#include <vw/Stereo/CorrelationView.h>
#include <vw/Stereo/DisparityMap.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <PatchMatch2NCC.h>
#include <SurfaceFitView.h>
#include <TVMin2.h>

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
      (stereo::patch_match_ncc((left_disk_image),
                               (right_disk_image),
                               search_region/4,
                               Vector2i(15, 15), 2 , 1),
       Vector2i(256, 256));
    write_image("patchmatch32-D.tif", pm_disparity);
  }

  // Fit a surface
  ImageView<PixelMask<Vector2f> > sf_disparity;
  {
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

  // Apply TV Minimization
  ImageView<PixelMask<Vector2f> > tv_disparity;
  {
    vw::Timer timer("TV Minimization");
    tv_disparity = block_rasterize(stereo::tvmin_fit(sf_disparity),
                                   Vector2i(64, 64));
    write_image("tvmin32-D.tif", tv_disparity);
  }

  return 0;
}
