#include <vw/Core.h>
#include <vw/Image.h>
#include <vw/FileIO.h>
#include <vw/Stereo/PreFilter.h>
#include <vw/Stereo/CorrelationView.h>
#include <vw/Stereo/DisparityMap.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <PatchMatch2.h>
#include <SurfaceFitView.h>
#include <IterativeMappingStereo.h>

using namespace vw;

int main(int argc, char **argv) {

  DiskImageView<float>
    left_disk_image("arctic/asp_al-L.crop.8.tif"),
    right_disk_image("arctic/asp_al-R.crop.8.tif");
  BBox2i search_region(Vector2i(-70,-25),
                       Vector2i(105,46));

  stereo::SubtractedMean filter(15.0);


  ImageView<PixelMask<Vector2i> > pm_disparity;
  {
    vw::Timer corr_timer("Correlation Time");

    pm_disparity =
      block_rasterize
      (stereo::patch_match(filter.filter(left_disk_image),
                           filter.filter(right_disk_image),
                           search_region,
                           Vector2i(15, 15), 2 /*cross*/, 5 /*iterations*/),
       Vector2i(256, 256));
    write_image("patchmatch-D.tif", pm_disparity);
  }

  ImageView<PixelMask<Vector2f> > sf_disparity;
  {
    vw::Timer surface_timer("Surface Fitting time");

    sf_disparity = block_rasterize(stereo::surface_fit(pm_disparity),
                                   Vector2i(128, 128));
    write_image("surface-D.tif", sf_disparity);
  }

  {
    select_channel(sf_disparity,0) =
      gaussian_filter(select_channel(sf_disparity,0),5);
    select_channel(sf_disparity,1) =
      gaussian_filter(select_channel(sf_disparity,1),5);
    write_image("surface-blur-D.tif", sf_disparity);
  }

  ImageView<float> right_transformed;
  {
    vw::Timer timer("Transform Right");
    right_transformed =
      block_rasterize
      (transform(right_disk_image,
                 stereo::DisparityTransform(sf_disparity)),
       Vector2i(256, 256));
    write_image("transformed-R.tif", right_transformed);
  }

  /*
  for (int i = 12; i > 0; i-- ) {
    std::ostringstream ostr;
    ostr << i;

    {
      vw::Timer timer("Iteration");
      pm_disparity =
        block_rasterize(stereo::patch_match(filter.filter(left_disk_image),
                                            filter.filter(right_transformed),
                                            BBox2i(Vector2i(-5, -5) * i,
                                                   Vector2i(5, 5) * i),
                                            Vector2i(15, 15), 2, 3),
                        Vector2i(256, 256));
      write_image("delta-"+ostr.str()+"-pm-D.tif", pm_disparity);
    }
    sf_disparity = sf_disparity + pm_disparity;
    for ( int j = 0; j < sf_disparity.rows(); j++ ) {
      for (int i = 0; i < sf_disparity.cols(); i++ ) {
        validate(sf_disparity(i,j));
      }
    }
    write_image("iteration-"+ostr.str()+"-pm-D.tif", sf_disparity);
    select_channel(sf_disparity,0) =
      gaussian_filter(select_channel(sf_disparity,0),5);
    select_channel(sf_disparity,1) =
      gaussian_filter(select_channel(sf_disparity,1),5);
    write_image("iteration-blur-"+ostr.str()+"-pm-D.tif", sf_disparity);

    {
      vw::Timer timer("Transform Right");
      right_transformed =
        block_rasterize
        (transform(right_disk_image,
                   stereo::DisparityTransform(sf_disparity)),
         Vector2i(256, 256));
      write_image("transformed-"+ostr.str()+"-R.tif", right_transformed);
    }
  }
  */

  return 0;
}
