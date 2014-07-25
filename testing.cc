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

void blur_disparity(ImageView<PixelMask<Vector2f> >& sf_disparity) {
  select_channel(sf_disparity,0) =
    gaussian_filter(select_channel(sf_disparity,0),5);
  select_channel(sf_disparity,1) =
    gaussian_filter(select_channel(sf_disparity,1),5);
}

int main(int argc, char **argv) {

  DiskImageView<float>
    left_disk_image("arctic/asp_al-L.crop.32.tif"),
    right_disk_image("arctic/asp_al-R.crop.32.tif");
  std::string match_filename("arctic/asp_al-L.crop.8__asp_al-R.crop.8.match");
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
                           search_region/4,
                           Vector2i(15, 15), 2 , 16),
       Vector2i(256, 256));
    write_image("patchmatch32-D.tif", pm_disparity);
  }
  ImageView<PixelMask<Vector2f> > sf_disparity;
  {
    vw::Timer surface_timer("Surface Fitting time");

    sf_disparity = block_rasterize(stereo::surface_fit(pm_disparity),
                                   Vector2i(64, 64));
    write_image("surface32-D.tif", sf_disparity);
  }
  {
    vw::Timer timer("Bluring");
    blur_disparity(sf_disparity);
    write_image("surface-blur32-D.tif", sf_disparity);
  }


  DiskImageView<float> left16("arctic/asp_al-L.crop.16.tif"), right16("arctic/asp_al-R.crop.16.tif");
  ImageView<PixelMask<Vector2f> > sf_disparity_super =
    2*crop(resample(sf_disparity,2,2), bounding_box(left16));
  sf_disparity = sf_disparity_super;
  ImageView<float> right16_t;
  {
    vw::Timer timer("Transform Right");
    right16_t =
      block_rasterize
      (transform(right16,
                 stereo::DisparityTransform(sf_disparity)),
       Vector2i(256, 256));
    write_image("transformed16-L.tif", left16);
    write_image("transformed16-R.tif", right16_t);
  }
  ImageView<PixelMask<Vector2f> > combined;
  {
    vw::Timer timer("Correlation Time");
    pm_disparity =
      block_rasterize
      (stereo::patch_match(filter.filter(left16), filter.filter(right16_t),
                           BBox2i(-4, -4, 8, 8),
                           Vector2i(15, 15), 2, 16),
       Vector2i(256, 256));
    write_image("pmdelta16-D.tif", pm_disparity);
    combined = sf_disparity_super + pm_disparity;
    write_image("patchmatch16-D.tif", combined);
  }

  {
    vw::Timer surface_timer("Surface Fitting time");

    sf_disparity = block_rasterize(stereo::surface_fit(combined),
                                   Vector2i(64, 64));
    write_image("surface16-D.tif", sf_disparity);
  }
  {
    // The sf_dispaity will be used to fill holes in combined. However
    // instead will reapply the patch match disparity and then blur it
    // in.
    //
    // We don't do this at the beginning because the original
    // disparity has a lot of noise.
    for (int j = 0; j < sf_disparity.rows(); j++ ) {
      for (int i = 0; i < sf_disparity.cols(); i++ ) {
        if (is_valid(combined(i,j))) {
          sf_disparity(i,j) = combined(i,j);
        }
      }
    }

    vw::Timer timer("Bluring");
    blur_disparity(sf_disparity);
    write_image("surface-blur16-D.tif", sf_disparity);
  }

  {
    right16_t =
      block_rasterize
      (transform(right16,
                 stereo::DisparityTransform(sf_disparity)),
       Vector2i(256, 256));
    write_image("transformed16_2-L.tif", left16);
    write_image("transformed16_2-R.tif", right16_t);
    right16_t =
      block_rasterize
      (transform(right16,
                 stereo::DisparityTransform(combined)),
       Vector2i(256, 256));
    write_image("transformed16_2-TR.tif", right16_t);
  }

  DiskImageView<float> left8("arctic/asp_al-L.crop.8.tif"), right8("arctic/asp_al-R.crop.8.tif");
  sf_disparity_super =
    2 * crop(resample(sf_disparity, 2, 2), bounding_box(left8));
  sf_disparity = sf_disparity_super;
  ImageView<float> right8_t;
  {
    vw::Timer timer("Transform Right");
    right8_t =
      block_rasterize
      (transform(right8,
                 stereo::DisparityTransform(sf_disparity)),
       Vector2i(256, 256));
    write_image("transformed8-L.tif", left8);
    write_image("transformed8-R.tif", right8_t);
  }
  {
    vw::Timer timer("Correlation Time");
    pm_disparity =
      block_rasterize
      (stereo::patch_match(filter.filter(left8), filter.filter(right8_t),
                           BBox2i(-4, -4, 8, 8),
                           Vector2i(13, 13), 2, 16),
       Vector2i(256, 256));
    write_image("pmdelta8-D.tif", pm_disparity);
    combined = sf_disparity_super + pm_disparity;
    write_image("patchmatch8-D.tif", combined);
  }
  {
    vw::Timer surface_timer("Surface Fitting time");

    sf_disparity = block_rasterize(stereo::surface_fit(combined),
                                   Vector2i(64, 64));
    write_image("surface8-D.tif", sf_disparity);
  }
  {
    // The sf_dispaity will be used to fill holes in combined. However
    // instead will reapply the patch match disparity and then blur it
    // in.
    //
    // We don't do this at the beginning because the original
    // disparity has a lot of noise.
    for (int j = 0; j < sf_disparity.rows(); j++ ) {
      for (int i = 0; i < sf_disparity.cols(); i++ ) {
        if (is_valid(combined(i,j))) {
          sf_disparity(i,j) = combined(i,j);
        }
      }
    }

    vw::Timer timer("Bluring");
    blur_disparity(sf_disparity);
    write_image("surface-blur8-D.tif", sf_disparity);
  }

  {
    right8_t =
      block_rasterize
      (transform(right8,
                 stereo::DisparityTransform(sf_disparity)),
       Vector2i(256, 256));
    write_image("transformed8_2-L.tif", left8);
    write_image("transformed8_2-R.tif", right8_t);
    right8_t =
      block_rasterize
      (transform(right8,
                 stereo::DisparityTransform(combined)),
       Vector2i(256, 256));
    write_image("transformed8_2-TR.tif", right8_t);
  }

  DiskImageView<float> left4("arctic/asp_al-L.crop.4.tif"), right4("arctic/asp_al-R.crop.4.tif");
  sf_disparity_super =
    2 * crop(resample(sf_disparity, 2, 2), bounding_box(left4));
  sf_disparity = sf_disparity_super;
  ImageView<float> right4_t;
  {
    vw::Timer timer("Transform Right");
    right4_t =
      block_rasterize
      (transform(right4,
                 stereo::DisparityTransform(sf_disparity)),
       Vector2i(256, 256));
    write_image("transformed4-L.tif", left4);
    write_image("transformed4-R.tif", right4_t);
  }
  {
    vw::Timer timer("Correlation Time");
    pm_disparity =
      block_rasterize
      (stereo::patch_match(filter.filter(left4), filter.filter(right4_t),
                           BBox2i(-4, -4, 8, 8),
                           Vector2i(13, 13), 2, 16),
       Vector2i(256, 256));
    write_image("pmdelta4-D.tif", pm_disparity);
    combined = sf_disparity_super + pm_disparity;
    write_image("patchmatch4-D.tif", combined);
  }
  {
    vw::Timer surface_timer("Surface Fitting time");

    sf_disparity = block_rasterize(stereo::surface_fit(combined),
                                   Vector2i(64, 64));
    write_image("surface4-D.tif", sf_disparity);
  }
  {
    // The sf_dispaity will be used to fill holes in combined. However
    // instead will reapply the patch match disparity and then blur it
    // in.
    //
    // We don't do this at the beginning because the original
    // disparity has a lot of noise.
    for (int j = 0; j < sf_disparity.rows(); j++ ) {
      for (int i = 0; i < sf_disparity.cols(); i++ ) {
        if (is_valid(combined(i,j))) {
          sf_disparity(i,j) = combined(i,j);
        }
      }
    }

    vw::Timer timer("Bluring");
    blur_disparity(sf_disparity);
    write_image("surface-blur4-D.tif", sf_disparity);
  }

  {
    right4_t =
      block_rasterize
      (transform(right4,
                 stereo::DisparityTransform(sf_disparity)),
       Vector2i(256, 256));
    write_image("transformed4_2-L.tif", left4);
    write_image("transformed4_2-R.tif", right4_t);
    right4_t =
      block_rasterize
      (transform(right4,
                 stereo::DisparityTransform(combined)),
       Vector2i(256, 256));
    write_image("transformed4_2-TR.tif", right4_t);
  }

  // RESTART

  DiskImageView<float> left2("arctic/asp_al-L.crop.2.tif"), right2("arctic/asp_al-R.crop.2.tif");
  sf_disparity_super =
    2 * crop(resample(sf_disparity, 2, 2), bounding_box(left2));
  sf_disparity = sf_disparity_super;
  ImageView<float> right2_t;
  {
    vw::Timer timer("Transform Right");
    right2_t =
      block_rasterize
      (transform(right2,
                 stereo::DisparityTransform(sf_disparity)),
       Vector2i(256, 256));
    write_image("transformed2-L.tif", left2);
    write_image("transformed2-R.tif", right2_t);
  }
  {
    vw::Timer timer("Correlation Time");
    pm_disparity =
      block_rasterize
      (stereo::patch_match(filter.filter(left2), filter.filter(right2_t),
                           BBox2i(-4, -4, 8, 8),
                           Vector2i(13, 13), 2, 16),
       Vector2i(256, 256));
    write_image("pmdelta2-D.tif", pm_disparity);
    combined = sf_disparity_super + pm_disparity;
    write_image("patchmatch2-D.tif", combined);
  }
  {
    vw::Timer surface_timer("Surface Fitting time");

    sf_disparity = block_rasterize(stereo::surface_fit(combined),
                                   Vector2i(64, 64));
    write_image("surface2-D.tif", sf_disparity);
  }
  {
    // The sf_dispaity will be used to fill holes in combined. However
    // instead will reapply the patch match disparity and then blur it
    // in.
    //
    // We don't do this at the beginning because the original
    // disparity has a lot of noise.
    for (int j = 0; j < sf_disparity.rows(); j++ ) {
      for (int i = 0; i < sf_disparity.cols(); i++ ) {
        if (is_valid(combined(i,j))) {
          sf_disparity(i,j) = combined(i,j);
        }
      }
    }

    vw::Timer timer("Bluring");
    blur_disparity(sf_disparity);
    write_image("surface-blur2-D.tif", sf_disparity);
  }

  {
    right2_t =
      block_rasterize
      (transform(right2,
                 stereo::DisparityTransform(sf_disparity)),
       Vector2i(256, 256));
    write_image("transformed2_2-L.tif", left2);
    write_image("transformed2_2-R.tif", right2_t);
    right2_t =
      block_rasterize
      (transform(right2,
                 stereo::DisparityTransform(combined)),
       Vector2i(256, 256));
    write_image("transformed2_2-TR.tif", right2_t);
  }


  return 0;
}
