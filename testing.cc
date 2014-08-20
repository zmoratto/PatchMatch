#include <vw/Core.h>
#include <vw/Image.h>
#include <vw/FileIO.h>
#include <vw/Stereo/CorrelationView.h>
#include <vw/Stereo/DisparityMap.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <PatchMatch2NCC.h>
#include <SurfaceFitView.h>
#include <IterativeMappingStereo.h>

using namespace vw;

void blur_disparity(ImageView<PixelMask<Vector2f> >& sf_disparity,
                    BBox2i const& disparity_bounds) {
  select_channel(sf_disparity,0) =
    clamp(gaussian_filter(select_channel(sf_disparity,0),5),
          disparity_bounds.min()[0],
          disparity_bounds.max()[0]);
  select_channel(sf_disparity,1) =
    clamp(gaussian_filter(select_channel(sf_disparity,1),5),
          disparity_bounds.min()[1],
          disparity_bounds.max()[1]);
}

void copy_valid(ImageView<PixelMask<Vector2f> >& destination,
                ImageView<PixelMask<Vector2f> >& source) {
  for (int j = 0; j < destination.rows(); j++ ) {
    for (int i = 0; i < destination.cols(); i++ ) {
      if (is_valid(source(i,j))) {
        destination(i,j) = source(i,j);
      }
    }
  }
}

int main(int argc, char **argv) {

  DiskImageView<float>
    left_disk_image("arctic/asp_al-L.crop.32.tif"),
    right_disk_image("arctic/asp_al-R.crop.32.tif");

  // This is the search range for the final full scale image
  BBox2i search_region(Vector2i(-566, -224),
                       Vector2i(818, 110));

  ImageView<PixelMask<Vector2i> > pm_disparity;
  {
    vw::Timer corr_timer("Correlation Time");

    pm_disparity =
      block_rasterize
      (stereo::patch_match_ncc((left_disk_image),
                               (right_disk_image),
                               search_region/32,
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
    blur_disparity(sf_disparity, search_region/32);
    write_image("surface-blur32-D.tif", sf_disparity);
  }


  for (int i = 4; i > 0; i--) {
    std::cout << "Processing level: " << pow(2, i) << std::endl;
    std::ostringstream ltag;
    ltag << pow(2,i);

    DiskImageView<float> left_int("arctic/asp_al-L.crop."+ltag.str()+".tif"),
      right_int("arctic/asp_al-R.crop."+ltag.str()+".tif");
    ImageView<PixelMask<Vector2f> > sf_disparity_super =
      2*crop(resample(sf_disparity,2,2), bounding_box(left_int));
    sf_disparity = sf_disparity_super;
    ImageView<float> right_t;
    {
      vw::Timer timer("Transform Right");
      right_t =
        block_rasterize
        (transform(right_int,
                   stereo::DisparityTransform(sf_disparity)),
         Vector2i(256, 256));
      write_image("transformed"+ltag.str()+"-L.tif", left_int);
      write_image("transformed"+ltag.str()+"-R.tif", right_t);
    }
    ImageView<PixelMask<Vector2f> > combined;
    {
      vw::Timer timer("Correlation Time");
      pm_disparity =
        block_rasterize
        (stereo::correlate(left_int, right_t,
                           stereo::NullOperation(),
                           BBox2i(-8, -8, 16, 16),
                           Vector2i(13, 13), stereo::CROSS_CORRELATION, 2),
         Vector2i(256, 256));
      write_image("pmdelta"+ltag.str()+"-D.tif", pm_disparity);
      combined = sf_disparity_super + pm_disparity;
      write_image("patchmatch"+ltag.str()+"-D.tif", combined);
    }

    {
      vw::Timer surface_timer("Surface Fitting time");

      sf_disparity = block_rasterize(stereo::surface_fit(combined),
                                     Vector2i(64, 64));
      write_image("surface"+ltag.str()+"-D.tif", sf_disparity);
    }
    {
      vw::Timer timer("Bluring");
      copy_valid(sf_disparity, combined);
      blur_disparity(sf_disparity, search_region/pow(2,i));
      write_image("surface-blur"+ltag.str()+"-D.tif", sf_disparity);
    }
  }

  // Final level .. I got lazy

  DiskImageView<float> left1("arctic/asp_al-L.crop.tif"), right1("arctic/asp_al-R.crop.tif");
  ImageView<PixelMask<Vector2f> > sf_disparity_super =
    2 * crop(resample(sf_disparity, 2, 2), bounding_box(left1));
  sf_disparity = sf_disparity_super;
  ImageView<float> right1_t;
  ImageView<PixelMask<Vector2f> > combined;
  {
    vw::Timer timer("Transform Right");
    right1_t =
      block_rasterize
      (transform(right1,
                 stereo::DisparityTransform(sf_disparity)),
       Vector2i(256, 256));
    write_image("transformed1-L.tif", left1);
    write_image("transformed1-R.tif", right1_t);
  }
  {
    vw::Timer timer("Correlation Time");
    pm_disparity =
      block_rasterize
      (stereo::correlate(left1, right1_t,
                         stereo::NullOperation(),
                         BBox2i(-8, -8, 16, 16),
                         Vector2i(13, 13), stereo::CROSS_CORRELATION, 2),
       Vector2i(256, 256));

    write_image("pmdelta1-D.tif", pm_disparity);
    combined = sf_disparity_super + pm_disparity;
    write_image("patchmatch1-D.tif", combined);
  }
  {
    vw::Timer surface_timer("Surface Fitting time");

    sf_disparity = block_rasterize(stereo::surface_fit(combined),
                                   Vector2i(64, 64));
    write_image("surface1-D.tif", sf_disparity);
  }
  {
    vw::Timer timer("Bluring");
    copy_valid(sf_disparity, combined);
    blur_disparity(sf_disparity, search_region);
    write_image("surface-blur1-D.tif", sf_disparity);
  }

  {
    right1_t =
      block_rasterize
      (transform(right1,
                 stereo::DisparityTransform(sf_disparity)),
       Vector2i(256, 256));
    write_image("transformed1_2-L.tif", left1);
    write_image("transformed1_2-R.tif", right1_t);
    right1_t =
      block_rasterize
      (transform(right1,
                 stereo::DisparityTransform(combined)),
       Vector2i(256, 256));
    write_image("transformed1_2-TR.tif", right1_t);
  }

  return 0;
}
