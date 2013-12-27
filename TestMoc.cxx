#include <PatchMatch.h>

#include <vw/Math/Vector.h>
#include <vw/FileIO/DiskImageResourceGDAL.h>
#include <vw/FileIO/DiskImageView.h>
#include <vw/Stereo/DisparityMap.h>

#include <gtest/gtest.h>

using namespace vw;

namespace vw {
  template<> struct PixelFormatID<Vector2f> { static const PixelFormatEnum value = VW_PIXEL_GENERIC_2_CHANNEL; };
  template<> struct PixelFormatID<Vector4f> { static const PixelFormatEnum value = VW_PIXEL_GENERIC_4_CHANNEL; };
}

template <class ImageT>
void block_write_image( const std::string &filename,
                        vw::ImageViewBase<ImageT> const& image,
                        vw::ProgressCallback const& progress_callback = vw::ProgressCallback::dummy_instance() ) {
  boost::scoped_ptr<vw::DiskImageResourceGDAL> rsrc
    (new vw::DiskImageResourceGDAL(filename, image.impl().format(), Vector2i(256,256)));
  vw::block_write_image( *rsrc, image.impl(), progress_callback );
}

TEST( Moc, Crop ) {
  DiskImageView<float >
    left_image("../SemiGlobalMatching/data/moc/epi-L.crop.tif"),
    right_image("../SemiGlobalMatching/data/moc/epi-R.crop.tif");

  DiskImageView<PixelMask<Vector2i> >
    correct_disp("../SemiGlobalMatching/data/moc/crop11/crop11-D.tif");

  block_write_image( "moc_crop_pmview-D.tif",
                     stereo::patch_match( left_image, right_image,
                                          BBox2i(Vector2i(-20,-10),Vector2i(24,50)),
                                          Vector2i(11,11) ),
                     TerminalProgressCallback("test","PatchMatch:") );
}

TEST( Moc, All ) {
  DiskImageView<float >
    left_image("../SemiGlobalMatching/data/moc/epi-L.tif"),
    right_image("../SemiGlobalMatching/data/moc/epi-R.tif");

  block_write_image( "moc_full_pmview-D.tif",
                     stereo::patch_match( left_image, right_image,
                                          BBox2i(Vector2i(-40,-10),Vector2i(44,50)),
                                          Vector2i(11,11) ),
                     TerminalProgressCallback("test","PatchMatch:") );
}

int main( int argc, char **argv ) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
