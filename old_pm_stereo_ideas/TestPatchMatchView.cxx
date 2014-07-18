#include <vw/Core.h>
#include <vw/Image.h>
#include <vw/FileIO.h>
#include <vw/Math/Vector.h>

#include <PatchMatch.h>

#include <gtest/gtest.h>

using namespace vw;

template <class ImageT>
void block_write_image( const std::string &filename,
                        vw::ImageViewBase<ImageT> const& image,
                        vw::ProgressCallback const& progress_callback = vw::ProgressCallback::dummy_instance() ) {
  boost::scoped_ptr<vw::DiskImageResourceGDAL> rsrc
    (new vw::DiskImageResourceGDAL(filename, image.impl().format(), Vector2i(256,256)));
  vw::block_write_image( *rsrc, image.impl(), progress_callback );
}

TEST( PatchMatchView, PatchMatchView ) {
  DiskImageView<PixelGray<uint8> >
    left_image("../SemiGlobalMatching/data/cones/im2.png"),
    right_image("../SemiGlobalMatching/data/cones/im6.png");

  block_write_image( "final_disparity_pmview-D.tif",
                     stereo::patch_match( left_image, right_image,
                                          BBox2i(Vector2i(-64,-1),Vector2i(0,1)),
                                          Vector2i(15,15) ),
                     TerminalProgressCallback("test","PatchMatch:") );
}

int main( int argc, char **argv ) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
