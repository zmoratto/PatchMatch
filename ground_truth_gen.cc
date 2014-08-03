#include <vw/Core.h>
#include <vw/Image.h>
#include <vw/FileIO.h>
#include <vw/Stereo/DisparityMap.h>

using namespace vw;

template <class ImageT>
void block_write_image( const std::string &filename,
                        vw::ImageViewBase<ImageT> const& image,
                        vw::ProgressCallback const& progress_callback = vw::ProgressCallback::dummy_instance() ) {
  boost::scoped_ptr<vw::DiskImageResourceGDAL> rsrc
    (new vw::DiskImageResourceGDAL(filename, image.impl().format(), Vector2i(256,256)));
  vw::block_write_image( *rsrc, image.impl(), progress_callback );
}

int main(int argc, char **argv) {
  // This is expected to run after the testing executable.

  std::string input_filename("patchmatch1-D.tif");

  std::string output_filename[] =
    { "ground_truth-2.tif",
      "ground_truth-4.tif",
      "ground_truth-8.tif",
      "ground_truth-16.tif",
      "ground_truth-32.tif"
    };

  for (int i = 0; i < 5; i++ ) {
    DiskImageView<PixelMask<Vector2f> > input(input_filename);
    block_write_image(output_filename[i],
                      stereo::disparity_subsample(input),
                      TerminalProgressCallback("", output_filename[i]));

    input_filename = output_filename[i];
  }  return 0;
}
