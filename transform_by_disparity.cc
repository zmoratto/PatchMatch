#include <vw/Core.h>
#include <vw/Image.h>
#include <vw/FileIO.h>
#include <vw/Stereo/PreFilter.h>
#include <vw/Stereo/CorrelationView.h>
#include <vw/Stereo/DisparityMap.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

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
  try {
    std::string left_file_name, right_file_name, disparity_file_name, tag;
 
    po::options_description desc("Options");
    desc.add_options()
      ("help,h", "Display this help message")
      ("left", po::value(&left_file_name), "Explicitly specify the \"left\" input file")
      ("right", po::value(&right_file_name), "Explicitly specify the \"right\" input file")
      ("disparity", po::value(&disparity_file_name), "Explicitly specify the disparity input file")
      ("tag", po::value(&tag)->default_value("transformed"), "Output prefix");
    po::positional_options_description p;
    p.add("left", 1);
    p.add("right", 1);
    p.add("disparity", 1);
    p.add("tag", 1);

    po::variables_map vm;
    try {
      po::store( po::command_line_parser( argc, argv ).options(desc).positional(p).run(), vm );
      po::notify( vm );
    } catch (const po::error& e) {
      std::cout << "An error occured while parsing command line arguments.\n";
      std::cout << "\t" << e.what() << "\n\n";
      std::cout << desc << std::endl;
      return 1;
    }

    if( vm.count("help") ) {
      vw_out() << desc << std::endl;
      return 1;
    }

    if ( vm.count("left") != 1 || vm.count("right") != 1 || vm.count("disparity") != 1) {
      vw_out() << "Error: Must specify one (and only one) left, right, and disparity input file!" << std::endl;
      vw_out() << desc << std::endl;
      return 1;
    }

    DiskImageView<float> left_disk_image(left_file_name );
    DiskImageView<float> right_disk_image(right_file_name );
    DiskImageView<PixelMask<Vector2f> > disparity_disk_image(disparity_file_name);
    block_write_image(tag + "-tfm.tif",
                      crop(transform(right_disk_image,
                                          stereo::DisparityTransform
                                          (disparity_disk_image)),
                                BBox2i(0, 0, left_disk_image.cols(),
                                       left_disk_image.rows())),
                      TerminalProgressCallback("","Rendering:"));

    block_write_image(tag + "-tfm-diff.tif",
                      abs(left_disk_image -
                           crop(transform(right_disk_image,
                                          stereo::DisparityTransform
                                          (disparity_disk_image)),
                                BBox2i(0, 0, left_disk_image.cols(),
                                       left_disk_image.rows()))),
                      TerminalProgressCallback("","Rendering:"));

  } catch (const vw::Exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}

