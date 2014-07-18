#include <PatchMatch.h>

#include <vw/Core/Debugging.h>
#include <vw/FileIO/DiskImageView.h>
#include <vw/FileIO/DiskImageResourceGDAL.h>
#include <vw/Stereo/DisparityMap.h>

#include <boost/program_options.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
namespace po = boost::program_options;
namespace fs = boost::filesystem;

using namespace vw;

template <class ImageT>
void block_write_image( const std::string &filename,
                        vw::ImageViewBase<ImageT> const& image,
                        vw::ProgressCallback const& progress_callback = vw::ProgressCallback::dummy_instance() ) {
  boost::scoped_ptr<vw::DiskImageResourceGDAL> rsrc
    (new vw::DiskImageResourceGDAL(filename, image.impl().format(), Vector2i(256,256)));
  vw::block_write_image( *rsrc, image.impl(), progress_callback );
}

int main( int argc, char *argv[] ) {
  try {
    std::string left_file_name, right_file_name, tag;
    int32 h_corr_min, h_corr_max;
    int32 v_corr_min, v_corr_max;
    int32 xkernel, ykernel;
    float cross_const;

    po::options_description desc("Options");
    desc.add_options()
      ("help,h", "Display this help message")
      ("left", po::value(&left_file_name), "Explicitly specify the \"left\" input file")
      ("right", po::value(&right_file_name), "Explicitly specify the \"right\" input file")
      ("tag", po::value(&tag)->default_value("patchmatch"), "Output prefix")
      ("h-corr-min", po::value(&h_corr_min)->default_value(-100), "Minimum horizontal disparity")
      ("h-corr-max", po::value(&h_corr_max)->default_value(100), "Maximum horizontal disparity")
      ("v-corr-min", po::value(&v_corr_min)->default_value(-20), "Minimum vertical disparity")
      ("v-corr-max", po::value(&v_corr_max)->default_value(20), "Maximum vertical disparity")
      ("xkernel", po::value(&xkernel)->default_value(15), "Horizontal correlation kernel size")
      ("ykernel", po::value(&ykernel)->default_value(15), "Vertical correlation kernel size")
      ("cross-const", po::value(&cross_const)->default_value(1.0), "Cross consistency check, less than 0 is off")
      ;
    po::positional_options_description p;
    p.add("left", 1);
    p.add("right", 1);
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

    if( vm.count("left") != 1 || vm.count("right") != 1 ) {
      vw_out() << "Error: Must specify one (and only one) left and right input file!" << std::endl;
      vw_out() << desc << std::endl;
      return 1;
    }

    DiskImageView<PixelGray<float> > left_disk_image(left_file_name );
    DiskImageView<PixelGray<float> > right_disk_image(right_file_name );

    // Actually invoke the rasterazation
    {
      vw::Timer corr_timer("Correlation Time");
      block_write_image( tag + "-D.tif",
                         stereo::patch_match( left_disk_image,
                                              right_disk_image,
                                              BBox2i(Vector2i(h_corr_min, v_corr_min),
                                                     Vector2i(h_corr_max, v_corr_max)),
                                              Vector2i(xkernel, ykernel),
                                              cross_const ),
                         TerminalProgressCallback( "", "Rendering: ") );
    }
  } catch (const vw::Exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
