#include <vw/Core.h>
#include <vw/Image.h>
#include <vw/FileIO.h>
#include <vw/Stereo/PreFilter.h>
#include <vw/Stereo/CorrelationView.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <PatchMatch2.h>

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
    std::string left_file_name, right_file_name, tag;
    int32 h_corr_min, h_corr_max;
    int32 v_corr_min, v_corr_max;
    int32 iterations;

    po::options_description desc("Options");
    desc.add_options()
      ("help,h", "Display this help message")
      ("left", po::value(&left_file_name), "Explicitly specify the \"left\" input file")
      ("right", po::value(&right_file_name), "Explicitly specify the \"right\" input file")
      ("tag", po::value(&tag)->default_value("patchmatch"), "Output prefix")
      ("iteration", po::value(&iterations)->default_value(1), "Number of patch match iterations")
      ("h-corr-min", po::value(&h_corr_min)->default_value(-70), "Minimum horizontal disparity")
      ("h-corr-max", po::value(&h_corr_max)->default_value(105), "Maximum horizontal disparity")
      ("v-corr-min", po::value(&v_corr_min)->default_value(-25), "Minimum vertical disparity")
      ("v-corr-max", po::value(&v_corr_max)->default_value(46), "Maximum vertical disparity")
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

    DiskImageView<float> left_disk_image(left_file_name );
    DiskImageView<float> right_disk_image(right_file_name );

    // The satellite images very like have different intensity
    // ranges. So in order for abs difference cost metric to succeed
    // with PatchMatch .. we need to subtract off the local mean.
    stereo::SubtractedMean filter(15.0);

    // Actually invoke the rasterazation
    {
      vw::Timer corr_timer("Correlation Time");
      block_write_image(tag + "-D.tif",
                        stereo::patch_match(filter.filter(left_disk_image),
                                            filter.filter(right_disk_image),
                                            BBox2i(Vector2i(h_corr_min, v_corr_min),
                                                   Vector2i(h_corr_max, v_corr_max)),
                                            Vector2i(15, 15) /* kernel size */,
                                            -1 /* cross correlation consistency */,
                                            iterations /* number of iterations */),
                        TerminalProgressCallback( "", "Rendering: "));
    }


  } catch (const vw::Exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
