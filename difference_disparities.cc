#include <vw/Core.h>
#include <vw/Math.h>
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

typedef PixelMask<Vector2f> input_type;

struct CompareDispFunc : public vw::ReturnFixedType<PixelRGB<uint8> > {
  typedef PixelRGB<uint8> return_type;
  return_type operator() (input_type const& input,
                          input_type const& ref ) const {
    if ( !is_valid(input) || !is_valid(ref) ) {
      return return_type(0, 0, 0);
    }

    float distance = norm_2(Vector2f(input.child()) - Vector2f(ref.child()));
    if (distance < 1) {
      return return_type(44,123,182);
    } else if (distance < 2) {
      return return_type(171,217,233);
    } else if (distance < 3) {
      return return_type(255,255,191);
    } else if (distance < 4) {
      return return_type(253,174,97);
    }
    return return_type(215,25,28);
  }
};

int main(int argc, char ** argv) {
  if (argc != 3) {
    std::cerr << "Please give two disparity files.";
    return 1;
  }

  boost::shared_ptr<DiskImageResource>
    input_rsrc(DiskImageResource::open(argv[1])),
    ref_rsrc(DiskImageResource::open(argv[2]));
  input_rsrc->set_rescale(false);
  ref_rsrc->set_rescale(false);
  DiskImageView<input_type > input_disp(input_rsrc), ref_disp(ref_rsrc);
  std::cout << input_disp(1024,1024) << " " << ref_disp(1024,1024) << std::endl;

  block_write_image("difference-map.tif",
                    per_pixel_filter(input_disp, ref_disp, CompareDispFunc()),
                    TerminalProgressCallback());

  return 0;
}
