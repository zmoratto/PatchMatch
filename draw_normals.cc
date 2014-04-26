#include <vw/Core.h>
#include <vw/Image.h>
#include <vw/Math/Vector.h>
#include <vw/FileIO.h>

namespace vw {
  template<> struct PixelFormatID<Vector2f> { static const PixelFormatEnum value = VW_PIXEL_GENERIC_2_CHANNEL; };
  template<> struct PixelFormatID<Vector4f> { static const PixelFormatEnum value = VW_PIXEL_GENERIC_4_CHANNEL; };
}

using namespace vw;

int main( int argc, char **argv ) {
  ImageView<Vector2f> normal_image;
  read_image(normal_image, std::string(argv[1]));
  ImageView<PixelRGB<uint8> > output_image(normal_image.cols(), normal_image.rows());

  for ( int j = 0; j < normal_image.rows(); j++ ) {
    for ( int i = 0; i < normal_image.cols(); i++ ) {
      Vector2f& in = normal_image(i,j);
      output_image(i,j) = PixelRGB<uint8>(PixelHSV<uint8>((255.0/3.14159)*(atan2(in.y(),in.x())+3.14159/2),255,std::min(255.0,255.0*norm_2(in))));
    }
  }

  write_image("hsv_" + std::string(argv[1]), output_image);

  return 0;
}
