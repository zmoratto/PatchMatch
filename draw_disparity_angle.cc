#include <vw/Core.h>
#include <vw/Image.h>
#include <vw/Math.h>
#include <vw/FileIO.h>

namespace vw {
  template<> struct PixelFormatID<Vector2f> { static const PixelFormatEnum value = VW_PIXEL_GENERIC_2_CHANNEL; };
  template<> struct PixelFormatID<Vector4f> { static const PixelFormatEnum value = VW_PIXEL_GENERIC_4_CHANNEL; };
}

using namespace vw;

int main( int argc, char **argv ) {
  ImageView<Vector2f> disparity_image;
  read_image(disparity_image, std::string(argv[1]));

  Vector2f mean = mean_pixel_value(disparity_image);
  std::cout << "Mean disparity: " << mean << std::endl;
  ImageView<PixelRGB<uint8> > output_image(disparity_image.cols(), disparity_image.rows());

  double angle = 0;
  for ( int j = 0; j < disparity_image.rows(); j++ ) {
    for ( int i = 0; i < disparity_image.cols(); i++ ) {
      Vector2f in = disparity_image(i,j) - mean;
      output_image(i,j) =
        PixelRGB<uint8>(PixelHSV<uint8>(255.0*(atan(in.y()/in.x())/3.14159 + 3.14159/2),255,255));
      angle += atan(in.y()/in.x());
    }
  }
  angle /= output_image.cols() * output_image.rows();
  std::cout << "Angle is: " << angle * 360 / 3.14159 << std::endl;

  Matrix2x2f r(cos(-angle), -sin(-angle), sin(-angle), cos(-angle));
  std::cout << r << std::endl;
  Vector2f t = -r * mean;
  BBox2f aligned_region;
  for ( int j = 0; j < disparity_image.rows(); j++ ) {
    for ( int i = 0; i < disparity_image.cols(); i++ ) {
      Vector2f a = r * disparity_image(i,j) + t;
      aligned_region.grow(a);
    }
  }
  std::cout << "Aligned search region: " << aligned_region << std::endl;

  write_image("angle_" + std::string(argv[1]), output_image);

  return 0;
}
