#include <vw/Core.h>
#include <vw/Image/ImageView.h>
#include <vw/Math/Vector.h>
#include <vw/FileIO.h>

#include <DisparityFromIP.h>

namespace vw {
  template<> struct PixelFormatID<Vector2f> { static const PixelFormatEnum value = VW_PIXEL_GENERIC_2_CHANNEL; };
  template<> struct PixelFormatID<Vector4f> { static const PixelFormatEnum value = VW_PIXEL_GENERIC_4_CHANNEL; };
}

using namespace vw;

int main( int argc, char ** argv ) {
  ImageView<Vector2f> disparity_image(960, 960);
  DisparityFromIP("arctic/asp_al-L.crop.8__asp_al-R.crop.8.match",
                  disparity_image);

  // Write output image
  write_image("guessed_disparity.tif", disparity_image);

  return 0;
}
