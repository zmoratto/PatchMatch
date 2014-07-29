#ifndef __VW_STEREO_TVMIN3_H__
#define __VW_STEREO_TVMIN3_H__

#include <vw/Math/BBox.h>
#include <vw/Math/Vector.h>
#include <vw/Math/Matrix.h>
#include <vw/Image/ImageMath.h>
#include <vw/Image/ImageView.h>
#include <vw/Image/ImageViewBase.h>
#include <vw/Image/Manipulation.h>

namespace vw {
  namespace stereo {

    void ROF( ImageView<float> const& input,
              float lambda, int iterations,
              float sigma, float tau, // These are gradient step sizes
              ImageView<float> & output );
    void HuberROF( ImageView<float> const& input,
                   ImageView<float> & output );
    void ROF_TVL1( ImageView<float> const& input,
                   ImageView<float> & output );
  }
}

#endif
