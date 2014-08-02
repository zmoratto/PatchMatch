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

    void divergence( ImageView<float> const& input_x,
                     ImageView<float> const& input_y,
                     ImageView<float> & output );

    void gradient( ImageView<float> const& input,
                   ImageView<float> & output_x,
                   ImageView<float> & output_y);

    void ROF( ImageView<float> const& input,
              float lambda, int iterations,
              float sigma, float tau, // These are gradient step sizes
              ImageView<float> & output );
    void HuberROF( ImageView<float> const& input,
                   float lambda, int iterations,
                   float alpha, // Huber threshold coeff,
                   float sigma, float tau, // Gradient step sizes
                   ImageView<float> & output );
    void ROF_TVL1( ImageView<float> const& input,
                   float lambda, int iterations,
                   float sigma, float tau, // Gradient step sizes
                   ImageView<float> & output );
  }
}

#endif
