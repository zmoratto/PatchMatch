#ifndef __VW_SURFACE_FIT_W_COST_H__
#define __VW_SURFACE_FIT_W_COST_H__

#include <vw/Image/ImageView.h>
#include <vw/Image/PixelMask.h>
#include <vw/Math/Vector.h>

namespace vw {
  namespace stereo {

    // This is largely a copy of the ARAP work I did .. except now I'm using a quadratic surface over 64 pixels

    void SurfaceFitWCost(ImageView<PixelMask<Vector2f> > surface,
                         ImageView<float> left, ImageView<float> right);

  }
}

#endif //__VW_SURFACE_FIT_W_COST_H__
