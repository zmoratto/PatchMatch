#ifndef __VW_STEREO_PATCHMATCHSIMPLE_H__
#define __VW_STEREO_PATCHMATCHSIMPLE_H__

#include <vw/Image/ImageView.h>
#include <vw/Math/Vector.h>
#include <vw/Math/BBox.h>

void AddDisparityNoise(vw::BBox2f const& search_range,
                       vw::BBox2f const& additive_search_range,
                       vw::BBox2i const& other_image_bbox,
                       vw::ImageView<vw::Vector2f>& disparity);

#endif  // __VW_STEREO_PATCHMATCHSIMPLE_H__
