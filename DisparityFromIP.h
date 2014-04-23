#ifndef __VW_STEREO_DISPARITYFROMIP_H__
#define __VW_STEREO_DISPARITYFROMIP_H__

#include <vw/Math/Vector.h>
#include <vw/Image/ImageView.h>

void DisparityFromIP(std::string const& match_filename,
                     vw::ImageView<vw::Vector2f> const& output,
                     bool swap_order = false);

#endif //__VW_STEREO_DISPARITYFROMIP_H__
