#ifndef __VW_STEREO_TVMIN_H__
#define __VW_STEREO_TVMIN_H__

#include <vw/Image/ImageView.h>
#include <vw/Math/Vector.h>

void backdiffx( vw::ImageView<vw::Vector2f> const& input,
                vw::ImageView<vw::Vector2f> & output );
void backdiffy( vw::ImageView<vw::Vector2f> const& input,
                vw::ImageView<vw::Vector2f> & output );
void forwarddiffx( vw::ImageView<vw::Vector2f> const& input,
                   vw::ImageView<vw::Vector2f> & output );
void forwarddiffy( vw::ImageView<vw::Vector2f> const& input,
                   vw::ImageView<vw::Vector2f> & output );

void imROF( vw::ImageView<vw::Vector2f> const& input,
            float lambda, int iterations,
            vw::ImageView<vw::Vector2f> & output );

#endif
