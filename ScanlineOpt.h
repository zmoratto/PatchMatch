#include <vw/Image/ImageView.h>
#include <vw/Math/Vector.h>

#include <asp/Core/BlobIndexThreaded.h>

void scanline_fill( BlobIndexThreaded const& blobs_in_a,
                    vw::ImageView<float> const& a,
                    vw::ImageView<float> const& b,
                    vw::ImageView<vw::Vector2f> & disparity );
