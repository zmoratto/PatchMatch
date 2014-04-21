#include <vw/Image.h>
#include <vw/Math/Vector.h>
#include <vw/Math/BBox.h>

#include <PatchMatchSimple.h>

#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/lexical_cast.hpp>

using namespace vw;

void AddDisparityNoise(BBox2f const& max_search_range,
                       BBox2f const& additive_search_range,
                       BBox2i const& other_image_bbox,
                       ImageView<Vector2f>& disparity) {
  boost::rand48 gen(0);
  typedef boost::variate_generator<boost::rand48, boost::random::uniform_01<> > vargen_type;
  vargen_type random_source(gen, boost::random::uniform_01<>());

  BBox2f local_search_range_max, local_search_range;
  for (int j = 0; j < disparity.rows(); j++) {
    local_search_range_max.min().y() =
      std::max(max_search_range.min().y(),
               float(other_image_bbox.min().y() - j));
    local_search_range_max.max().y() =
      std::min(max_search_range.max().y(),
               float(other_image_bbox.max().y() - j));
    for (int i = 0; i < disparity.cols(); i++) {
      local_search_range_max.min().x() =
        std::max(max_search_range.min().x(),
                 float(other_image_bbox.min().x() - i));
      local_search_range_max.max().x() =
        std::min(max_search_range.max().x(),
                 float(other_image_bbox.max().x() - i));
      local_search_range = additive_search_range + disparity(i,j);
      local_search_range.crop(local_search_range_max);
      disparity(i,j) =
        elem_prod(Vector2f(random_source(),random_source()),
                  local_search_range.size()) + local_search_range.min();
    }
  }
}
