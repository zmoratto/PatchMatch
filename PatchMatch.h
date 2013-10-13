#ifndef __PATCH_MATCH__
#define __PATCH_MATCH__

#include <PatchMatchObjective.h>

namespace vw {
namespace stereo {

  template <class ObjectiveT>
  struct PatchMatchStereoBase {
    int test() { return 2; }
  };

  template <class ObjectiveT, class Enable = void>
  class PatchMatchStereo : public PatchMatchStereoBase<ObjectiveT> { /* Not used */ };

  // Floating point accessible version
  template <class ObjectiveT>
  class PatchMatchStereo<ObjectiveT, typename boost::enable_if<boost::is_float<typename ObjectiveT::state_vector_type> >::type> : public PatchMatchStereoBase<ObjectiveT> {
  };

  // Intger accessible version
  template <class ObjectiveT>
  class PatchMatchStereo<ObjectiveT, typename boost::enable_if<boost::is_integral<typename ObjectiveT::state_vector_type> >::type> : public PatchMatchStereoBase<ObjectiveT> {
  };

  template <class Image1T, class Image2T, class ObjectiveT>
  class PatchMatchStereoView {

  };

}} //vw::stereo

#endif//__PATCH_MATCH__
