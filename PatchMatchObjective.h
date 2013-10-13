#ifndef __PATCH_MATCH_OBJECTIVE__
#define __PATCH_MATCH_OBJECTIVE__

#include <vw/Math/Vector.h>

namespace vw {
namespace stereo {

  // Base class to help enforcement of common API
  class PatchMatchObjectiveBase {
  public:
    static const int32 state_vector_size = 0;
    typedef int32 state_vector_type;
    typedef Vector<state_vector_type, state_vector_size> StateVector;
    virtual int32 pixel_buffer() = 0;

    // template <class PAccT1, class PAccT2>
    // state_vector_type evaluate_cost( PAccT1 a, PAccT2 b );
    virtual Vector<state_vector_type,2> evaluate_state_vector( Vector2i const& a, StateVector const& state ) = 0;
    virtual StateVector invert_state_vector( StateVector const& state ) = 0;
  };

  class TranslateIntObjective {
  public:
    static const int32 state_vector_size = 2;
    typedef int32 state_vector_type;
    typedef Vector<state_vector_type, state_vector_size> StateVector;

    TranslateIntObjective( Vector2i const& kernel_size );
    int32 pixel_buffer() const;
  };

  class TranslateFloatObjective {
  public:
    static const int32 state_vector_size = 2;
    typedef float state_vector_type;
  };

}} // vw::stereo

#endif//__PATCH_MATCH_OBJECTIVE__
