#ifndef SPH_MORPH_CORE_MORPH_COMPILER_H_
#define SPH_MORPH_CORE_MORPH_COMPILER_H_

#include <vector>
#include "core/dcel.h"
#include "core/pseudomorph_builder.h"
#include "protos/morph_sequence.pb.h"

namespace morphing {
namespace core {

class MorphCompiler {
 public:
  MorphCompiler() = default;

  // Compiles a recorded pseudomorph sequence into keyframe morph points.
  // Performs the reverse expansion operations, calculating local epsilon-perturbations.
  protos::CompiledMorphProto Compile(const DCEL& initial_dcel, const std::deque<MorphOperation>& sequence);
};

} // namespace core
} // namespace morphing

#endif // SPH_MORPH_CORE_MORPH_COMPILER_H_
