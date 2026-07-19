#ifndef SPH_MORPH_CORE_LINEAR_PROGRAM_H_
#define SPH_MORPH_CORE_LINEAR_PROGRAM_H_

#include <vector>
#include <optional>
#include "core/primitives.h"

namespace morphing {
namespace core {

// Solves the dual-cone LP for checking if a set of points (up to 5) has a valid kernel point inside a hemisphere.
// Returns a valid HomogeneousCoord kernel point (interior normal vector) if it exists.
std::optional<HomogeneousCoord> SolveDualConeLP(const std::vector<HomogeneousCoord>& constraints);

} // namespace core
} // namespace morphing

#endif // SPH_MORPH_CORE_LINEAR_PROGRAM_H_
