#ifndef SPH_MORPH_CORE_PRIMITIVES_H_
#define SPH_MORPH_CORE_PRIMITIVES_H_

#include <boost/multiprecision/cpp_int.hpp>
#include <Eigen/Dense>
#include <string>
#include "protos/geometry.pb.h"

namespace morphing {
namespace core {

using IntType = boost::multiprecision::cpp_int;
using RationalType = boost::multiprecision::cpp_rational;

// A 3D Homogeneous coordinate representation using exact rational types.
struct HomogeneousCoord {
  RationalType x;
  RationalType y;
  RationalType z;

  HomogeneousCoord();
  HomogeneousCoord(RationalType x, RationalType y, RationalType z);

  // Normalize to 3D Cartesian coordinates on the unit sphere
  Eigen::Vector3d ToSphereCartesian() const;

  // Serialize to and from Protobuf representation
  protos::ExtendedPrecisionVectorProto ToProto() const;
  static HomogeneousCoord FromProto(const protos::ExtendedPrecisionVectorProto& proto);
};

// Calculates the orientation determinant of 3 points in 2D projection or 4 points in 3D.
// These predicates return exact sign values: 1 (positive/left), -1 (negative/right), or 0 (collinear/coplanar).
int Orient2D(const HomogeneousCoord& a, const HomogeneousCoord& b, const HomogeneousCoord& c);
int Orient3D(const HomogeneousCoord& a, const HomogeneousCoord& b, const HomogeneousCoord& c, const HomogeneousCoord& d);

} // namespace core
} // namespace morphing

#endif // SPH_MORPH_CORE_PRIMITIVES_H_
