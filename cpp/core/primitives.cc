#include "core/primitives.h"
#include <cmath>
#include <utility>

namespace morphing {
namespace core {

HomogeneousCoord::HomogeneousCoord() : x(0), y(0), z(0) {}

HomogeneousCoord::HomogeneousCoord(RationalType x, RationalType y, RationalType z)
    : x(std::move(x)), y(std::move(y)), z(std::move(z)) {}

Eigen::Vector3d HomogeneousCoord::ToSphereCartesian() const {
  double dx = x.convert_to<double>();
  double dy = y.convert_to<double>();
  double dz = z.convert_to<double>();
  double length = std::sqrt(dx * dx + dy * dy + dz * dz);
  if (length == 0.0) {
    return Eigen::Vector3d(0.0, 0.0, 0.0);
  }
  return Eigen::Vector3d(dx / length, dy / length, dz / length);
}

protos::ExtendedPrecisionVectorProto HomogeneousCoord::ToProto() const {
  protos::ExtendedPrecisionVectorProto proto;
  proto.set_x(x.str());
  proto.set_y(y.str());
  proto.set_z(z.str());
  return proto;
}

HomogeneousCoord HomogeneousCoord::FromProto(const protos::ExtendedPrecisionVectorProto& proto) {
  return HomogeneousCoord(RationalType(proto.x()), RationalType(proto.y()), RationalType(proto.z()));
}

int Orient2D(const HomogeneousCoord& a, const HomogeneousCoord& b, const HomogeneousCoord& c) {
  RationalType det = a.x * (b.y - c.y) - a.y * (b.x - c.x) + (b.x * c.y - b.y * c.x);
  if (det > 0) return 1;
  if (det < 0) return -1;
  return 0;
}

int Orient3D(const HomogeneousCoord& a, const HomogeneousCoord& b, const HomogeneousCoord& c, const HomogeneousCoord& d) {
  RationalType ax_dx = a.x - d.x;
  RationalType ay_dy = a.y - d.y;
  RationalType az_dz = a.z - d.z;

  RationalType bx_dx = b.x - d.x;
  RationalType by_dy = b.y - d.y;
  RationalType bz_dz = b.z - d.z;

  RationalType cx_dx = c.x - d.x;
  RationalType cy_dy = c.y - d.y;
  RationalType cz_dz = c.z - d.z;

  RationalType det = ax_dx * (by_dy * cz_dz - bz_dz * cy_dy)
                   - ay_dy * (bx_dx * cz_dz - bz_dz * cx_dx)
                   + az_dz * (bx_dx * cy_dy - by_dy * cx_dx);

  if (det > 0) return 1;
  if (det < 0) return -1;
  return 0;
}

} // namespace core
} // namespace morphing
