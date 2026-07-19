#include "core/linear_program.h"

namespace morphing {
namespace core {

namespace {

HomogeneousCoord CrossProduct(const HomogeneousCoord& a, const HomogeneousCoord& b) {
  return HomogeneousCoord(
      a.y * b.z - a.z * b.y,
      a.z * b.x - a.x * b.z,
      a.x * b.y - a.y * b.x
  );
}

RationalType DotProduct(const HomogeneousCoord& a, const HomogeneousCoord& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

} // namespace

std::optional<HomogeneousCoord> SolveDualConeLP(const std::vector<HomogeneousCoord>& constraints) {
  if (constraints.empty()) {
    return HomogeneousCoord(0, 0, 1);
  }

  std::vector<HomogeneousCoord> rays;
  size_t n = constraints.size();
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      HomogeneousCoord r1 = CrossProduct(constraints[i], constraints[j]);
      HomogeneousCoord r2 = HomogeneousCoord(-r1.x, -r1.y, -r1.z);
      rays.push_back(r1);
      rays.push_back(r2);
    }
  }

  for (const auto& c : constraints) {
    rays.push_back(c);
  }

  HomogeneousCoord interior_ray(0, 0, 0);
  bool found = false;

  for (const auto& ray : rays) {
    if (ray.x == 0 && ray.y == 0 && ray.z == 0) {
      continue;
    }

    bool all_positive = true;
    for (const auto& c : constraints) {
      if (DotProduct(ray, c) <= 0) {
        all_positive = false;
        break;
      }
    }

    if (all_positive) {
      interior_ray.x += ray.x;
      interior_ray.y += ray.y;
      interior_ray.z += ray.z;
      found = true;
    }
  }

  if (found) {
    return interior_ray;
  }

  return std::nullopt;
}

} // namespace core
} // namespace morphing
