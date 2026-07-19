#include "core/dcel.h"
#include "gtest/gtest.h"
#include <set>
#include <vector>
#include <algorithm>

namespace morphing {
namespace core {

TEST(DCELTest, ExtractFacesTetrahedron) {
  DCEL dcel;

  HomogeneousCoord p0(0, 0, 1);
  HomogeneousCoord p1(1, 0, 0);
  HomogeneousCoord p2(0, 1, 0);
  HomogeneousCoord p3(-1, -1, -1);

  // CCW rotation system for a tetrahedron
  dcel.AddVertex(p0, {1, 2, 3});
  dcel.AddVertex(p1, {0, 3, 2});
  dcel.AddVertex(p2, {0, 1, 3});
  dcel.AddVertex(p3, {0, 2, 1});

  auto faces = dcel.ExtractFaces();

  auto normalize_face = [](const std::vector<IndexType>& face) {
    if (face.empty()) return face;
    auto min_it = std::min_element(face.begin(), face.end());
    std::vector<IndexType> normalized;
    normalized.insert(normalized.end(), min_it, face.end());
    normalized.insert(normalized.end(), face.begin(), min_it);
    return normalized;
  };

  std::set<std::vector<IndexType>> normalized_faces;
  for (const auto& f : faces) {
    normalized_faces.insert(normalize_face(f));
  }

  std::set<std::vector<IndexType>> expected_faces = {
    {0, 1, 2},
    {0, 2, 3},
    {0, 3, 1},
    {1, 3, 2}
  };

  EXPECT_EQ(normalized_faces.size(), expected_faces.size());
  for (const auto& ef : expected_faces) {
    EXPECT_TRUE(normalized_faces.count(ef) > 0);
  }
}

} // namespace core
} // namespace morphing
