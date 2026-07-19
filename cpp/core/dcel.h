#ifndef SPH_MORPH_CORE_DCEL_H_
#define SPH_MORPH_CORE_DCEL_H_

#include <vector>
#include <cstdint>
#include "core/primitives.h"

namespace morphing {
namespace core {

using IndexType = std::uint32_t;
constexpr IndexType kInvalidIndex = static_cast<IndexType>(-1);

struct DCELVertex {
  HomogeneousCoord coord;
  IndexType outgoing_half_edge = kInvalidIndex;
  bool is_primal = true;
  std::vector<IndexType> rotation_system_neighbors;
};

struct DCELHalfEdge {
  IndexType origin_vertex = kInvalidIndex;
  IndexType twin = kInvalidIndex;
  IndexType next = kInvalidIndex;
  IndexType prev = kInvalidIndex;
  IndexType face = kInvalidIndex;
};

struct DCELFace {
  IndexType outer_half_edge = kInvalidIndex;
};

class DCEL {
 public:
  DCEL() = default;

  // Add elements
  IndexType AddVertex(HomogeneousCoord coord, bool is_primal = true);
  IndexType AddVertex(HomogeneousCoord coord, std::vector<IndexType> rotation_system_neighbors, bool is_primal = true);
  IndexType AddHalfEdge(IndexType origin_vertex);
  IndexType AddFace(IndexType outer_half_edge);

  // Extract combinatorial faces from the rotation system
  std::vector<std::vector<IndexType>> ExtractFaces() const;

  // Status accessors
  bool IsVertexActive(IndexType vertex_idx) const;
  bool IsHalfEdgeActive(IndexType edge_idx) const;
  bool IsFaceActive(IndexType face_idx) const;

  void SetVertexActive(IndexType vertex_idx, bool active);
  void SetHalfEdgeActive(IndexType edge_idx, bool active);
  void SetFaceActive(IndexType face_idx, bool active);

  // Core arrays
  std::vector<DCELVertex> vertices;
  std::vector<DCELHalfEdge> half_edges;
  std::vector<DCELFace> faces;

 private:
  std::vector<uint8_t> vertex_active_mask_;
  std::vector<uint8_t> half_edge_active_mask_;
  std::vector<uint8_t> face_active_mask_;
};

} // namespace core
} // namespace morphing

#endif // SPH_MORPH_CORE_DCEL_H_
