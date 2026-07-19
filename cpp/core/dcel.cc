#include "core/dcel.h"

namespace morphing {
namespace core {

#include <set>
#include <algorithm>
#include <utility>

IndexType DCEL::AddVertex(HomogeneousCoord coord, bool is_primal) {
  IndexType idx = static_cast<IndexType>(vertices.size());
  vertices.push_back({coord, kInvalidIndex, is_primal, {}});
  vertex_active_mask_.push_back(1);
  return idx;
}

IndexType DCEL::AddVertex(HomogeneousCoord coord, std::vector<IndexType> rotation_system_neighbors, bool is_primal) {
  IndexType idx = static_cast<IndexType>(vertices.size());
  vertices.push_back({coord, kInvalidIndex, is_primal, std::move(rotation_system_neighbors)});
  vertex_active_mask_.push_back(1);
  return idx;
}

IndexType DCEL::AddHalfEdge(IndexType origin_vertex) {
  IndexType idx = static_cast<IndexType>(half_edges.size());
  half_edges.push_back({origin_vertex, kInvalidIndex, kInvalidIndex, kInvalidIndex, kInvalidIndex});
  half_edge_active_mask_.push_back(1);
  return idx;
}

IndexType DCEL::AddFace(IndexType outer_half_edge) {
  IndexType idx = static_cast<IndexType>(faces.size());
  faces.push_back({outer_half_edge});
  face_active_mask_.push_back(1);
  return idx;
}

bool DCEL::IsVertexActive(IndexType vertex_idx) const {
  return vertex_idx < vertex_active_mask_.size() && vertex_active_mask_[vertex_idx];
}

bool DCEL::IsHalfEdgeActive(IndexType edge_idx) const {
  return edge_idx < half_edge_active_mask_.size() && half_edge_active_mask_[edge_idx];
}

bool DCEL::IsFaceActive(IndexType face_idx) const {
  return face_idx < face_active_mask_.size() && face_active_mask_[face_idx];
}

void DCEL::SetVertexActive(IndexType vertex_idx, bool active) {
  if (vertex_idx < vertex_active_mask_.size()) {
    vertex_active_mask_[vertex_idx] = active ? 1 : 0;
  }
}

void DCEL::SetHalfEdgeActive(IndexType edge_idx, bool active) {
  if (edge_idx < half_edge_active_mask_.size()) {
    half_edge_active_mask_[edge_idx] = active ? 1 : 0;
  }
}

void DCEL::SetFaceActive(IndexType face_idx, bool active) {
  if (face_idx < face_active_mask_.size()) {
    face_active_mask_[face_idx] = active ? 1 : 0;
  }
}

std::vector<std::vector<IndexType>> DCEL::ExtractFaces() const {
  std::vector<std::vector<IndexType>> extracted_faces;
  std::set<std::pair<IndexType, IndexType>> visited_half_edges;

  for (IndexType u = 0; u < vertices.size(); ++u) {
    if (!IsVertexActive(u)) continue;

    const auto& u_neighbors = vertices[u].rotation_system_neighbors;
    for (IndexType v : u_neighbors) {
      if (!IsVertexActive(v)) continue;

      std::pair<IndexType, IndexType> edge = {u, v};
      if (visited_half_edges.count(edge) > 0) continue;

      std::vector<IndexType> face;
      IndexType curr_u = u;
      IndexType curr_v = v;

      bool valid_face = true;
      while (visited_half_edges.count({curr_u, curr_v}) == 0) {
        visited_half_edges.insert({curr_u, curr_v});
        face.push_back(curr_u);

        const auto& v_neighbors = vertices[curr_v].rotation_system_neighbors;
        auto it = std::find(v_neighbors.begin(), v_neighbors.end(), curr_u);
        if (it == v_neighbors.end()) {
          valid_face = false;
          break;
        }

        size_t idx = std::distance(v_neighbors.begin(), it);
        IndexType next_w = v_neighbors[(idx + 1) % v_neighbors.size()];

        curr_u = curr_v;
        curr_v = next_w;
      }

      if (valid_face && !face.empty()) {
        extracted_faces.push_back(std::move(face));
      }
    }
  }

  return extracted_faces;
}

} // namespace core
} // namespace morphing
