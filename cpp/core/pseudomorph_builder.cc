#include "core/pseudomorph_builder.h"
#include "core/linear_program.h"
#include <absl/status/status.h>

namespace morphing {
namespace core {

PseudomorphBuilder::PseudomorphBuilder(DCEL dcel) : dcel_(std::move(dcel)) {}

absl::Status PseudomorphBuilder::ContractEdge(IndexType edge_idx) {
  if (!dcel_.IsHalfEdgeActive(edge_idx)) {
    return absl::InvalidArgumentError("Edge is inactive or invalid");
  }

  IndexType origin = dcel_.half_edges[edge_idx].origin_vertex;
  IndexType twin = dcel_.half_edges[edge_idx].twin;
  if (twin == kInvalidIndex) {
    return absl::FailedPreconditionError("Edge twin is invalid");
  }
  IndexType target = dcel_.half_edges[twin].origin_vertex;

  dcel_.SetVertexActive(target, false);
  dcel_.SetHalfEdgeActive(edge_idx, false);
  dcel_.SetHalfEdgeActive(twin, false);

  sequence_.push_back(EdgeContraction{edge_idx, origin, target});

  return absl::OkStatus();
}

absl::Status PseudomorphBuilder::MoveVertex(IndexType vertex_idx, const HomogeneousCoord& new_pos) {
  if (!dcel_.IsVertexActive(vertex_idx)) {
    return absl::InvalidArgumentError("Vertex is inactive or invalid");
  }

  std::vector<HomogeneousCoord> constraints;
  IndexType start_edge = dcel_.vertices[vertex_idx].outgoing_half_edge;
  if (start_edge != kInvalidIndex) {
    IndexType curr_edge = start_edge;
    do {
      IndexType twin = dcel_.half_edges[curr_edge].twin;
      if (twin != kInvalidIndex) {
        IndexType neighbor_idx = dcel_.half_edges[twin].origin_vertex;
        if (dcel_.IsVertexActive(neighbor_idx)) {
          constraints.push_back(dcel_.vertices[neighbor_idx].coord);
        }
      }
      curr_edge = dcel_.half_edges[curr_edge].next;
      if (curr_edge == kInvalidIndex) break;
    } while (curr_edge != start_edge);
  }

  auto kernel = SolveDualConeLP(constraints);
  if (!kernel.has_value()) {
    return absl::FailedPreconditionError("Vertex move violates kernel constraints (not strictly planar/hemispherical)");
  }

  dcel_.vertices[vertex_idx].coord = new_pos;
  sequence_.push_back(VertexMove{vertex_idx, new_pos});

  return absl::OkStatus();
}

const std::deque<MorphOperation>& PseudomorphBuilder::GetSequence() const {
  return sequence_;
}

protos::PseudomorphSequenceProto PseudomorphBuilder::ToProto() const {
  protos::PseudomorphSequenceProto proto;
  for (const auto& op : sequence_) {
    auto* op_proto = proto.add_operations();
    if (std::holds_alternative<EdgeContraction>(op)) {
      const auto& contract = std::get<EdgeContraction>(op);
      auto* c_proto = op_proto->mutable_contraction();
      c_proto->set_edge_id(contract.edge_idx);
      c_proto->set_kept_vertex_id(contract.kept_vertex_idx);
      c_proto->set_removed_vertex_id(contract.removed_vertex_idx);
    } else if (std::holds_alternative<EdgeExpansion>(op)) {
      const auto& expand = std::get<EdgeExpansion>(op);
      auto* e_proto = op_proto->mutable_expansion();
      e_proto->set_edge_id(expand.edge_idx);
      e_proto->set_new_vertex_1_id(expand.new_vertex_1_idx);
      e_proto->set_new_vertex_2_id(expand.new_vertex_2_idx);
    } else if (std::holds_alternative<VertexMove>(op)) {
      const auto& move = std::get<VertexMove>(op);
      auto* m_proto = op_proto->mutable_move();
      m_proto->set_vertex_id(move.vertex_idx);
      *m_proto->mutable_extended_dest() = move.destination.ToProto();
    }
  }
  return proto;
}

} // namespace core
} // namespace morphing
