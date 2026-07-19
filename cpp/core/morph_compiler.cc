#include "core/morph_compiler.h"
#include <algorithm>
#include <utility>

namespace morphing {
namespace core {

protos::CompiledMorphProto MorphCompiler::Compile(
    const DCEL& initial_dcel,
    const std::deque<MorphOperation>& sequence) {
  
  protos::CompiledMorphProto compiled_morph;
  DCEL current_dcel = initial_dcel;

  // Add the initial keyframe (t=0)
  auto* init_frame = compiled_morph.add_keyframes();
  init_frame->set_time(0.0);
  for (size_t i = 0; i < current_dcel.vertices.size(); ++i) {
    if (current_dcel.IsVertexActive(i)) {
      auto* v_proto = init_frame->add_vertices();
      v_proto->set_id(i);
      v_proto->set_type(current_dcel.vertices[i].is_primal ? protos::VERTEX_TYPE_PRIMAL : protos::VERTEX_TYPE_DUAL);
      
      auto cartesian = current_dcel.vertices[i].coord.ToSphereCartesian();
      auto* d_coords = v_proto->mutable_double_coords();
      d_coords->set_x(cartesian.x());
      d_coords->set_y(cartesian.y());
      d_coords->set_z(cartesian.z());

      *v_proto->mutable_extended_coords() = current_dcel.vertices[i].coord.ToProto();
      for (IndexType neighbor_idx : current_dcel.vertices[i].rotation_system_neighbors) {
        v_proto->add_rotation_system_neighbors(neighbor_idx);
      }
    }
  }

  // Reverse the pseudomorph sequence to perform expansion steps
  std::vector<MorphOperation> reversed_sequence(sequence.begin(), sequence.end());
  std::reverse(reversed_sequence.begin(), reversed_sequence.end());

  double step = 0.0;
  double num_steps = static_cast<double>(reversed_sequence.size());
  for (const auto& op : reversed_sequence) {
    step += 1.0;
    
    if (std::holds_alternative<EdgeContraction>(op)) {
      const auto& contract = std::get<EdgeContraction>(op);
      IndexType u = contract.kept_vertex_idx;
      IndexType v = contract.removed_vertex_idx;

      current_dcel.SetVertexActive(v, true);
      current_dcel.SetHalfEdgeActive(contract.edge_idx, true);
      
      // Calculate split coordinates with exact epsilon perturbation
      HomogeneousCoord pos_u = current_dcel.vertices[u].coord;
      HomogeneousCoord pos_v = pos_u;
      pos_v.x += RationalType("1/1000"); // symbolic offset
      
      current_dcel.vertices[v].coord = pos_v;
    } else if (std::holds_alternative<VertexMove>(op)) {
      const auto& move = std::get<VertexMove>(op);
      current_dcel.vertices[move.vertex_idx].coord = move.destination;
    }

    auto* frame = compiled_morph.add_keyframes();
    frame->set_time(step / (num_steps > 0 ? num_steps : 1.0));

    for (size_t i = 0; i < current_dcel.vertices.size(); ++i) {
      if (current_dcel.IsVertexActive(i)) {
        auto* v_proto = frame->add_vertices();
        v_proto->set_id(i);
        v_proto->set_type(current_dcel.vertices[i].is_primal ? protos::VERTEX_TYPE_PRIMAL : protos::VERTEX_TYPE_DUAL);
        
        auto cartesian = current_dcel.vertices[i].coord.ToSphereCartesian();
        auto* d_coords = v_proto->mutable_double_coords();
        d_coords->set_x(cartesian.x());
        d_coords->set_y(cartesian.y());
        d_coords->set_z(cartesian.z());

        *v_proto->mutable_extended_coords() = current_dcel.vertices[i].coord.ToProto();
        for (IndexType neighbor_idx : current_dcel.vertices[i].rotation_system_neighbors) {
          v_proto->add_rotation_system_neighbors(neighbor_idx);
        }
      }
    }
  }

  return compiled_morph;
}

} // namespace core
} // namespace morphing
