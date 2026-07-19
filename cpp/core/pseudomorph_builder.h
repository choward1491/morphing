#ifndef SPH_MORPH_CORE_PSEUDOMORPH_BUILDER_H_
#define SPH_MORPH_CORE_PSEUDOMORPH_BUILDER_H_

#include <deque>
#include <variant>
#include <absl/status/status.h>
#include "core/dcel.h"
#include "protos/morph_sequence.pb.h"

namespace morphing {
namespace core {

struct EdgeContraction {
  IndexType edge_idx;
  IndexType kept_vertex_idx;
  IndexType removed_vertex_idx;
};

struct EdgeExpansion {
  IndexType edge_idx;
  IndexType new_vertex_1_idx;
  IndexType new_vertex_2_idx;
};

struct VertexMove {
  IndexType vertex_idx;
  HomogeneousCoord destination;
};

using MorphOperation = std::variant<EdgeContraction, EdgeExpansion, VertexMove>;

class PseudomorphBuilder {
 public:
  explicit PseudomorphBuilder(DCEL dcel);

  // Attempt to perform an edge contraction topological operation
  absl::Status ContractEdge(IndexType edge_idx);

  // Attempt to move a vertex to a new position, checking kernel validation
  absl::Status MoveVertex(IndexType vertex_idx, const HomogeneousCoord& new_pos);

  // Retrieve the recorded sequence of operations
  const std::deque<MorphOperation>& GetSequence() const;

  // Serialize sequence to protobuf
  protos::PseudomorphSequenceProto ToProto() const;

 private:
  DCEL dcel_;
  std::deque<MorphOperation> sequence_;
};

} // namespace core
} // namespace morphing

#endif // SPH_MORPH_CORE_PSEUDOMORPH_BUILDER_H_
