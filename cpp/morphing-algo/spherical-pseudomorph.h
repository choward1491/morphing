#ifndef SPHERICAL_PSEUDOMORPH_H_
#define SPHERICAL_PSEUDOMORPH_H_

#include "protos/geometry.pb.h"
#include "protos/morph_sequence.pb.h"
#include <absl/status/statusor.h>

namespace morphing {

// Computes a spherical pseudomorph.
class SphericalPseudomorph {
public:
  static absl::StatusOr<SphericalPseudomorph>
  Construct(const protos::IsomorphicGraphEmbeddings &pair_embeddings_proto);

  SphericalPseudomorph() = delete;
  ~SphericalPseudomorph() = default;

  // Returns the initial graph embedding.
  const protos::GraphEmbedding &InitialEmbedding() const;

  // Returns the final graph embedding.
  const protos::GraphEmbedding &FinalEmbedding() const;

  // Number of vertices in the underlying graph.
  size_t NumVertices() const;

  // Number of pseudomorph steps.
  size_t NumPseudomorphSteps() const;

  // Get the k-th pseudomorph step in the sequence.
  const protos::PseudomorphStep &GetStep(int index) const;

private:
  SphericalPseudomorph(
      const protos::IsomorphicGraphEmbeddings &pair_embeddings_proto);

  protos::GraphEmbedding initial_embedding_;
  protos::GraphEmbedding final_embedding_;
  protos::Pseudomorph pseudomorph_;
};

} // namespace morphing

#endif // SPHERICAL_PSEUDOMORPH_H_