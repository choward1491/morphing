/*
** Created by Christian Howard on 4/1/25.
*/
#ifndef MORPHING__MORPH_STEP_H_
#define MORPHING__MORPH_STEP_H_

#include <vector>
#include <armadillo>

namespace morphing {

using VertexEmbeddingRepr = std::vector<arma::vec3>;

class MorphStep {
 public:
  MorphStep() = default;
  virtual ~MorphStep() = default;
  virtual VertexEmbeddingRepr evaluate_morph_at_time(double t) const = 0;
};

}

#endif //MORPHING__MORPH_STEP_H_
