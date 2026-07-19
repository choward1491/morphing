/*
** Created by Christian Howard on 4/1/25.
*/
#ifndef MORPHING__LONGITUDINAL_MORPH_STEP_H_
#define MORPHING__LONGITUDINAL_MORPH_STEP_H_

#include <armadillo>
#include <memory>

#include "morph_step.h"
#include "sphere_embedding.h"

namespace morphing {

class LongitudinalMorphStep : public MorphStep {
 public:
  static std::unique_ptr<LongitudinalMorphStep> Create(const VertexEmbeddingRepr &G0,
                                                       const VertexEmbeddingRepr &G1,
                                                       const arma::vec3 &north_pole);
  virtual ~LongitudinalMorphStep() = default;
  virtual VertexEmbeddingRepr evaluate_morph_at_time(double t) const;

 private:
  LongitudinalMorphStep() = delete;
  LongitudinalMorphStep(const VertexEmbeddingRepr &G0, const VertexEmbeddingRepr &G1, const arma::vec3 &north_pole);
  VertexEmbeddingRepr G0_;
  VertexEmbeddingRepr G1_;
  arma::vec3 npole_;
};

}

#endif //MORPHING__LONGITUDINAL_MORPH_STEP_H_
