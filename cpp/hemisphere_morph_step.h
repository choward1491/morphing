/*
** Created by Christian Howard on 4/1/25.
*/
#ifndef MORPHING__HEMISPHERE_MORPH_STEP_H_
#define MORPHING__HEMISPHERE_MORPH_STEP_H_

#include <armadillo>
#include <memory>

#include "morph_step.h"
#include "sphere_embedding.h"

namespace morphing {

class HemisphereMorphStep: public MorphStep {
 public:
  static std::unique_ptr<HemisphereMorphStep> Create(const VertexEmbeddingRepr& G0, const VertexEmbeddingRepr& G1, const arma::vec3& hemisphere_normal);
  virtual ~HemisphereMorphStep() = default;
  virtual VertexEmbeddingRepr evaluate_morph_at_time(double t) const;

 private:
  HemisphereMorphStep() = delete;
  HemisphereMorphStep(const VertexEmbeddingRepr& G0, const VertexEmbeddingRepr& G1, const arma::vec3& hemisphere_normal);
  VertexEmbeddingRepr G0_, G1_;
  arma::vec3 hemi_normal_;
};

}

#endif //MORPHING__HEMISPHERE_MORPH_STEP_H_
