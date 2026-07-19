/*
** Created by Christian Howard on 4/1/25.
*/
#ifndef MORPHING__GLP3R_MORPH_STEP_H_
#define MORPHING__GLP3R_MORPH_STEP_H_

#include <armadillo>
#include <memory>

#include "morph_step.h"
#include "sphere_embedding.h"

namespace morphing {

class GLP3RMorphStep : public MorphStep {
 public:
  static std::unique_ptr<GLP3RMorphStep> Create(const VertexEmbeddingRepr &G0, const arma::mat33 &A);
  virtual ~GLP3RMorphStep() = default;
  virtual VertexEmbeddingRepr evaluate_morph_at_time(double t) const;

 private:
  GLP3RMorphStep() = delete;
  GLP3RMorphStep(const VertexEmbeddingRepr &G0, const arma::mat33 &A);
  VertexEmbeddingRepr G0_;
  arma::cx_vec3 lambda_;
  arma::cx_mat33 lU_, rU_;
};

}

#endif //MORPHING__GLP3R_MORPH_STEP_H_
