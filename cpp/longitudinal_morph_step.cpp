/*
** Created by Christian Howard on 4/1/25.
*/
#include "longitudinal_morph_step.h"

#include <armadillo>
#include <cassert>
#include <memory>

namespace morphing {

std::unique_ptr<LongitudinalMorphStep> LongitudinalMorphStep::Create(const VertexEmbeddingRepr& G0, const VertexEmbeddingRepr& G1, const arma::vec3& north_pole) {
  assert(G0.size() == G1.size());
  return std::unique_ptr<LongitudinalMorphStep>(new LongitudinalMorphStep(G0, G1, north_pole));
}
VertexEmbeddingRepr LongitudinalMorphStep::evaluate_morph_at_time(double t) const {
  VertexEmbeddingRepr Gnew(G0_.size());
  for(size_t i = 0; i < G0_.size(); ++i){
    Gnew[i] = (1.0 - t)*G0_[i] + t*G1_[i];
  }
  return Gnew;
}
LongitudinalMorphStep::LongitudinalMorphStep(const VertexEmbeddingRepr& G0, const VertexEmbeddingRepr& G1, const arma::vec3& north_pole):G0_(G0), G1_(G1) {
  npole_ = arma::normalise(north_pole);
  for(int i = 0; i < G0.size(); ++i){
    const arma::vec3& vi0 = G0_[i];
    const arma::vec3& vi1 = G1[i];
    const double xyproj_len0 = arma::norm(vi0 - npole_ * (arma::dot(npole_, vi0)));
    const double xyproj_len1 = arma::norm(vi1 - npole_ * (arma::dot(npole_, vi1)));
    const double s = xyproj_len0 / xyproj_len1;
    G1_[i] *= s;
  }
}

}