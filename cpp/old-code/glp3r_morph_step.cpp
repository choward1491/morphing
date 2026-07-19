/*
** Created by Christian Howard on 4/1/25.
*/
#include "glp3r_morph_step.h"

#include <cassert>
#include <armadillo>
#include <memory>

#include "morph_step.h"
#include "sphere_embedding.h"

namespace morphing {

  std::unique_ptr<GLP3RMorphStep> GLP3RMorphStep::Create(const VertexEmbeddingRepr& G0, const arma::mat33& A) {
    assert(arma::det(A) > 0.0);
    return std::unique_ptr<GLP3RMorphStep>(new GLP3RMorphStep(G0, A));
  }
  VertexEmbeddingRepr GLP3RMorphStep::evaluate_morph_at_time(double t) const {
    arma::cx_vec3 ones{1.0, 1.0, 1.0};
    arma::cx_vec3 lambda_t = (1.0 - t) * ones + t * lambda_;
    arma::mat33 B_t = arma::real(lU_ * arma::diagmat(lambda_t) * rU_);
    VertexEmbeddingRepr Gnew(G0_.size());
    for(size_t i = 0; i < G0_.size(); ++i){
      Gnew[i] = B_t * G0_[i];
    }
    return Gnew;
  }
  GLP3RMorphStep::GLP3RMorphStep(const VertexEmbeddingRepr& G0, const arma::mat33& A) {
    G0_ = G0;
    arma::eig_gen(lambda_,lU_, rU_);
  }

}