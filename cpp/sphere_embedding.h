/*
** Created by Christian Howard on 4/7/23.
*/
#ifndef MORPHING__SPHERE_EMBEDDING_H_
#define MORPHING__SPHERE_EMBEDDING_H_

#include <utility>
#include <memory>
#include <optional>
#include <cstdint>
#include <armadillo>
#include "polyhedron.h"

namespace morphing {

class SphereEmbedding {
 public:

  // useful typedefs
  using quad_convexify_output_t = std::pair<std::vector<arma::vec3>, arma::vec3>;
  struct HendersonInput {
    size_t num_link_vertices = 3;
    double twist = M_PI / 6.0;
    double radial_dist_poles = 0.5;
  };

  // useful build methods
  static std::unique_ptr<SphereEmbedding> build_henderson_bad_example(const HendersonInput& input);

  // ctors/dtors
  explicit SphereEmbedding(std::unique_ptr<Polyhedron>& P);
  ~SphereEmbedding() = default;

  // methods for morphing and manipulating a spherical embedding
  std::optional<quad_convexify_output_t> try_quad_convexification(const std::array<id_type, 3>& tri1,
                                                                  const std::array<id_type, 3>& tri2);

  void rotate(const arma::mat33& R);
  void transform(const arma::mat33& T);

  // method for retrieving internal polyhedron representation
  Polyhedron* get_internal_polyhedron();
  const Polyhedron* get_internal_polyhedron() const;

 private:
  std::unique_ptr<Polyhedron> P_;

  // useful methods
  void project_polyhedron_onto_unit_sphere();
  double xy_determinant(const std::array<id_type, 3>& face_ids, int idx) const;
  double simplex_orientation(const std::array<id_type, 3>& face_ids) const;
  bool is_union_nonconvex(const std::array<id_type, 3>& tri1, const std::array<id_type, 3>& tri2) const;
  bool is_quad_nonconvex(const std::array<id_type, 4>& quad) const;
  static bool triangles_are_adjacent(const std::array<id_type, 3>& tri1, const std::array<id_type, 3>& tri2) ;
  static double phi_position(const arma::vec3& pos) ;
  std::optional<std::pair<int, arma::vec3>> get_reflex_idx_and_direction(const std::array<id_type, 4>& quad) const;
  static std::array<id_type, 4> form_quad_using_adjacent_triangles(const std::array<id_type, 3>& tri1, const std::array<id_type, 3>& tri2);
  std::vector<std::vector<id_type>> get_face_id_list();
};

}

#endif //MORPHING__SPHERE_EMBEDDING_H_
