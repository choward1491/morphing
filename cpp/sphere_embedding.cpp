/*
** Created by Christian Howard on 4/7/23.
*/
#include <cassert>
#include <algorithm>
#include <set>
#include <cmath>
#include <unordered_set>

#include "sphere_embedding.h"
#include "linear_program.hpp"

namespace morphing {

SphereEmbedding::SphereEmbedding(std::unique_ptr<Polyhedron> &P) : P_(std::move(P)) {
  project_polyhedron_onto_unit_sphere();
}
std::optional<std::pair<std::vector<arma::vec3>,
                        arma::vec3>> SphereEmbedding::try_quad_convexification(const std::array<id_type, 3> &tri1,
                                                                               const std::array<id_type, 3> &tri2) {
  auto num_vertices = P_->num_vertices();

  // form the quad and check that it is nonconvex, returning
  // a direction to slide vertices along if so
  auto quad = form_quad_using_adjacent_triangles(tri1, tri2);
  auto opt = get_reflex_idx_and_direction(quad);
  if (not opt.has_value()) {
    return std::nullopt;
  }

  // extract reflex index and direction
  auto [reflex_idx, direction] = opt.value();

  // extract old coordinates
  auto old_coordinates = P_->get_vertex_coordinates();

  // build new coordinate frame where `direction` points in the positive z direction
  arma::mat::fixed<1, 3> A;
  A.row(0) = direction;
  arma::mat V = arma::null(A);
  arma::mat33 U;
  U.row(0) = V.col(0).t();
  U.row(1) = V.col(1).t();
  U.row(2) = direction;

  // rotate the sphere
  P_->apply_transformation(U);

  // build up linear program
  ::math::linear_program lp;
  lp.obj_is_maximized(true);

  size_t num_face_constraints = 0;
  auto face_ids_list = get_face_id_list();

  size_t num_constraints = 0;
  num_constraints += num_vertices; // number of delta_i constraints
  num_constraints += 2*num_vertices; // number of constraints of form -delta_i <= z_i - z_{i,old} <= delta_i
  num_constraints += num_face_constraints; // face constraints for non-quad faces
  num_constraints += 1; // quad face constraint

  // solve linear program

  // extract coordinates from solution to LP

  // set the sphere back to its original coordinates
  P_->apply_transformation(U.t());

  return std::nullopt;
}
void SphereEmbedding::project_polyhedron_onto_unit_sphere() {
  arma::vec3 origin(arma::fill::zeros);
  P_->recenter(origin);
  P_->normalize();
}
double SphereEmbedding::xy_determinant(const std::array<id_type, 3> &face_ids, int idx) const {

  // grab the vertex positions of the triangular face
  arma::vec3 v0 = P_->get_vertex(face_ids[0])->pos;
  arma::vec3 v1 = P_->get_vertex(face_ids[0])->pos;
  arma::vec3 v2 = P_->get_vertex(face_ids[0])->pos;

  // build up matrix to take determinant of
  double coef = (idx%2==0) ? 1.0 : -1.0;
  switch (idx) {
    case 0: {
      arma::mat22 X{{v1[0], v1[1]}, {v2[0], v2[1]}};
      return coef*arma::det(X);
    }
    case 1: {
      arma::mat22 X{{v0[0], v0[1]}, {v2[0], v2[1]}};
      return coef*arma::det(X);
    }
    case 2:
    default: {
      arma::mat22 X{{v0[0], v0[1]}, {v1[0], v1[1]}};
      return coef*arma::det(X);
    }
  }
}
double SphereEmbedding::simplex_orientation(const std::array<id_type, 3> &face_ids) const {
  // grab the vertex positions of the triangular face
  arma::vec3 v0 = P_->get_vertex(face_ids[0])->pos;
  arma::vec3 v1 = P_->get_vertex(face_ids[0])->pos;
  arma::vec3 v2 = P_->get_vertex(face_ids[0])->pos;

  // form matrix to do determinant of
  arma::mat33 X{{v0[0], v0[1], v0[2]},
                {v1[0], v1[1], v1[2]},
                {v2[0], v2[1], v2[2]}};

  // compute signed area and evaluate the sign value of it
  return arma::sign(arma::det(X));
}
bool SphereEmbedding::is_union_nonconvex(const std::array<id_type, 3> &tri1, const std::array<id_type, 3> &tri2) const {
  assert(triangles_are_adjacent(tri1, tri2));
  auto quad = form_quad_using_adjacent_triangles(tri1, tri2);
  return is_quad_nonconvex(quad);
}
bool SphereEmbedding::is_quad_nonconvex(const std::array<id_type, 4> &quad) const {
  if (auto opt = get_reflex_idx_and_direction(quad)) {
    auto [reflex_idx, dir] = opt.value();
    return reflex_idx > -1;
  }
  return true;
}
bool SphereEmbedding::triangles_are_adjacent(const std::array<id_type, 3> &tri1,
                                             const std::array<id_type, 3> &tri2) {
  int num_shared = 0;
  for (id_type id1 : tri1) {
    for (id_type id2 : tri2) {
      num_shared += (id1==id2);
    }
  }
  return (num_shared==2);
}
double SphereEmbedding::phi_position(const arma::vec3 &pos) {
  return std::atan2(pos[2], arma::norm(pos.rows(0, 1)));
}
std::optional<std::pair<int, arma::vec3>> SphereEmbedding::get_reflex_idx_and_direction(const std::array<id_type,
                                                                                                         4> &quad) const {
  // identify the reflex vertex
  int reflex_idx = -1;
  std::array<double, 4> signs{0.0, 0.0, 0.0, 0.0};
  int num_pos_signs = 0, num_neg_signs = 0;
  for (int i = 0; i < 4; ++i) {
    auto p_i = P_->get_vertex(quad[i])->pos;
    auto p_ip1 = P_->get_vertex(quad[(i + 1)%4])->pos;
    auto p_im1 = P_->get_vertex(quad[(i + 3)%4])->pos;

    arma::vec3 d1 = p_ip1 - p_i;
    arma::vec3 d2 = p_i - p_im1;
    d1 *= (1.0/arma::norm(d1)); // normalize
    d2 *= (1.0/arma::norm(d2)); // normalize

    // compute cross products and dot with -p_i vector to get the sign
    signs[i] = arma::sign(-arma::dot(p_i, arma::cross(d2, d1)));
    num_pos_signs += (signs[i] > 0.0);
    num_neg_signs += (signs[i] < 0.0);
  }

  // check that we have either 3 positive or 3 negative signs
  if (num_pos_signs==3) {
    reflex_idx = std::distance(signs.begin(), std::min_element(signs.begin(), signs.end()));
  } else if (num_neg_signs==3) {
    reflex_idx = std::distance(signs.begin(), std::max_element(signs.begin(), signs.end()));
  }

  // if we found a reflex index, compute a good direction to slide things along
  // so that we can hopefully make the quad stop being nonconvex
  if (reflex_idx >= 0) {

    // compute initial direction
    arma::vec3 p = P_->get_vertex(quad[reflex_idx])->pos;
    arma::vec3 t = P_->get_vertex(quad[(reflex_idx + 2)%4])->pos;
    arma::vec3 direction = p - t;

    // fix up the direction and normalize it
    direction -= (p*arma::dot(direction, p));
    direction *= 1.0/arma::norm(direction);

    return std::pair<int, arma::vec3>{reflex_idx, direction};
  }
  return std::nullopt;
}
std::array<id_type, 4> SphereEmbedding::form_quad_using_adjacent_triangles(const std::array<id_type, 3> &tri1,
                                                                           const std::array<id_type, 3> &tri2) {
  int sidx = 0, eidx = 0;
  std::set<id_type> s1(tri1.begin(), tri1.end()), s2(tri2.begin(), tri2.end());
  for (int i = 0; i < 3; ++i) {
    if (s2.count(tri1[i])==0) {
      sidx = i;
    }
    if (s1.count(tri2[i])==0) {
      eidx = i;
    }
  }
  return std::array<id_type, 4>{tri1[sidx], tri1[(sidx + 1)%3], tri2[eidx], tri1[(sidx + 2)%3]};
}
std::vector<std::vector<id_type>> SphereEmbedding::get_face_id_list() {
  auto num_vertices = P_->num_vertices();

  // add a set of the unvisited edges
  std::unordered_set<Dart *> unvisited_edges;
  for (size_t i = 0; i < num_vertices; ++i) {
    for (size_t j = 0; j < P_->out_degree(i); ++j) {
      Dart *d = P_->get_dart(i, j);
      unvisited_edges.insert(d);
    }
  }

  // construct faces
  std::vector<std::vector<id_type>> out_face_id_lists;
  while (not unvisited_edges.empty()) {

    // init an empty face
    std::vector<id_type> face;

    // remove an arbitary edge from the set
    auto iter = unvisited_edges.begin();
    Dart *d = *iter;
    unvisited_edges.erase(iter);
    face.push_back(d->tail()->id);

    // loop over the cycle representing the face using
    // the rotation system
    Dart *dn = P_->next(d->rev());
    while (dn!=d) {
      face.push_back(dn->tail()->id);
      unvisited_edges.erase(dn);
      dn = P_->next(d->rev());
    }
    std::reverse(face.begin(), face.end());
    out_face_id_lists.push_back(face);
  }// end while

  // return list of faces represented by vertex IDs
  return out_face_id_lists;
}
Polyhedron *SphereEmbedding::get_internal_polyhedron() {
  return P_.get();
}
std::unique_ptr<SphereEmbedding> SphereEmbedding::build_henderson_bad_example(const HendersonInput &input) {
  size_t n = input.num_link_vertices;
  size_t num_vertices = 2*(n + 1);
  size_t pole_vert_id = n;

  // setup data structures to build the vertices
  std::vector<id_type> top_ids, bottom_ids;
  for (id_type i = 0; i < (n + 1); ++i) { top_ids.push_back(i); }
  for (id_type i = n + 1; i < 2*(n + 1); ++i) { bottom_ids.push_back(i); }

  std::vector<Vertex> vertices(num_vertices);
  std::vector<std::vector<id_type>> adjacency_list(num_vertices);
  for (id_type i = 0; i < num_vertices; ++i) {
    vertices[i].id = i;
    vertices[i].type = VertexType::Primal;
  }

  // form the north and south poles
  vertices[top_ids[pole_vert_id]].pos = arma::vec3{0.0, 0.0, 1.0};
  vertices[bottom_ids[pole_vert_id]].pos = arma::vec3{0.0, 0.0, -1.0};

  // construct other vertices
  double r = input.radial_dist_poles;
  double dtheta = (2.0*M_PI)/static_cast<double>(n);
  for (size_t i = 0; i < n; ++i) {
    double theta_i = dtheta*static_cast<double>(i);
    vertices[top_ids[i]].pos = arma::vec3{r*std::cos(theta_i), r*std::sin(theta_i), 1.0};
    vertices[bottom_ids[i]].pos =
        arma::vec3{r*std::cos(theta_i + input.twist), r*std::sin(theta_i + input.twist), -1.0};
  }

  // build the adjacency lists

  // connectivity for north and south pole
  for (size_t i = 0; i < n; ++i) {
    adjacency_list[top_ids[pole_vert_id]].push_back(n - 1 - i);
    adjacency_list[bottom_ids[pole_vert_id]].push_back(n + 1 + i);
  }

  // add connectivity of top vertices not including north pole
  for (size_t i = 0; i < n; ++i) {
    adjacency_list[top_ids[i]].push_back(top_ids[pole_vert_id]);
    adjacency_list[top_ids[i]].push_back(top_ids[(i + 1)%n]);
    adjacency_list[top_ids[i]].push_back(bottom_ids[(i + 1)%n]);
    adjacency_list[top_ids[i]].push_back(bottom_ids[i]);
    adjacency_list[top_ids[i]].push_back(top_ids[(i + n - 1)%n]);
  }

  // add connectivity of bottom vertices not including south pole
  for (size_t i = 0; i < n; ++i) {
    adjacency_list[bottom_ids[i]].push_back(bottom_ids[pole_vert_id]);
    adjacency_list[bottom_ids[i]].push_back(bottom_ids[(i + n - 1)%n]);
    adjacency_list[bottom_ids[i]].push_back(top_ids[(i + n - 1)%n]);
    adjacency_list[bottom_ids[i]].push_back(top_ids[i]);
    adjacency_list[bottom_ids[i]].push_back(bottom_ids[(i + 1)%n]);
  }

  auto P = std::make_unique<Polyhedron>(vertices, adjacency_list);
  return std::make_unique<SphereEmbedding>(P);
}
void SphereEmbedding::rotate(const arma::mat33 &R) {
  P_->apply_transformation(R);
  P_->normalize(); // just in case there are precision issues
}
void SphereEmbedding::transform(const arma::mat33 &T) {
  P_->apply_transformation(T);
  P_->normalize(); // to fix any shearing
}
const Polyhedron *SphereEmbedding::get_internal_polyhedron() const {
  return P_.get();
}
}