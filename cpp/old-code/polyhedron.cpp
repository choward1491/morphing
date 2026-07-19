/*
** Created by Christian Howard on 4/7/23.
*/

#include <cassert>
#include <utility>
#include <cstdint>
#include <unordered_map>
#include "polyhedron.h"

namespace morphing {

Polyhedron::Polyhedron(const std::vector<Vertex> &vertices, std::vector<std::vector<id_type>> &adjacency_list)
    : vertices_(vertices) {
  build_primal_representation(adjacency_list);
}
Polyhedron &Polyhedron::operator=(const Polyhedron &P) {
  if (this!=&P) {
    num_darts_ = P.num_darts_;
    vertices_.resize(P.num_vertices());
    out_connectivity_.resize(P.num_vertices());

    // setup all the vertices
    for (size_t i = 0; i < P.num_vertices(); ++i) {
      vertices_[i] = P.vertices_[i];
    }

    // setup all the darts
    using dart_ids_t = std::pair<size_t, size_t>;
    std::unordered_map<std::uintptr_t, dart_ids_t> dart_to_loc;
    for (size_t i = 0; i < P.num_vertices(); ++i) {
      vertices_[i] = P.vertices_[i];
      out_connectivity_[i].clear();
      size_t d_idx = 0;
      for (auto &d : P.out_connectivity_[i]) {
        auto &v_tail = vertices_[d.tail()->id];
        auto &v_head = vertices_[d.head()->id];
        dart_to_loc[reinterpret_cast<std::uintptr_t>(&d)] = dart_ids_t(i, d_idx);
        out_connectivity_[i].emplace_back(&v_head, &v_tail);
        d_idx += 1;
      }
    }

    // setup the twins
    setup_twins();

    // setup the face data for each vertex, if necessary
    for (auto &v : vertices_) {
      for (size_t i = 0; i < v.face.size(); ++i) {
        auto [tail_idx, head_idx] = dart_to_loc[reinterpret_cast<std::uintptr_t>(v.face[i])];
        v.face[i] = &out_connectivity_[tail_idx][head_idx];
      }
    }
  }
  return *this;
}

Vertex *Polyhedron::get_vertex(id_type id) {
  return &vertices_[id];
}
Dart *Polyhedron::get_dart(id_type tail_vertex_id, id_type head_vertex_id) {
  return &out_connectivity_[tail_vertex_id][head_vertex_id];
}
std::vector<Dart *> Polyhedron::get_out_darts(id_type tail_vertex_id) {
  std::vector<Dart *> out_darts;
  for (auto &d : out_connectivity_[tail_vertex_id])
    out_darts.push_back(&d);
  return out_darts;
}
const Vertex *Polyhedron::get_vertex(std::uint64_t id) const {
  return &vertices_[id];
}
const Dart *Polyhedron::get_dart(std::uint64_t tail_vertex_id, std::uint64_t head_vertex_id) const {
  return &out_connectivity_[tail_vertex_id][head_vertex_id];
}
std::vector<const Dart *> Polyhedron::get_out_darts(std::uint64_t tail_vertex_id) const {
  std::vector<const Dart*> out;
  for(auto& d: out_connectivity_[tail_vertex_id])
    out.push_back(&d);
  return out;
}
size_t Polyhedron::out_degree(id_type vertex_id) const {
  return out_connectivity_[vertex_id].size();
}
size_t Polyhedron::in_degree(id_type vertex_id) const {
  return out_degree(vertex_id); // assumed to be the same for now
}
size_t Polyhedron::num_vertices() const {
  return vertices_.size();
}
size_t Polyhedron::num_darts() const {
  return num_darts_;
}
std::unique_ptr<Polyhedron> Polyhedron::build_dual() const {
  return std::unique_ptr<Polyhedron>(); // TODO
}
std::unique_ptr<Polyhedron> Polyhedron::build_polar() const {
  return std::unique_ptr<Polyhedron>(); // TODO
}
void Polyhedron::recenter(const arma::vec3 &new_origin) {
  for (auto &v : vertices_) {
    v.pos = (v.pos - new_origin);
  }
}
void Polyhedron::apply_transformation(const arma::mat33 &A) {
  for (auto &v : vertices_) {
    v.pos = A*v.pos;
  }
}
double Polyhedron::radius() const {
  double radius_ = 0.0;
  for (auto &v : vertices_)
    radius_ = std::max(radius_, arma::norm(v.pos));
  return radius_;
}
std::vector<arma::vec3> Polyhedron::get_vertex_coordinates() const {
  std::vector<arma::vec3> coordinates(vertices_.size());
  size_t i = 0;
  for (auto &v : vertices_)
    coordinates[i++] = v.pos;
  return coordinates;
}
void Polyhedron::translate(const arma::vec3 &t) {
  for (auto &v : vertices_) {
    v.pos = (v.pos + t);
  }
}
void Polyhedron::scale(double s) {
  for (auto &v : vertices_) {
    v.pos *= s;
  }
}
void Polyhedron::setup_twins() {
  for (size_t i = 0; i < vertices_.size(); ++i) {
    for (Dart &d : out_connectivity_[i]) {

      // check if we have set the twin yet,
      // if not try to find the twin and set it
      if (d.twin()==nullptr) {
        auto [t, h] = d.vertex_ids();
        for (Dart &e : out_connectivity_[h]) {
          if (e.head()->id==t) {
            e.set_twin(&d);
            d.set_twin(&e);
          }
        }// end for e
      }
    }// end for d
  }// end for i
}
void Polyhedron::build_primal_representation(const std::vector<std::vector<id_type>> &adjacency_list) {
  num_darts_ = 0;

  // initialize the outward directed darts
  out_connectivity_.resize(vertices_.size());

  // setup the darts
  for (size_t i = 0; i < vertices_.size(); ++i) {
    auto &vi = vertices_[i];
    for (auto vid : adjacency_list[i]) {
      auto &v = vertices_[vid];
      num_darts_ += 1;
      out_connectivity_[i].emplace_back(&v, &vi);
    }
  }

  // setup the reverse of each dart, i.e. the twins
  setup_twins();
}
void Polyhedron::compute_polar_normal() {
  // TODO
}
int Polyhedron::rot_index(Dart &d) {
  int idx = 0;
  for (auto &e : out_connectivity_[d.tail()->id]) {
    if (e.head()->id==d.head()->id) {
      return idx;
    } else {
      idx += 1;
    }
  }
  return -1;
}
Dart *Polyhedron::next(Dart *d) {
  auto out_deg = out_degree(d->tail()->id);
  int idx = rot_index(*d);
  return &out_connectivity_[d->tail()->id][(idx + 1)%out_deg];
}
Dart *Polyhedron::prev(Dart *d) {
  auto out_deg = out_degree(d->tail()->id);
  int idx = rot_index(*d);
  return &out_connectivity_[d->tail()->id][(idx + out_deg - 1)%out_deg];
}
void Polyhedron::normalize() {
  for (auto &v : vertices_) {
    v.pos *= 1.0 / arma::norm(v.pos);
  }
}
void Polyhedron::set_vertex_coordinates(const std::vector<arma::vec3> &new_coordinates) {
  assert(new_coordinates.size() == num_vertices());
  for(size_t i = 0; i < num_vertices(); ++i){
    vertices_[i].pos = new_coordinates[i];
  }
}
}
