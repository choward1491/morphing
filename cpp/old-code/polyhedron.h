/*
** Created by Christian Howard on 4/7/23.
*/
#ifndef MORPHING__POLYHEDRON_H_
#define MORPHING__POLYHEDRON_H_

#include <vector>

#include <armadillo>

#include "primitives.h"

namespace morphing {

class Polyhedron {
 public:

  // ctor/dtor
  explicit Polyhedron(const std::vector<Vertex> &vertices,
                      std::vector<std::vector<std::uint64_t>> &adjacency_list);
  Polyhedron& operator=(const Polyhedron& P);
  ~Polyhedron() = default;

  // methods for accessing underlying graph information
  [[nodiscard]] Vertex *get_vertex(std::uint64_t id);
  [[nodiscard]] const Vertex *get_vertex(std::uint64_t id) const;
  [[nodiscard]] Dart *get_dart(std::uint64_t tail_vertex_id, std::uint64_t head_vertex_id);
  [[nodiscard]] const Dart *get_dart(std::uint64_t tail_vertex_id, std::uint64_t head_vertex_id) const;
  [[nodiscard]] std::vector<Dart *> get_out_darts(std::uint64_t tail_vertex_id);
  [[nodiscard]] std::vector<const Dart *> get_out_darts(std::uint64_t tail_vertex_id) const;
  [[nodiscard]] size_t out_degree(std::uint64_t vertex_id) const;
  [[nodiscard]] size_t in_degree(std::uint64_t vertex_id) const;
  [[nodiscard]] size_t num_vertices() const;
  [[nodiscard]] size_t num_darts() const;
  [[nodiscard]] Dart* next(Dart* d);
  [[nodiscard]] Dart* prev(Dart* d);

  // methods for building a dual and polar of the polyhedron
  [[nodiscard]] std::unique_ptr<Polyhedron> build_dual() const;
  [[nodiscard]] std::unique_ptr<Polyhedron> build_polar() const;

  // methods for accessing/manipulating the geometry of the polyhedron
  void recenter(const arma::vec3 &new_origin);
  void translate(const arma::vec3 &t);
  void scale(double s);
  void normalize();
  void apply_transformation(const arma::mat33 &A);
  [[nodiscard]] double radius() const;
  [[nodiscard]] std::vector<arma::vec3> get_vertex_coordinates() const;
  void set_vertex_coordinates(const std::vector<arma::vec3>& new_coordinates);

 private:
  size_t num_darts_;
  std::vector<Vertex> vertices_;
  std::vector<std::vector<Dart>> out_connectivity_;

  // useful methods
  void build_primal_representation(const std::vector<std::vector<std::uint64_t>>& adjacency_list);
  void compute_polar_normal();
  int rot_index(Dart& d);
  void setup_twins();


};

}

#endif //MORPHING__POLYHEDRON_H_
