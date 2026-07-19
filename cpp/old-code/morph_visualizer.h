/*
** Created by Christian Howard on 4/11/23.
*/
#ifndef MORPHING__MORPH_VISUALIZER_H_
#define MORPHING__MORPH_VISUALIZER_H_

#include <optional>
#include <string>

#include <matplot/matplot.h>
#include <armadillo>

#include "polyhedron.h"
#include "sphere_embedding.h"

namespace morphing {

class Visualizer {
 public:
  struct VisParams {
    ::matplot::figure_handle existing_handle = nullptr;
    std::vector<id_type> focus_vertices;
    std::optional<double> azimuth, elevation;
  };

  // ctor/dtor
  Visualizer();
  ~Visualizer() = default;

  // visualizing polyhedron
  ::matplot::figure_handle draw(const Polyhedron &P, std::optional<VisParams> vis_params = std::nullopt);

  // visualizing spheres
  ::matplot::figure_handle draw(const SphereEmbedding &S,
                                bool draw_back_edges_dotted = false,
                                std::optional<VisParams> vis_params = std::nullopt);

  // visualizing morphs
  void set_num_morph_animation_frames(size_t num_frames);
  ::matplot::figure_handle draw_directional_morph(const SphereEmbedding &S, const arma::vec3 &direction,
                                                  std::optional<std::string> save_directory = std::nullopt,
                                                  std::optional<VisParams> vis_params = std::nullopt);

 private:
  size_t num_morph_frames_;

  // variables to cache
  static constexpr int num_points = 100;
  std::vector<double> edge_alphas, edge_x, edge_y, edge_z;

  // useful methods
  void draw_shortest_geodesic(::matplot::axes_handle axes_handle, const arma::vec3 &p1, const arma::vec3 &p2,
                              bool draw_back_edges_dotted = false,
                              std::optional<double> az = std::nullopt,
                              std::optional<double> el = std::nullopt,
                              std::array<float,3> color = {0.0, 0.0, 0.0},
                              float line_width = 1.0);
};

}

#endif //MORPHING__MORPH_VISUALIZER_H_
