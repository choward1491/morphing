/*
** Created by Christian Howard on 4/11/23.
*/
#include "morph_visualizer.h"
#include <set>
#include <filesystem>

namespace morphing {
namespace {
  std::vector<double> linspace(double start, double end, int num_points) {
    std::vector<double> out(num_points);
    const double delta = end - start;
    const double inv_n = 1.0 / static_cast<double>(num_points-1);
    for(int i = 0; i < num_points; ++i){
      double alpha = i * inv_n;
      out[i] = start + alpha*delta;
    }
    return out;
  }
}

Visualizer::Visualizer():num_morph_frames_(50), edge_alphas(linspace(0.0, 1.0, num_points)), edge_x(num_points), edge_y(num_points), edge_z(num_points) {

}
::matplot::figure_handle Visualizer::draw(const Polyhedron &P, std::optional<VisParams> vis_params) {

  // set the parameters
  ::matplot::figure_handle fig = nullptr;
  std::optional<double> az, el;
  std::vector<id_type> focus_vertices;
  if( vis_params ){
    fig = vis_params->existing_handle;
    az = vis_params->azimuth;
    el = vis_params->elevation;
    focus_vertices = vis_params->focus_vertices;
  }
  if( !fig ){
    fig = ::matplot::figure();
  }
  ::matplot::axes_handle axes = fig->current_axes();
  if( az ){
    axes->azimuth(az.value());
  }
  if( el ){
    axes->elevation(el.value());
  }

  // plot the polyhedron
  ::matplot::hold(axes, true);

  // draw all the edges
  for(size_t i = 0; i < P.num_vertices(); ++i){
    // loop over the edges and draw edges given vi.id < neighbor.id
    for(const Dart* d: P.get_out_darts(i)){
      if( d->tail()->id < d->head()->id ) { // okay to plot if this is satisfied
        arma::vec3 p1 = d->tail()->pos;
        arma::vec3 p2 = d->head()->pos;
        std::vector<double> x{ p1[0], p2[0] }, y{ p1[1], p2[1] }, z{ p1[2], p2[2] };
        ::matplot::plot3(axes, x, y, z, "k-");
      }
    }
  }

  // draw all the vertices
  std::set<id_type> focus_set(focus_vertices.begin(), focus_vertices.end());
  for(size_t i = 0; i < P.num_vertices(); ++i){
    arma::vec3 pos = P.get_vertex(i)->pos;
    std::vector<double> x{ pos[0] }, y{ pos[1] }, z{ pos[2] };
    if( focus_set.count(i) ) {
      ::matplot::plot3(axes, x, y, z, "ro");
    }else{
      ::matplot::plot3(axes, x, y, z, "bo");
    }
  }

  return fig;
}
::matplot::figure_handle Visualizer::draw(const SphereEmbedding &S, bool draw_back_edges_dotted, std::optional<VisParams> vis_params) {
  // set the parameters
  ::matplot::figure_handle fig = nullptr;
  std::optional<double> az, el;
  std::vector<id_type> focus_vertices;
  if( vis_params ){
    fig = vis_params->existing_handle;
    az = vis_params->azimuth;
    el = vis_params->elevation;
    focus_vertices = vis_params->focus_vertices;
  }
  if( !fig ){
    fig = ::matplot::figure();
  }
  ::matplot::axes_handle axes = fig->current_axes();
  if( az ){
    axes->azimuth(az.value());
  }
  if( el ){
    axes->elevation(el.value());
  }

  // plot the polyhedron
  ::matplot::hold(axes, true);

  // grab the underlying polyhedron
  const Polyhedron* P = S.get_internal_polyhedron();

  // draw all the edges
  for(size_t i = 0; i < P->num_vertices(); ++i){
    // loop over the edges and draw edges given vi.id < neighbor.id
    for(const Dart* d: P->get_out_darts(i)){
      if( d->tail()->id < d->head()->id ) { // okay to plot if this is satisfied
        arma::vec3 p1 = d->tail()->pos;
        arma::vec3 p2 = d->head()->pos;
        draw_shortest_geodesic(axes, p1, p2);
      }
    }
  }

  // draw all the vertices
  std::set<id_type> focus_set(focus_vertices.begin(), focus_vertices.end());
  for(size_t i = 0; i < P->num_vertices(); ++i){
    arma::vec3 pos = P->get_vertex(i)->pos;
    std::vector<double> x{ pos[0] }, y{ pos[1] }, z{ pos[2] };
    if( focus_set.count(i) ) {
      ::matplot::plot3(axes, x, y, z, "ro")->fill(true);
    }else{
      ::matplot::plot3(axes, x, y, z, "bo")->fill(true);
    }
  }

  return fig;
}
void Visualizer::set_num_morph_animation_frames(size_t num_frames) {
  num_morph_frames_ = num_frames;
}
::matplot::figure_handle Visualizer::draw_directional_morph(const SphereEmbedding &S,
                                                            const arma::vec3 &direction,
                                                            std::optional<std::string> save_directory,
                                                            std::optional<VisParams> vis_params) {
  return matplot::figure_handle();
}
void Visualizer::draw_shortest_geodesic(::matplot::axes_handle axes_handle,
                                        const arma::vec3 &p1,
                                        const arma::vec3 &p2,
                                        std::array<float,3> color,
                                        float line_width,
                                        bool draw_back_edges_dotted,
                                        std::optional<double> az,
                                        std::optional<double> el) {
  // get the alphas we will use to construct edges
  size_t idx = 0;
  for(auto alpha: edge_alphas){
    arma::vec3 p = p1*(1.0 - alpha) + p2*alpha;
    p *= 1.0 / arma::norm(p);
    edge_x[idx] = p[0];
    edge_y[idx] = p[1];
    edge_z[idx++] = p[2];
  }
  ::matplot::plot3(axes_handle, edge_x, edge_y, edge_z)->color({color[0], color[1], color[2]}).line_width(line_width);
}
}