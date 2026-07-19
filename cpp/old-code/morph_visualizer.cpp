/*
** Created by Christian Howard on 4/11/23.
*/
#include "morph_visualizer.h"
#include <set>
#include <filesystem>

namespace morphing {
namespace {
  constexpr double deg2rad = (M_PI/180.0);

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
        draw_shortest_geodesic(axes, p1, p2, draw_back_edges_dotted, az, el);
      }
    }
  }

  // draw all the vertices
  std::set<id_type> focus_set(focus_vertices.begin(), focus_vertices.end());
  for(size_t i = 0; i < P->num_vertices(); ++i){
    arma::vec3 pos = P->get_vertex(i)->pos;
    std::vector<double> x{ pos[0] }, y{ pos[1] }, z{ pos[2] };
    if( focus_set.count(i) ) {
      ::matplot::plot3(axes, x, y, z, "ro")->marker_face_color("r").fill(true);
    }else{
      ::matplot::plot3(axes, x, y, z, "bo")->marker_face_color("b").fill(true);
    }
  }

  /*
  if( az and el ){
    axes->view(az.value(), el.value()+90);
  }
   */
  axes->ylabel("y");
  axes->xlabel("x");

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
                                        bool draw_back_edges_dotted,
                                        std::optional<double> az,
                                        std::optional<double> el,
                                        std::array<float,3> color,
                                        float line_width) {
  // get the alphas we will use to construct edges
  if( not draw_back_edges_dotted || not az.has_value() || not el.has_value() ) {
    size_t idx = 0;
    for (auto alpha : edge_alphas) {
      arma::vec3 p = p1*(1.0 - alpha) + p2*alpha;
      p *= 1.0/arma::norm(p);
      edge_x[idx] = p[0];
      edge_y[idx] = p[1];
      edge_z[idx++] = p[2];
    }
    ::matplot::plot3(axes_handle, edge_x, edge_y, edge_z)->color({color[0], color[1], color[2]}).line_width(line_width);
  }else{
    const double el_needed = (el.value())*deg2rad;
    const double az_needed = az.value()*deg2rad;
    const double s_el = std::sin(el_needed);
    const double c_el = std::cos(el_needed);
    const double s_az = std::sin(az_needed);
    const double c_az = std::cos(az_needed);
    arma::vec3 dir{ s_el*c_az, s_el*s_az, c_el};
    const double dot_p1 = arma::dot(dir, p1), dot_p2 = arma::dot(dir, p2);
    if( dot_p1 >= 0 and dot_p2 >= 0 ){ // plot segment in "front view" as usual
      size_t idx = 0;
      for (auto alpha : edge_alphas) {
        arma::vec3 p = p1*(1.0 - alpha) + p2*alpha;
        p *= 1.0/arma::norm(p);
        edge_x[idx] = p[0];
        edge_y[idx] = p[1];
        edge_z[idx++] = p[2];
      }
      ::matplot::plot3(axes_handle, edge_x, edge_y, edge_z)->color({color[0], color[1], color[2]}).line_width(line_width).line_style("-");
    }else if( dot_p1 < 0 and dot_p2 < 0 ) { // plot segment in "back view"
      size_t idx = 0;
      for (auto alpha : edge_alphas) {
        arma::vec3 p = p1*(1.0 - alpha) + p2*alpha;
        p *= 1.0/arma::norm(p);
        edge_x[idx] = p[0];
        edge_y[idx] = p[1];
        edge_z[idx++] = p[2];
      }
      ::matplot::plot3(axes_handle, edge_x, edge_y, edge_z)->color({color[0], color[1], color[2]}).line_width(line_width).line_style("--");
    }else if( dot_p1 >= 0 and dot_p2 < 0 ) { // start in front and end in back
      size_t idx = 0;
      { // draw portion of edge in front view
        for (auto alpha : edge_alphas) {
          arma::vec3 p = p1*(1.0 - alpha) + p2*alpha;
          p *= 1.0/arma::norm(p);
          if (arma::dot(p, dir) < 0) { break; }
          edge_x[idx] = p[0];
          edge_y[idx] = p[1];
          edge_z[idx++] = p[2];
        }
        edge_x.resize(idx);
        edge_y.resize(idx);
        edge_z.resize(idx);
        ::matplot::plot3(axes_handle, edge_x, edge_y, edge_z)->color({color[0], color[1], color[2]})
            .line_width(line_width).line_style("-");
      }
      edge_x.resize(num_points);
      edge_y.resize(num_points);
      edge_z.resize(num_points);
      { // draw portion of edge in front view
        size_t idx2 = 0;
        for (size_t i = idx; i < num_points; ++i) {
          auto alpha = edge_alphas[i];
          arma::vec3 p = p1*(1.0 - alpha) + p2*alpha;
          p *= 1.0/arma::norm(p);
          edge_x[idx2] = p[0];
          edge_y[idx2] = p[1];
          edge_z[idx2++] = p[2];
        }
        edge_x.resize(idx2);
        edge_y.resize(idx2);
        edge_z.resize(idx2);
        ::matplot::plot3(axes_handle, edge_x, edge_y, edge_z)->color({color[0], color[1], color[2]})
            .line_width(line_width).line_style("--");
      }
      edge_x.resize(num_points);
      edge_y.resize(num_points);
      edge_z.resize(num_points);

    }else if( dot_p1 < 0 and dot_p2 >= 0 ){ // start in back and end in front
      size_t idx = 0;
      { // draw portion of edge in front view
        for (auto alpha : edge_alphas) {
          arma::vec3 p = p2*(1.0 - alpha) + p1*alpha;
          p *= 1.0/arma::norm(p);
          if (arma::dot(p, dir) < 0) { break; }
          edge_x[idx] = p[0];
          edge_y[idx] = p[1];
          edge_z[idx++] = p[2];
        }
        edge_x.resize(idx);
        edge_y.resize(idx);
        edge_z.resize(idx);
        ::matplot::plot3(axes_handle, edge_x, edge_y, edge_z)->color({color[0], color[1], color[2]})
            .line_width(line_width).line_style("-");
      }
      edge_x.resize(num_points);
      edge_y.resize(num_points);
      edge_z.resize(num_points);
      { // draw portion of edge in front view
        size_t idx2 = 0;
        for (size_t i = idx; i < num_points; ++i) {
          auto alpha = edge_alphas[i];
          arma::vec3 p = p2*(1.0 - alpha) + p1*alpha;
          p *= 1.0/arma::norm(p);
          edge_x[idx2] = p[0];
          edge_y[idx2] = p[1];
          edge_z[idx2++] = p[2];
        }
        edge_x.resize(idx2);
        edge_y.resize(idx2);
        edge_z.resize(idx2);
        ::matplot::plot3(axes_handle, edge_x, edge_y, edge_z)->color({color[0], color[1], color[2]})
            .line_width(line_width).line_style("--");
      }
      edge_x.resize(num_points);
      edge_y.resize(num_points);
      edge_z.resize(num_points);
    }

  }
}
}