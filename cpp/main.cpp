/*
** Created by Christian Howard on 4/9/23.
*/

#include <vector>
#include <iostream>
#include <cstdio>
#include "sphere_embedding.h"
#include "morph_visualizer.h"
#include <matplot/matplot.h>

int main(int argc, char **argv) {
  auto S = morphing::SphereEmbedding::build_henderson_bad_example({.num_link_vertices = 4});
  morphing::Visualizer vis;
  morphing::Visualizer::VisParams vparams{.azimuth = 0.0, .elevation = 0.0};
  auto fig_handle = vis.draw(*S.get(), true, vparams);
  ::matplot::view(0.0, 90.0);
  while(1){
    double az, el;
    std::cout << "give the azimuth and elevation seperated by a space, in degrees:";
    std::cin >> az >> el;
    if( az == -1.23 ){

      break;
    }
    ::matplot::view(az, el);
  }
  auto axes_handle = fig_handle->current_axes();
  return 0;
}