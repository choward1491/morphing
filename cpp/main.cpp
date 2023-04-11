/*
** Created by Christian Howard on 4/9/23.
*/

#include <vector>
#include <cstdio>
#include "sphere_embedding.h"
#include "morph_visualizer.h"
#include <matplot/matplot.h>

int main(int argc, char **argv) {
  auto S = morphing::SphereEmbedding::build_henderson_bad_example({.num_link_vertices = 4});
  morphing::Visualizer vis;
  auto fig_handle = vis.draw(*S.get());
  ::matplot::show(fig_handle);
  return 0;
}