import numpy as np
import counter_example_search as ces
import sphere_repr as sr
import polyhedron_repr as pr

import pulp as pl

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

mpl.use('macosx')
#mpl.use('Qt5Agg')
# mpl.use('GTK3Agg')
# mpl.use('pyqt5')
import matplotlib.pyplot as plt


def counter_example_search_basic_convexifying():
    sphere_embed_gen: ces.SphereEmbeddingGenerator = ces.SphereEmbeddingGenerator(num_layers=2, num_verts_per_layer=10)
    bad_examples = ces.search_for_counter_examples_convexify(sphere_embedding_generator=sphere_embed_gen, num_trials=50)
    if len(bad_examples) == 0:
        print("Found no counter examples so far!")
    else:
        print(f'Found {len(bad_examples)} potential counter examples!')
        for example in bad_examples:
            embedding: sr.SphereEmbedding = example[0]
            print(f"Num vertices embedding = {embedding.polyhedron_repr.num_vertices}")
            output_coordinates = example[2]
            R = example[3]
            pr.rotate(embedding.polyhedron_repr, R)
            num_vertices: int = embedding.polyhedron_repr.num_vertices
            embedding.draw(other_vertices=example[1][0], sphere_color=np.array([1.0, 0.8, 0.8, 1.0]))
            plt.show()

            for i in range(0, num_vertices):
                embedding.polyhedron_repr.get_vertex(i).aux_data.pos = output_coordinates[i][:]
            pr.rotate(embedding.polyhedron_repr, R)
            embedding.draw(other_vertices=example[1][0], sphere_color=np.array([1.0, 0.8, 0.8, 1.0]))
            plt.show()

def counter_example_search_basic_morph_to_south_hemisphere():
    sphere_embed_gen: ces.SphereEmbeddingGenerator = ces.SphereEmbeddingGenerator(num_layers=2, num_verts_per_layer=10)
    bad_examples = ces.search_for_counter_examples_morph_south_hemisphere(sphere_embedding_generator=sphere_embed_gen, num_trials=20)
    if len(bad_examples) == 0:
        print("Found no counter examples so far!")
    else:
        print(f'Found {len(bad_examples)} potential counter examples!')
        for example in bad_examples:
            bad_north_poles = example[0]
            embedding: sr.SphereEmbedding = example[1]
            embedding.draw()
            plt.show()

if __name__ == '__main__':
    #solver_list = pl.listSolvers(onlyAvailable=True)
    #print(solver_list)
    #counter_example_search_basic_convexifying()
    counter_example_search_basic_morph_to_south_hemisphere()
