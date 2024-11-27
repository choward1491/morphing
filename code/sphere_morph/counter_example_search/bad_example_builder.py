import random

import numpy as np
import sphere_repr as sr
import polyhedron_repr as pr

class SphereEmbeddingGenerator:
    def __init__(self, num_layers = 2, num_verts_per_layer = 4, seed = 17):
        self.num_layers = num_layers
        self.num_verts_per_layer = num_verts_per_layer
        self.num_layers_range = (2, 3)
        self.num_verts_per_layer_range = (4, 4)
        self.rotation_range = (0.0, np.pi/2.2)
        self.vertical_add_range = (0.0, 0.1)
        random.seed = seed
        np.random.seed = seed

    def sample(self):
        while True:
            num_layers = random.randint(self.num_layers_range[0], self.num_layers_range[1])
            num_verts_per_layer = random.randint(self.num_verts_per_layer_range[0], self.num_verts_per_layer_range[1])
            rotation = self.rotation_range[1] * np.random.rand(1)[0]
            vertical_delta = self.vertical_add_range[1] * np.random.rand(1)[0]

            bad_polyhedron: pr.Polyhedron = pr.GeneralizedTwistedPolyhedron(num_layers=num_layers,
                                                                            num_vertices_per_layer=num_verts_per_layer,
                                                                            amount_relative_rotate=rotation,
                                                                            amount_subtract_angle_odd=0.0,
                                                                            amount_add_vertical_odd=vertical_delta)

            # construct random linear transformation
            #A = np.random.rand(3, 3)

            # transform the polyhedron
            #pr.rotate(bad_polyhedron, A)

            embedding: sr.SphereEmbedding = sr.SphereEmbedding(polyhedron=bad_polyhedron)
            if embedding._all_signed_vols_have_same_sign():
                return embedding

        # form spherical embedding based on it
        return None