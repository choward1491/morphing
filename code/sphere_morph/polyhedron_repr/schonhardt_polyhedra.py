import numpy as np
import polyhedron_repr as poly_repr
from polyhedron_repr import VertexData


def HendersonBadExample(n: int = 3, twist=np.pi / 3.0, radial_dist_poles=0.5):
    num_vertices = 2 * (n + 1)
    pole_vertex_ID = n
    top_ids = list(range(0, n + 1))
    bottom_ids = list(range(n+1, 2 * (n + 1)))

    # generate the aux data with coordinates for the vertices
    aux_data = [None] * num_vertices

    # form the north pole
    aux_data[top_ids[pole_vertex_ID]] = VertexData(position=np.array([0, 0, 1.0]))
    aux_data[bottom_ids[pole_vertex_ID]] = VertexData(position=np.array([0, 0, -1.0]))

    # construct the other vertices
    r = radial_dist_poles
    theta = np.linspace(0, 2 * np.pi, n + 1)
    for i in range(0, n):
        aux_data[top_ids[i]] = VertexData(position=np.array([r * np.cos(theta[i]), r * np.sin(theta[i]), 1.0]))
        aux_data[bottom_ids[i]] = VertexData(
            position=np.array([r * np.cos(theta[i] + twist), r * np.sin(theta[i] + twist), -1.0]))

    # generate the connectivity
    connectivity = dict()

    # connectivity of top and bottom pole vertices
    connectivity[top_ids[pole_vertex_ID]] = list(range(n, -1, -1))
    connectivity[bottom_ids[pole_vertex_ID]] = range(n+1, 2 * n + 1)

    # connectivity of top vertices not including north pole
    for i in range(0, n):
        connectivity[top_ids[i]] = [top_ids[pole_vertex_ID], top_ids[(i + 1) % n], bottom_ids[(i + 1) % n],
                                    bottom_ids[i], top_ids[(i + n - 1) % n]]

    # connectivity of bottom vertices not including south pole
    for i in range(0, n):
        connectivity[bottom_ids[i]] = [bottom_ids[pole_vertex_ID], bottom_ids[(i + n - 1) % n],
                                       top_ids[(i + n - 1) % n], top_ids[i], bottom_ids[(i + 1) % n]]

    # return the bad example
    return poly_repr.Polyhedron(num_vertices=num_vertices, connectivity=connectivity, aux_data=aux_data)