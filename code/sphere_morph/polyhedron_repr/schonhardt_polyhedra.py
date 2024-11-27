import numpy as np
import polyhedron_repr as poly_repr
from polyhedron_repr import VertexData


def HendersonBadExample(n: int = 3, twist=np.pi / 3.0, radial_dist_poles=0.5):
    num_vertices = 2 * (n + 1)
    pole_vertex_ID = n
    top_ids = list(range(0, n + 1))
    bottom_ids = list(range(n + 1, 2 * n + 2))

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
    connectivity[top_ids[pole_vertex_ID]] = list(range(n - 1, -1, -1))
    connectivity[bottom_ids[pole_vertex_ID]] = list(range(n + 1, 2 * n + 1))

    # connectivity of top vertices not including north pole
    for i in range(0, n):
        connectivity[top_ids[i]] = [top_ids[pole_vertex_ID], top_ids[(i + 1) % n], bottom_ids[(i + 1) % n],
                                    bottom_ids[i], top_ids[(i + n - 1) % n]]

    # connectivity of bottom vertices not including south pole
    for i in range(0, n):
        connectivity[bottom_ids[i]] = [bottom_ids[pole_vertex_ID], bottom_ids[(i + n - 1) % n],
                                       top_ids[(i + n - 1) % n], top_ids[i], bottom_ids[(i + 1) % n]]

    #for i in range(0, num_vertices):
    #    connectivity[i].reverse()

    # return the bad example
    return poly_repr.Polyhedron(num_vertices=num_vertices, connectivity=connectivity, aux_data=aux_data)

def GeneralizedTwistedPolyhedron(num_layers: int, num_vertices_per_layer: int, amount_relative_rotate: float=0.0):
    num_vertices = num_layers * num_vertices_per_layer + 2
    s_vertex_idx = num_vertices - 2
    t_vertex_idx = num_vertices - 1

    # construct the vertices of each layer
    aux_data = [None] * num_vertices

    # construct baseline theta and height values
    theta = np.linspace(0, 2.0 * np.pi, num_vertices_per_layer+1)
    max_height = 0.5
    heights = np.linspace(-max_height, max_height, num_layers)

    def flat_idx(layer_num, layer_index):
        return layer_index + layer_num * num_vertices_per_layer

    # construct the vertices of the polygon
    for l in range(0, num_layers):
        for i in range(0, num_vertices_per_layer):
            angle = theta[i] + l * amount_relative_rotate
            aux_data[flat_idx(l, i)] = VertexData(position=np.array([np.cos(angle), np.sin(angle), heights[l]]))
    aux_data[s_vertex_idx] = VertexData(position=np.array([0, 0, -max_height]))
    aux_data[t_vertex_idx] = VertexData(position=np.array([0, 0, max_height]))

    # generate the connectivity
    connectivity = dict()

    # add the connectivity information for the interior layers
    nv = num_vertices_per_layer
    for l in range(1, num_layers-1):
        for i in range(0, num_vertices_per_layer):
            neighbors = [flat_idx(l+1, i), flat_idx(l, (i+nv-1) % nv), flat_idx(l-1, (i+nv-1) % nv), flat_idx(l-1, i),
                         flat_idx(l, (i+1) % nv), flat_idx(l+1, (i+1) % nv)]
            connectivity[flat_idx(l, i)] = neighbors

    # add the connectivity info between the s vertex and first layer
    s_neighbors = []
    for i in range(0, num_vertices_per_layer):
        s_neighbors.append(flat_idx(0, i))
    s_neighbors.reverse()
    connectivity[s_vertex_idx] = s_neighbors

    for i in range(0, num_vertices_per_layer):
        neighbors = [flat_idx(1, i), flat_idx(0, (i + nv - 1) % nv), s_vertex_idx, flat_idx(0, (i + 1) % nv), flat_idx(1, (i + 1) % nv)]
        connectivity[flat_idx(0, i)] = neighbors

    # add the connectivity info between the t vertex and last layer
    t_neighbors = []
    ll = num_layers - 1
    for i in range(0, num_vertices_per_layer):
        t_neighbors.append(flat_idx(ll, i))
    connectivity[t_vertex_idx] = t_neighbors

    for i in range(0, num_vertices_per_layer):
        neighbors = [t_vertex_idx, flat_idx(ll, (i+nv-1) % nv), flat_idx(ll-1, (i+nv-1) % nv), flat_idx(ll-1, i), flat_idx(ll, (i+1) % nv)]
        connectivity[flat_idx(ll, i)] = neighbors

    for i in range(0, num_vertices):
        connectivity[i].reverse()

    # return the bad example
    return poly_repr.Polyhedron(num_vertices=num_vertices, connectivity=connectivity, aux_data=aux_data)

def GeneralizedTwistedPolyhedron(num_layers: int, num_vertices_per_layer: int, amount_relative_rotate: float=0.0, amount_subtract_angle_odd: float=0.0, amount_add_vertical_odd: float = 0.0):
    num_vertices = num_layers * num_vertices_per_layer + 2
    s_vertex_idx = num_vertices - 2
    t_vertex_idx = num_vertices - 1

    # construct the vertices of each layer
    aux_data = [None] * num_vertices

    # construct baseline theta and height values
    theta = np.linspace(0, 2.0 * np.pi, num_vertices_per_layer+1)
    max_height = 0.5
    heights = np.linspace(-max_height, max_height, num_layers)

    def flat_idx(layer_num, layer_index):
        return layer_index + layer_num * num_vertices_per_layer

    # construct the vertices of the polygon
    for l in range(0, num_layers):
        for i in range(0, num_vertices_per_layer):
            angle = theta[i] + l * amount_relative_rotate - (i % 2 == 1)*amount_subtract_angle_odd
            aux_data[flat_idx(l, i)] = VertexData(position=np.array([np.cos(angle), np.sin(angle), heights[l] + (i % 2 == 1)*amount_add_vertical_odd]))
    aux_data[s_vertex_idx] = VertexData(position=np.array([0, 0, -max_height]))
    aux_data[t_vertex_idx] = VertexData(position=np.array([0, 0, max_height]))

    # generate the connectivity
    connectivity = dict()

    # add the connectivity information for the interior layers
    nv = num_vertices_per_layer
    for l in range(1, num_layers-1):
        for i in range(0, num_vertices_per_layer):
            neighbors = [flat_idx(l+1, i), flat_idx(l, (i+nv-1) % nv), flat_idx(l-1, (i+nv-1) % nv), flat_idx(l-1, i),
                         flat_idx(l, (i+1) % nv), flat_idx(l+1, (i+1) % nv)]
            connectivity[flat_idx(l, i)] = neighbors

    # add the connectivity info between the s vertex and first layer
    s_neighbors = []
    for i in range(0, num_vertices_per_layer):
        s_neighbors.append(flat_idx(0, i))
    s_neighbors.reverse()
    connectivity[s_vertex_idx] = s_neighbors

    for i in range(0, num_vertices_per_layer):
        neighbors = [flat_idx(1, i), flat_idx(0, (i + nv - 1) % nv), s_vertex_idx, flat_idx(0, (i + 1) % nv), flat_idx(1, (i + 1) % nv)]
        connectivity[flat_idx(0, i)] = neighbors

    # add the connectivity info between the t vertex and last layer
    t_neighbors = []
    ll = num_layers - 1
    for i in range(0, num_vertices_per_layer):
        t_neighbors.append(flat_idx(ll, i))
    connectivity[t_vertex_idx] = t_neighbors

    for i in range(0, num_vertices_per_layer):
        neighbors = [t_vertex_idx, flat_idx(ll, (i+nv-1) % nv), flat_idx(ll-1, (i+nv-1) % nv), flat_idx(ll-1, i), flat_idx(ll, (i+1) % nv)]
        connectivity[flat_idx(ll, i)] = neighbors

    # return the bad example
    return poly_repr.Polyhedron(num_vertices=num_vertices, connectivity=connectivity, aux_data=aux_data)

