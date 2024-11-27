# This is a sample Python script.
import numpy as np
import scipy as spy
from scipy.stats import ortho_group
import cvxpy as cp
import polyhedron_repr as poly_repr
import sphere_repr
import sphere_triangulation as stri
from polyhedron_repr import VertexData
from polyhedron_repr import HendersonBadExample
from polyhedron_repr import GeneralizedTwistedPolyhedron
from sphere_repr import *
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

mpl.use('macosx')
# mpl.use('Qt5Agg')
# mpl.use('GTK3Agg')
# mpl.use('pyqt5')
import matplotlib.pyplot as plt

def test_edge_marking_method():
    e = [np.array([1.0, 0, 0]), np.array([1.0, 1.0, 0])]
    e1 = [np.array([1.0, 0.5, 0]), np.array([1.0, 2.0, 0])]
    e2 = [np.array([1.0, 2.0, 0]), np.array([1.0, 3.0, 0])]
    e3 = [np.array([1.0, 0.5, -1]), np.array([1.0, 0.5, 1])]
    e4 = [np.array([1.0, 2.0, -1]), np.array([1.0, 2.0, 1.0])]
    e5 = [np.array([1.0, -2, 1]), np.array([1.0, -2.0, -1.0])]
    e6 = [np.array([1.0, 1.0, 0]), np.array([1.0, 0.5, 1.0])]
    e7 = [np.array([1.0, 2.0, 0]), np.array([1.0, 0.5, 1.0])]
    e8 = [np.array([1.0, 0.5, -1.0]), np.array([1.0, 1.0, 0])]
    e9 = [np.array([1.0, 0, 0]), np.array([1.0, 0.5, -1.0])]
    e10 = [np.array([1.0, 0.5, 1.0]), np.array([1.0, 0, 0])]

    flag1 = stri.test_edges_cross(e, e1)
    flag2 = stri.test_edges_cross(e, e2)
    flag3 = stri.test_edges_cross(e, e3)
    flag4 = stri.test_edges_cross(e, e4)
    flag5 = stri.test_edges_cross(e, e5)
    flag6 = stri.test_edges_cross(e, e6)
    flag7 = stri.test_edges_cross(e, e7)
    flag8 = stri.test_edges_cross(e, e8)
    flag9 = stri.test_edges_cross(e, e9)
    flag10 = stri.test_edges_cross(e, e10)
    print("testing")

def test_edge_triangulation_with_awartani_henderson():
    bad_polyhedron: poly_repr.Polyhedron = HendersonBadExample(n=6, twist=np.pi / 6)
    embedding: SphereEmbedding = SphereEmbedding(bad_polyhedron)
    fig = embedding.draw(do_draw_polyhedron=False)
    plt.show()

    # insert some edge
    new_edge = [np.array([0.1, 0, 0.3]), np.array([0.1, 0, -0.3])]
    #new_edge = [np.array([0.5, -0.5, 0.5]), np.array([0.5, 0.5, 0.5])]
    init_vertices = []
    for i in range(0, embedding.polyhedron_repr.num_vertices):
        init_vertices.append(np.copy(embedding.polyhedron_repr.get_vertex(i).aux_data.pos))
    init_triangles = embedding._get_facet_id_lists()
    new_vertices, new_triangles = stri.insert_edge_and_triangulate(initial_vertices=init_vertices,
                                                                          initial_triangles=init_triangles,
                                                                          new_edge=new_edge)
    vertices, adj_list = stri.convert_repr_to_adj_list(new_vertices, new_triangles)
    vertex_data = []
    for v in vertices:
        vertex_data.append(VertexData(position=v))
    new_polyhedron: poly_repr.Polyhedron = poly_repr.Polyhedron(num_vertices=len(vertices), connectivity=adj_list, aux_data=vertex_data)
    embedding2: SphereEmbedding = SphereEmbedding(new_polyhedron)
    fig = embedding2.draw(do_draw_polyhedron=False)
    plt.show()

def test_edge_triangulation_with_awartani_henderson2():
    bad_polyhedron: poly_repr.Polyhedron = HendersonBadExample(n=6, twist=np.pi / 6)
    embedding: SphereEmbedding = SphereEmbedding(bad_polyhedron)
    fig = embedding.draw(do_draw_polyhedron=False)
    plt.show()

    # insert some edge
    new_edge = [np.array([1, 0, 0.1]), np.array([0, 1, 0.1])]
    #new_edge = [np.array([0.5, -0.5, 0.5]), np.array([0.5, 0.5, 0.5])]
    init_vertices = []
    for i in range(0, embedding.polyhedron_repr.num_vertices):
        init_vertices.append(np.copy(embedding.polyhedron_repr.get_vertex(i).aux_data.pos))
    init_triangles = embedding._get_facet_id_lists()
    new_vertices, new_triangles = stri.insert_edge_and_triangulate(initial_vertices=init_vertices,
                                                                          initial_triangles=init_triangles,
                                                                          new_edge=new_edge)
    vertices, adj_list = stri.convert_repr_to_adj_list(new_vertices, new_triangles)
    vertex_data = []
    for v in vertices:
        vertex_data.append(VertexData(position=v))
    new_polyhedron: poly_repr.Polyhedron = poly_repr.Polyhedron(num_vertices=len(vertices), connectivity=adj_list, aux_data=vertex_data)
    embedding2: SphereEmbedding = SphereEmbedding(new_polyhedron)
    fig = embedding2.draw(do_draw_polyhedron=False)
    plt.show()

def test_edge_triangulation_with_tetrahedron():
    nv = 4  # number of vertices
    connectivity = [[1, 2, 3], [2, 0, 3], [0, 1, 3], [0, 2, 1]]
    primal_data = [
        VertexData(position=np.array([-0.1, -0.1, -0.1])),
        VertexData(position=np.array([1.0, 0.0, 0.0])),
        VertexData(position=np.array([0.0, 1.0, 0.0])),
        VertexData(position=np.array([0.0, 0.0, 1.0]))
    ]
    tetrahedron: poly_repr.Polyhedron = poly_repr.Polyhedron(nv, connectivity, primal_data)
    embedding: SphereEmbedding = SphereEmbedding(tetrahedron)
    fig = embedding.draw(do_draw_polyhedron=False)
    plt.show()

    # insert some edge
    new_edge = [np.array([0.1, 0.1, 0.8]), np.array([0.1, 0.1, -0.8])]
    init_vertices = []
    for i in range(0, embedding.polyhedron_repr.num_vertices):
        init_vertices.append(np.copy(embedding.polyhedron_repr.get_vertex(i).aux_data.pos))
    init_triangles = embedding._get_facet_id_lists()
    new_vertices, new_triangles = stri.insert_edge_and_triangulate(initial_vertices=init_vertices,
                                                                          initial_triangles=init_triangles,
                                                                          new_edge=new_edge)
    vertices, adj_list = stri.convert_repr_to_adj_list(new_vertices, new_triangles)
    vertex_data = []
    for v in vertices:
        vertex_data.append(VertexData(position=v))
    new_polyhedron: poly_repr.Polyhedron = poly_repr.Polyhedron(num_vertices=len(vertices), connectivity=adj_list, aux_data=vertex_data)
    embedding2: SphereEmbedding = SphereEmbedding(new_polyhedron)
    fig = embedding2.draw(do_draw_polyhedron=False)
    plt.show()

def test_edge_triangulation_with_tetrahedron2():
    nv = 4  # number of vertices
    connectivity = [[1, 2, 3], [2, 0, 3], [0, 1, 3], [0, 2, 1]]
    primal_data = [
        VertexData(position=np.array([-0.1, -0.1, -0.1])),
        VertexData(position=np.array([1.0, 0.0, 0.0])),
        VertexData(position=np.array([0.0, 1.0, 0.0])),
        VertexData(position=np.array([0.0, 0.0, 1.0]))
    ]
    tetrahedron: poly_repr.Polyhedron = poly_repr.Polyhedron(nv, connectivity, primal_data)
    embedding: SphereEmbedding = SphereEmbedding(tetrahedron)
    fig = embedding.draw(do_draw_polyhedron=False)
    plt.show()

    # insert some edge
    new_edge = [np.array([0.5, -0.5, 0.5]), np.array([0.5, 0.5, 0.5])]
    init_vertices = []
    for i in range(0, embedding.polyhedron_repr.num_vertices):
        init_vertices.append(np.copy(embedding.polyhedron_repr.get_vertex(i).aux_data.pos))
    init_triangles = embedding._get_facet_id_lists()
    new_vertices, new_triangles = stri.insert_edge_and_triangulate(initial_vertices=init_vertices,
                                                                          initial_triangles=init_triangles,
                                                                          new_edge=new_edge)
    vertices, adj_list = stri.convert_repr_to_adj_list(new_vertices, new_triangles)
    vertex_data = []
    for v in vertices:
        vertex_data.append(VertexData(position=v))
    new_polyhedron: poly_repr.Polyhedron = poly_repr.Polyhedron(num_vertices=len(vertices), connectivity=adj_list, aux_data=vertex_data)
    embedding2: SphereEmbedding = SphereEmbedding(new_polyhedron)
    fig = embedding2.draw(do_draw_polyhedron=False)
    plt.show()

def test_edge_triangulation_with_tetrahedron2_perturb():
    nv = 4  # number of vertices
    connectivity = [[1, 2, 3], [2, 0, 3], [0, 1, 3], [0, 2, 1]]
    primal_data = [
        VertexData(position=np.array([-0.1, -0.1, -0.1])),
        VertexData(position=np.array([1.0, 0.0, 0.0])),
        VertexData(position=np.array([0.0, 1.0, 0.0])),
        VertexData(position=np.array([0.0, 0.0, 1.0]))
    ]
    tetrahedron: poly_repr.Polyhedron = poly_repr.Polyhedron(nv, connectivity, primal_data)
    embedding: SphereEmbedding = SphereEmbedding(tetrahedron)
    fig = embedding.draw(do_draw_polyhedron=False)
    plt.show()

    # insert some edge
    new_edge = [np.array([0.5, -0.5, 0.5]), np.array([0.5, 0.5, 0.8])]
    init_vertices = []
    for i in range(0, embedding.polyhedron_repr.num_vertices):
        init_vertices.append(np.copy(embedding.polyhedron_repr.get_vertex(i).aux_data.pos))
    init_triangles = embedding._get_facet_id_lists()
    new_vertices, new_triangles = stri.insert_edge_and_triangulate(initial_vertices=init_vertices,
                                                                          initial_triangles=init_triangles,
                                                                          new_edge=new_edge)
    vertices, adj_list = stri.convert_repr_to_adj_list(new_vertices, new_triangles)
    vertex_data = []
    for v in vertices:
        vertex_data.append(VertexData(position=v))
    new_polyhedron: poly_repr.Polyhedron = poly_repr.Polyhedron(num_vertices=len(vertices), connectivity=adj_list, aux_data=vertex_data)
    embedding2: SphereEmbedding = SphereEmbedding(new_polyhedron)
    fig = embedding2.draw(do_draw_polyhedron=False)
    plt.show()

def test_edge_triangulation_with_tetrahedron3():
    nv = 4  # number of vertices
    connectivity = [[1, 2, 3], [2, 0, 3], [0, 1, 3], [0, 2, 1]]
    primal_data = [
        VertexData(position=np.array([-0.1, -0.1, -0.1])),
        VertexData(position=np.array([1.0, 0.0, 0.0])),
        VertexData(position=np.array([0.0, 1.0, 0.0])),
        VertexData(position=np.array([0.0, 0.0, 1.0]))
    ]
    tetrahedron: poly_repr.Polyhedron = poly_repr.Polyhedron(nv, connectivity, primal_data)
    embedding: SphereEmbedding = SphereEmbedding(tetrahedron)
    fig = embedding.draw(do_draw_polyhedron=False)
    plt.show()

    # insert some edge
    new_edge = [np.array([0.5, -0.5, -0.8]), np.array([0.5, 0.8, 0.5])]
    init_vertices = []
    for i in range(0, embedding.polyhedron_repr.num_vertices):
        init_vertices.append(np.copy(embedding.polyhedron_repr.get_vertex(i).aux_data.pos))
    init_triangles = embedding._get_facet_id_lists()
    new_vertices, new_triangles = stri.insert_edge_and_triangulate(initial_vertices=init_vertices,
                                                                          initial_triangles=init_triangles,
                                                                          new_edge=new_edge)
    vertices, adj_list = stri.convert_repr_to_adj_list(new_vertices, new_triangles)
    vertex_data = []
    for v in vertices:
        vertex_data.append(VertexData(position=v))
    new_polyhedron: poly_repr.Polyhedron = poly_repr.Polyhedron(num_vertices=len(vertices), connectivity=adj_list, aux_data=vertex_data)
    embedding2: SphereEmbedding = SphereEmbedding(new_polyhedron)
    fig = embedding2.draw(do_draw_polyhedron=False)
    plt.show()

def test_edge_triangulation_with_tetrahedron4():
    nv = 4  # number of vertices
    connectivity = [[1, 2, 3], [2, 0, 3], [0, 1, 3], [0, 2, 1]]
    primal_data = [
        VertexData(position=np.array([-0.1, -0.1, -0.1])),
        VertexData(position=np.array([1.0, 0.0, 0.0])),
        VertexData(position=np.array([0.0, 1.0, 0.0])),
        VertexData(position=np.array([0.0, 0.0, 1.0]))
    ]
    tetrahedron: poly_repr.Polyhedron = poly_repr.Polyhedron(nv, connectivity, primal_data)
    embedding: SphereEmbedding = SphereEmbedding(tetrahedron)
    fig = embedding.draw(do_draw_polyhedron=False)
    plt.show()

    # insert some edge
    new_edge = [np.array([1.0, 0.0, 1.0]), np.array([1.0, 0.0, -1.0])]
    init_vertices = []
    for i in range(0, embedding.polyhedron_repr.num_vertices):
        init_vertices.append(np.copy(embedding.polyhedron_repr.get_vertex(i).aux_data.pos))
    init_triangles = embedding._get_facet_id_lists()
    new_vertices, new_triangles = stri.insert_edge_and_triangulate(initial_vertices=init_vertices,
                                                                          initial_triangles=init_triangles,
                                                                          new_edge=new_edge)
    vertices, adj_list = stri.convert_repr_to_adj_list(new_vertices, new_triangles)
    vertex_data = []
    for v in vertices:
        vertex_data.append(VertexData(position=v))
    new_polyhedron: poly_repr.Polyhedron = poly_repr.Polyhedron(num_vertices=len(vertices), connectivity=adj_list, aux_data=vertex_data)
    embedding2: SphereEmbedding = SphereEmbedding(new_polyhedron)
    fig = embedding2.draw(do_draw_polyhedron=False)
    plt.show()

def test_edge_triangulation_with_tetrahedron5():
    nv = 4  # number of vertices
    connectivity = [[1, 2, 3], [2, 0, 3], [0, 1, 3], [0, 2, 1]]
    primal_data = [
        VertexData(position=np.array([-0.1, -0.1, -0.1])),
        VertexData(position=np.array([1.0, 0.0, 0.0])),
        VertexData(position=np.array([0.0, 1.0, 0.0])),
        VertexData(position=np.array([0.0, 0.0, 1.0]))
    ]
    tetrahedron: poly_repr.Polyhedron = poly_repr.Polyhedron(nv, connectivity, primal_data)
    embedding: SphereEmbedding = SphereEmbedding(tetrahedron)
    fig = embedding.draw(do_draw_polyhedron=False)
    plt.show()

    # insert some edge
    new_edge = [3, np.array([1.0, 0.0, -1.0])]
    init_vertices = []
    for i in range(0, embedding.polyhedron_repr.num_vertices):
        init_vertices.append(np.copy(embedding.polyhedron_repr.get_vertex(i).aux_data.pos))
    init_triangles = embedding._get_facet_id_lists()
    new_vertices, new_triangles = stri.insert_edge_and_triangulate(initial_vertices=init_vertices,
                                                                          initial_triangles=init_triangles,
                                                                          new_edge=new_edge)
    vertices, adj_list = stri.convert_repr_to_adj_list(new_vertices, new_triangles)
    vertex_data = []
    for v in vertices:
        vertex_data.append(VertexData(position=v))
    new_polyhedron: poly_repr.Polyhedron = poly_repr.Polyhedron(num_vertices=len(vertices), connectivity=adj_list, aux_data=vertex_data)
    embedding2: SphereEmbedding = SphereEmbedding(new_polyhedron)
    fig = embedding2.draw(do_draw_polyhedron=False)
    plt.show()

def test_edge_triangulation_with_tetrahedron6():
    nv = 4  # number of vertices
    connectivity = [[1, 2, 3], [2, 0, 3], [0, 1, 3], [0, 2, 1]]
    primal_data = [
        VertexData(position=np.array([-0.1, -0.1, -0.1])),
        VertexData(position=np.array([1.0, 0.0, 0.0])),
        VertexData(position=np.array([0.0, 1.0, 0.0])),
        VertexData(position=np.array([0.0, 0.0, 1.0]))
    ]
    tetrahedron: poly_repr.Polyhedron = poly_repr.Polyhedron(nv, connectivity, primal_data)
    embedding: SphereEmbedding = SphereEmbedding(tetrahedron)
    fig = embedding.draw(do_draw_polyhedron=False)
    plt.show()

    # insert some edge
    new_edge = [np.array([1.0, 0.0, 1.0]), 1]
    init_vertices = []
    for i in range(0, embedding.polyhedron_repr.num_vertices):
        init_vertices.append(np.copy(embedding.polyhedron_repr.get_vertex(i).aux_data.pos))
    init_triangles = embedding._get_facet_id_lists()
    new_vertices, new_triangles = stri.insert_edge_and_triangulate(initial_vertices=init_vertices,
                                                                          initial_triangles=init_triangles,
                                                                          new_edge=new_edge)
    vertices, adj_list = stri.convert_repr_to_adj_list(new_vertices, new_triangles)
    vertex_data = []
    for v in vertices:
        vertex_data.append(VertexData(position=v))
    new_polyhedron: poly_repr.Polyhedron = poly_repr.Polyhedron(num_vertices=len(vertices), connectivity=adj_list, aux_data=vertex_data)
    embedding2: SphereEmbedding = SphereEmbedding(new_polyhedron)
    fig = embedding2.draw(do_draw_polyhedron=False)
    plt.show()

if __name__ == '__main__':
    test_edge_triangulation_with_awartani_henderson2()
