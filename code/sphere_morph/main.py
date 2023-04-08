# This is a sample Python script.
import numpy as np
import polyhedron_repr as poly_repr
from polyhedron_repr import VertexData
from polyhedron_repr import HendersonBadExample
from sphere_repr import SphereEmbedding
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

# mpl.use('macosx')
mpl.use('Qt5Agg')
# mpl.use('GTK3Agg')
# mpl.use('pyqt5')
import matplotlib.pyplot as plt


def test_plot_polyhedron():
    # primal tet graph construction
    nv = 4  # number of vertices
    connectivity = [[1, 3, 2], [0, 2, 3], [0, 3, 1], [0, 1, 2]]
    primal_data = [
        VertexData(position=np.array([0.0, 0.0, 0.0])),
        VertexData(position=np.array([1.0, 0.0, 0.0])),
        VertexData(position=np.array([0.0, 1.0, 0.0])),
        VertexData(position=np.array([0.0, 0.0, 1.0]))
    ]
    tetrahedron: poly_repr.Polyhedron = poly_repr.Polyhedron(nv, connectivity, primal_data)
    fig = tetrahedron.plot_surface(face_color=np.array([1, 0, 0, 0.3]), linewidth=1.0)
    tetrahedron.recenter()
    fig = tetrahedron.plot_surface(face_color=np.array([0, 1, 1, 1.0]), linewidth=1.0, handle=fig)
    plt.show()


def test_plot_polar():
    # primal tet graph construction
    nv = 4  # number of vertices
    connectivity = [[1, 3, 2], [0, 2, 3], [0, 3, 1], [0, 1, 2]]
    primal_data = [
        VertexData(position=np.array([-1.0, -1.0, -1.0])),
        VertexData(position=np.array([1.0, 0.0, 0.0])),
        VertexData(position=np.array([0.0, 1.0, 0.0])),
        VertexData(position=np.array([0.0, 0.0, 1.0]))
    ]
    tetrahedron: poly_repr.Polyhedron = poly_repr.Polyhedron(nv, connectivity, primal_data)
    tet_polar: poly_repr.Polyhedron = tetrahedron.build_polar()
    fig = tetrahedron.plot_surface(face_color=np.array([1, 0, 0, 0.3]), linewidth=1.0)
    fig = tet_polar.plot_surface(face_color=np.array([0, 1.0, 0.8, 0.5]), handle=fig)
    plt.show()


def test_plot_sphere_embedding():
    # primal tet graph construction
    nv = 4  # number of vertices
    connectivity = [[1, 3, 2], [0, 2, 3], [0, 3, 1], [0, 1, 2]]
    primal_data = [
        VertexData(position=np.array([0.0, 0.0, 0.0])),
        VertexData(position=np.array([1.0, 0.0, 0.0])),
        VertexData(position=np.array([0.0, 1.0, 0.0])),
        VertexData(position=np.array([0.0, 0.0, 1.0]))
    ]
    tetrahedron: poly_repr.Polyhedron = poly_repr.Polyhedron(nv, connectivity, primal_data)
    embedding: SphereEmbedding = SphereEmbedding(tetrahedron)
    fig = embedding.draw(do_draw_polyhedron=True)
    plt.show()


def test_plot_henderson_bad_example():
    bad_polyhedron: poly_repr.Polyhedron = HendersonBadExample(n=4, twist=np.pi / 4)
    embedding: SphereEmbedding = SphereEmbedding(bad_polyhedron)
    fig = embedding.draw(do_draw_polyhedron=False)
    plt.show()

def test_morph_sphere_good_example():
    good_polyhedron: poly_repr.Polyhedron = HendersonBadExample(n=4, twist=0.0)
    embedding: SphereEmbedding = SphereEmbedding(good_polyhedron)
    did_succeed, new_coordinates, vol_lbd = embedding._morph_into_shemisphere_with_convex_boundary(4)
    print("Volume lower bound is {0}".format(vol_lbd))

    # visualize morph
    if did_succeed:
        embedding.draw_longitude_morph(north_pole_idx=4,
                                       new_coordinates=new_coordinates,
                                       num_snapshots=50,
                                       save_file_pattern="images/snapshot_{0}.png")

    else:
        print("Did not succeed, which is unexpected!")

def test_quad_convexification():
    '''
    test_poly: poly_repr.Polyhedron = poly_repr.Polyhedron(num_vertices=4,
                                                           connectivity=[[3, 1, 2], [0, 3, 2], [1, 3, 0], [2, 1, 0]],
                                                           aux_data=[VertexData(position=np.array([1.0, 1.0, 1.0])),
                                                                     VertexData(position=np.array([0.0, 0.0, 1.0])),
                                                                     VertexData(position=np.array([1.0, -1.0, 1.0])),
                                                                     VertexData(position=np.array([0.5, 0.0, 1.0]))])
    test_embedding: SphereEmbedding = SphereEmbedding(test_poly)
    did_find, quad, tri1, tri2 = test_embedding.try_find_nonconvex_quad()
    quad = [0, 1, 2, 3]
    if test_embedding._is_quad_nonconvex(quad):
        print("expected")
    else:
        print("unexpected")
    '''

    bad_polyhedron: poly_repr.Polyhedron = HendersonBadExample(n=10, twist=np.pi/3.0)
    embedding: SphereEmbedding = SphereEmbedding(bad_polyhedron)
    did_find, quad, tri1, tri2 = embedding.try_find_nonconvex_quad()
    '''
    if did_find:
        print(quad)
        embedding.draw(other_vertices=quad)
    else:
        embedding.draw()
    plt.show()
    '''

    # try to convexify the quad
    did_succeed, new_coordinates, direction = embedding.try_quad_convexification(quad, tri1, tri2)


    fig_handle, ax = embedding.draw(other_vertices=quad)
    plt.figure(fig_handle.number)
    ax.plot3D([-direction[0], direction[0]], [-direction[1], direction[1]], [-direction[2], direction[2]], 'red')
    plt.show()


    if did_succeed:
        embedding.draw_longitude_directed_morph(quad=quad,
                                                new_coordinates=new_coordinates,
                                                direction=direction,
                                                save_file_pattern="quad_conv_imgs/snapshot_{0}.png")



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_quad_convexification()
