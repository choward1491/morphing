# This is a sample Python script.
import numpy as np
import scipy as spy
from scipy.stats import ortho_group
import cvxpy as cp
import polyhedron_repr as poly_repr
from polyhedron_repr import VertexData
from polyhedron_repr import HendersonBadExample
from polyhedron_repr import GeneralizedTwistedPolyhedron
from sphere_repr import SphereEmbedding
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

mpl.use('macosx')
#mpl.use('Qt5Agg')
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


def draw_example_with_multiple_vertices_with_large_area_star():
    v1 = np.array([0, -1.0, -0.1])
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.array([0, -0.1, -1.0])
    v2 = v2 / np.linalg.norm(v2)
    v3 = np.array([-0.5, -0.5, -0.5])
    v3 = v3 / np.linalg.norm(v3)
    test_poly: poly_repr.Polyhedron = poly_repr.Polyhedron(num_vertices=7,
                                                           connectivity=[[1, 4, 3, 2],
                                                                         [4, 0, 2, 5],
                                                                         [1, 0, 3, 6, 5],
                                                                         [2, 0, 4, 6],
                                                                         [3, 0, 1, 5, 6],
                                                                         [2, 6, 4, 1],
                                                                         [2, 3, 4, 5]],
                                                           aux_data=[VertexData(position=np.array([0.0, 0.0, 1.0])),
                                                                     VertexData(position=np.array([0.0, 1.0, 0.0])),
                                                                     VertexData(position=np.array([1.0, 0.0, 0.0])),
                                                                     VertexData(position=v1),
                                                                     VertexData(position=np.array([-1.0, 0, 0])),
                                                                     VertexData(position=v2),
                                                                     VertexData(position=v3)
                                                                     ])
    test_embedding: SphereEmbedding = SphereEmbedding(test_poly)
    test_embedding.draw()
    plt.show()


def test_quad_convexification():
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


def visualize_quad_convexification_example():
    # Construct a bad twisted example from Henderson's paper
    bad_polyhedron: poly_repr.Polyhedron = HendersonBadExample(n=10, twist=np.pi / 3.0)
    embedding: SphereEmbedding = SphereEmbedding(bad_polyhedron)

    # Identify a non-convex quad
    did_find, quad, tri1, tri2 = embedding.try_find_nonconvex_quad()

    # Plot the spherical embedding so that the quad is notably visible.
    if did_find:
        print(quad)
        embedding.draw(other_vertices=quad)
        plt.show()
    else:
        print("No non-convex quad in the embedding")
        return

    # Try to convexify the quad
    did_succeed, new_coordinates, direction = embedding.try_quad_convexification(quad, tri1, tri2)
    if not did_succeed:
        print("Did not succeed in convexifying the quad in question")
        return

    # Plot line along with embedding to show what direction vertices will move along
    fig_handle, ax = embedding.draw(other_vertices=quad)
    plt.figure(fig_handle.number)
    ax.plot3D([-direction[0], direction[0]], [-direction[1], direction[1]], [-direction[2], direction[2]], 'red')
    plt.show()

    # Plot the longitudinal morph along the chosen direction.
    print("Show longitudinal morph")
    embedding.draw_longitude_directed_morph(quad=quad,
                                            new_coordinates=new_coordinates,
                                            direction=direction,
                                            save_file_pattern="quad_conv_imgs/snapshot_{0}.png")


def point_in_kernel(point, quad_indices, quad_arcs, embedding: SphereEmbedding):
    pass


def try_spherical_grid_of_points(quad_indices, embedding: SphereEmbedding):
    pass


def kernel_vshaped_quad():
    v0 = np.array([-1.0, 0.0, 0.05])
    v1 = np.array([-1.0, 1.0, 0.2])
    v2 = np.array([1.0, 1.0, 0.0])
    v3 = np.array([1.0, -1.0, 0.0])
    v0 *= 1.0 / np.linalg.norm(v0)
    v1 *= 1.0 / np.linalg.norm(v1)
    v2 *= 1.0 / np.linalg.norm(v2)
    v3 *= 1.0 / np.linalg.norm(v3)
    test_poly: poly_repr.Polyhedron = poly_repr.Polyhedron(num_vertices=4,
                                                           connectivity=[[1, 2, 3], [2, 0, 3], [0, 1], [1, 0]],
                                                           aux_data=[VertexData(position=v0),
                                                                     VertexData(position=v1),
                                                                     VertexData(position=v2),
                                                                     VertexData(position=v3)])
    test_embedding: SphereEmbedding = SphereEmbedding(test_poly)
    did_find, quad, tri1, tri2 = test_embedding.try_find_nonconvex_quad()
    quad = [0, 1, 2, 3]
    if not did_find:
        print("Did not find a non-convex quad, uh oh")
        return
    else:
        print("Found a non-convex quad!")


def semidefinite_embedding():
    m = 10
    # Construct a bad twisted example from Henderson's paper
    bad_polyhedron: poly_repr.Polyhedron = HendersonBadExample(n=m, twist=np.pi / 3.0)
    embedding: SphereEmbedding = SphereEmbedding(bad_polyhedron)
    n: int = bad_polyhedron.num_vertices

    # extract the adjacency matrix of the underlying graph
    A = np.zeros((n, n))
    for i in range(0, n):
        v_list = bad_polyhedron.out_connectivity[i]
        for j in v_list:
            A[i, j.head.id] = 1.0

    # construct the semidefinite program that will attempt to make an embedding
    # Define and solve the CVXPY problem.
    # Create a symmetric matrix variable.
    X = cp.Variable((n, n), PSD=True)

    # The operator >> denotes matrix inequality.
    #constraints = [X >> 0]

    # add constraint so barycenter of the points is the origin
    J = np.ones((n, n))
    constraints = [
        cp.trace(J @ X) == 0.0
    ]

    # add constraints so the points are restricted to the surface of the sphere
    for i in range(0, n):
        C = np.zeros((n, n))
        C[i, i] = 1.0
        constraints += [cp.trace(C @ X) == 1.0]

    # solve the problem
    prob = cp.Problem(cp.Minimize(cp.trace(A @ X)),
                      constraints)
    prob.solve()
    print(prob.status)

    # compute the cholesky factorization
    (L, D, P) = spy.linalg.ldl(X.value)
    (U, S, Vh) = np.linalg.svd(X.value)
    print(U @ np.sqrt(S))
    print(f"S = \n{S}")
    print(L)
    print(P)
    print(f"D = \n{D}")
    # = np.linalg.cholesky(X.value)

    # check if the rank is at most 3
    rank = np.linalg.matrix_rank(L)
    print(rank)
    R = ortho_group.rvs(dim=n)
    Lp = R @ L


    for i in range(0, 3):
        D[i, i] = np.sqrt(D[i, i])
    for i in range(3, n):
        D[i, i] = 0.0
    Lp = L @ D
    print(f"Matrix rank of Lp is {np.linalg.matrix_rank(Lp)}")


    coordinates = Lp[:, :3].T
    coordinates = U[:, :3].T
    print(coordinates)

    # update the coordinates
    bad_polyhedron2: poly_repr.Polyhedron = HendersonBadExample(n=m, twist=np.pi / 3.0)
    for i in range(0, n):
        bad_polyhedron2.get_vertex(i).aux_data.pos = coordinates[:, i]
    embedding2: SphereEmbedding = SphereEmbedding(bad_polyhedron2)

    # plot the result
    embedding2.draw()
    plt.show()

def try_untwist_embedding():

    # define the bad twisted embedding
    bad_polyhedron: poly_repr.Polyhedron = GeneralizedTwistedPolyhedron(num_layers=3, num_vertices_per_layer=3, amount_relative_rotate=np.pi/12)

    # form spherical embedding based on it
    embedding: SphereEmbedding = SphereEmbedding(polyhedron=bad_polyhedron)
    num_vertices = embedding.polyhedron_repr.num_vertices

    # plot the initial embedding
    embedding.draw()
    plt.show()

    # state the two directions we will use to untwist
    x_dir = np.array([1.0, 0.0, 0.0])
    y_dir = np.array([0.0, 1.0, 0.0])

    # try to untwist in x dir
    succeed, output_coordinates, direction = embedding._unidirectional_untwist_version1(direction = x_dir, phi_max = np.pi/4)
    if not succeed:
        print("Failed on x-dir twist!!")
        return

    for i in range(0, num_vertices):
        embedding.polyhedron_repr.get_vertex(i).aux_data.pos = output_coordinates[i][:]

    #try to untwist in y dir
    succeed, output_coordinates, direction = embedding._unidirectional_untwist_version1(direction=y_dir,
                                                                                        phi_max=np.pi / 4)
    if not succeed:
        print("Failed on y-dir twist!!")
        return

    for i in range(0, num_vertices):
        embedding.polyhedron_repr.get_vertex(i).aux_data.pos = output_coordinates[i][:]

    # plot the final embedding
    embedding.draw()
    plt.show()

def try_convexify_embedding_v2():
    # define the bad twisted embedding
    bad_polyhedron: poly_repr.Polyhedron = GeneralizedTwistedPolyhedron(num_layers=4, num_vertices_per_layer=4,
                                                                        amount_relative_rotate=np.pi / 12)
    # define the bad twisted embedding
    bad_polyhedron0: poly_repr.Polyhedron = GeneralizedTwistedPolyhedron(num_layers=2, num_vertices_per_layer=8,
                                                                        amount_relative_rotate=np.pi / 20,
                                                                        amount_subtract_angle_odd=0.0 * (np.pi / 12.0),
                                                                        amount_add_vertical_odd=0.0)

    # form spherical embedding based on it
    embedding: SphereEmbedding = SphereEmbedding(polyhedron=bad_polyhedron)
    num_vertices = embedding.polyhedron_repr.num_vertices

    # Identify a non-convex quad
    did_find, quad, tri1, tri2 = embedding.try_find_nonconvex_quad()

    # Plot the spherical embedding so that the quad is notably visible.
    if did_find:
        print(quad)
        embedding.draw(other_vertices=quad, sphere_color=np.array([1.0, 0.8, 0.8, 1.0]))
        plt.show()
    else:
        print("No non-convex quad in the embedding")
        return

    # plot the initial embedding
    #embedding.draw()

    # try to convexify chosen quad
    succeed, output_coordinates, direction = embedding._convexify_quad2(quad, tri1, tri2)
    if not succeed:
        print("Failed to convexify!!")
        return

    for i in range(0, num_vertices):
        embedding.polyhedron_repr.get_vertex(i).aux_data.pos = output_coordinates[i][:]

    # plot the final embedding
    embedding.draw(other_vertices=quad, sphere_color=np.array([1.0, 0.8, 0.8, 1.0]))
    plt.show()

def try_convexify_embedding_v3():
    # define the bad twisted embedding
    bad_polyhedron: poly_repr.Polyhedron = GeneralizedTwistedPolyhedron(num_layers=2, num_vertices_per_layer=8,
                                                                        amount_relative_rotate=np.pi / 2.5,
                                                                        amount_subtract_angle_odd=1.0*(np.pi / 12.0),
                                                                        amount_add_vertical_odd=0.01)

    test_poly: poly_repr.Polyhedron = poly_repr.Polyhedron(num_vertices=7,
                                                           connectivity=[[3, 1, 2, 4], [0, 3, 6, 5, 2], [0, 1, 5, 4], [1, 0, 4, 6], [3, 0, 2, 5, 6], [2, 1, 6, 4], [5, 1, 3, 4]],
                                                           aux_data=[VertexData(position=np.array([0.9, 0, 1.0])),
                                                                     VertexData(position=np.array([0.0, 0.0, 1.0])),
                                                                     VertexData(position=np.array([1.0, -1.0, 0.0])),
                                                                     VertexData(position=np.array([1.0, 1.0, 0.0])),
                                                                     VertexData(position=np.array([1e-3, 0, -1.0])),
                                                                     VertexData(position=np.array([-1.0, -1.0, 0.0])),
                                                                     VertexData(position=np.array([-1.0, 1.0, 0.0]))])

    # form spherical embedding based on it
    embedding: SphereEmbedding = SphereEmbedding(polyhedron=bad_polyhedron)
    num_vertices = embedding.polyhedron_repr.num_vertices

    # Identify a non-convex quad
    did_find, quad, tri1, tri2 = embedding.try_find_nonconvex_quad()
    #quad = [0, 3, 1, 2]
    #tri1 = [2, 0, 1]
    #tri2 = [0, 3, 1]

    # Plot the spherical embedding so that the quad is notably visible.
    if did_find:
        print(quad)
        embedding.draw(other_vertices=quad, sphere_color=np.array([1.0, 0.8, 0.8, 1.0]))
        plt.show()
    else:
        print("No non-convex quad in the embedding")
        return

    # plot the initial embedding
    #embedding.draw()

    # try to convexify chosen quad
    succeed, output_coordinates, direction = embedding._convexify_quad3(quad, tri1, tri2)
    if not succeed:
        print("Failed to convexify!!")

    if succeed:
        print("Succeeded!")
        for i in range(0, num_vertices):
            embedding.polyhedron_repr.get_vertex(i).aux_data.pos = output_coordinates[i][:]

    # plot the final embedding
    embedding.draw(other_vertices=quad, sphere_color=np.array([1.0, 0.8, 0.8, 1.0]))
    plt.show()

def compare_convexification_approaches():
    # define the bad twisted embedding
    bad_polyhedron: poly_repr.Polyhedron = GeneralizedTwistedPolyhedron(num_layers=2, num_vertices_per_layer=8,
                                                                        amount_relative_rotate=0.7*np.pi,
                                                                        amount_subtract_angle_odd=0.0 * (np.pi / 12.0),
                                                                        amount_add_vertical_odd=0.0*0.01)

    bad_polyhedron2: poly_repr.Polyhedron = GeneralizedTwistedPolyhedron(num_layers=8, num_vertices_per_layer=8,
                                                                        amount_relative_rotate=np.pi / 2.2,
                                                                        amount_subtract_angle_odd=1.0 * (np.pi / 12.0),
                                                                        amount_add_vertical_odd=0.01)

    # form spherical embedding based on it
    embedding: SphereEmbedding = SphereEmbedding(polyhedron=bad_polyhedron)
    num_vertices = embedding.polyhedron_repr.num_vertices

    # Identify a non-convex quad
    did_find, quad, tri1, tri2 = embedding.try_find_nonconvex_quad()

    # Plot the spherical embedding so that the quad is notably visible.
    if did_find:
        print(quad)
    else:
        print("No non-convex quad in the embedding")
        return
    reflex_idx, dir = embedding._get_reflex_vertex_and_direction(quad)

    succeed, output_coordinates, direction, R3 = embedding._convexify_quad3(quad, tri1, tri2)
    #succeed, output_coordinates, direction, R3 = embedding._convexify_quad2_threefixed(quad, tri1, tri2)
    if not succeed:
        print("Convexify-3 Failed to convexify!!\n")
    else:
        print("Convexify-3 succeeded!\n")

    succeed, output_coordinates, direction, R2 = embedding._convexify_quad2(quad, tri1, tri2)
    if not succeed:
        print("Convexify-2 Failed to convexify!!")
    else:
        print("Convexify-2 succeeded!")

    poly_repr.rotate(embedding.polyhedron_repr, R2)
    embedding.draw(other_vertices=quad, sphere_color=np.array([1.0, 0.8, 0.8, 1.0]))
    plt.show()

    '''
    reflex_vert = quad[reflex_idx]
    coord1 = embedding.polyhedron_repr.get_vertex(reflex_vert).aux_data.pos
    coord1[2] = coord1[2] - 1.0
    coord1 = coord1 / np.linalg.norm(coord1)
    embedding.polyhedron_repr.get_vertex(reflex_vert).aux_data.pos = coord1

    bad_vert = quad[reflex_idx] + 1
    coord2 = embedding.polyhedron_repr.get_vertex(bad_vert).aux_data.pos
    coord2[2] = coord2[2] - 1.0
    coord2 = coord2 / np.linalg.norm(coord2)
    embedding.polyhedron_repr.get_vertex(bad_vert).aux_data.pos = coord2

    left_vert = quad[(reflex_idx + 3) % 4]
    coord3 = embedding.polyhedron_repr.get_vertex(left_vert).aux_data.pos
    coord3[2] = coord3[2] + 0.6
    coord3 = coord3 / np.linalg.norm(coord3)
    embedding.polyhedron_repr.get_vertex(left_vert).aux_data.pos = coord3
    '''

    if succeed:
        for i in range(0, num_vertices):
            embedding.polyhedron_repr.get_vertex(i).aux_data.pos = output_coordinates[i][:]
        poly_repr.rotate(embedding.polyhedron_repr, R2)

    if not embedding._is_quad_nonconvex(quad):
        print("New vertex locations are convex!")

    embedding.draw(other_vertices=quad, sphere_color=np.array([1.0, 0.8, 0.8, 1.0]))
    plt.show()


if __name__ == '__main__':
    compare_convexification_approaches()
