# This is a sample Python script.
import numpy as np
import polyhedron_repr as poly_repr
from polyhedron_repr import VertexData
from mpl_toolkits import mplot3d
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('macosx')

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
    fig = tetrahedron.plot_surface(face_color=np.array([1, 0, 0, 1.0]), linewidth=1.0)
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_plot_polar()
