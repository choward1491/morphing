import polyhedron_repr as poly_repr
import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


class SphereEmbedding:
    def __init__(self, polyhedron: poly_repr.Polyhedron):
        self.polyhedron_repr = polyhedron
        self._build_unit_sphere_triangulation()  # build unit triangulation for visualization

    def draw(self, do_draw_polyhedron=False, handle=None):
        # set the figure handle
        fig_handle = None
        ax = None
        if handle is None:
            fig_handle = plt.figure()
        else:
            fig_handle = handle
        plt.figure(fig_handle.number)
        ax = fig_handle.gca(projection='3d')

        # center and normalize the polyhedron
        self.polyhedron_repr.recenter()
        poly_repr.scale(self.polyhedron_repr, 0.8 / self.polyhedron_repr.radius())

        # draw polyhedron if necessary
        if do_draw_polyhedron:
            self.polyhedron_repr.plot_surface(face_color=np.array([1.0, 0.0, 0.6, 0.3]), handle=fig_handle)

        # draw a transparent sphere
        ax.plot_trisurf(self.x, self.y, triangles=self.triangles, linewidth=0, color=np.array([0.9, 0.9, 0.9, 0.35]),
                        Z=self.z)

        # draw the edges of the embedding on the sphere
        self._draw_edges(ax)

        # draw the vertices of the embedding on the sphere
        self._draw_vertices(ax)

    def _build_unit_sphere_triangulation(self):
        (n, m) = (30, 50)

        # Meshing a unit sphere according to n, m
        theta = np.linspace(0, 2 * np.pi, num=n, endpoint=False)
        phi = np.linspace(np.pi * (-0.5 + 1. / (m + 1)), np.pi * 0.5, num=m, endpoint=False)
        theta, phi = np.meshgrid(theta, phi)
        theta, phi = theta.ravel(), phi.ravel()
        theta = np.append(theta, [0.])  # Adding the north pole...
        phi = np.append(phi, [np.pi * 0.5])
        mesh_x, mesh_y = ((np.pi * 0.5 - phi) * np.cos(theta), (np.pi * 0.5 - phi) * np.sin(theta))
        self.triangles = mtri.Triangulation(mesh_x, mesh_y).triangles
        self.x, self.y, self.z = np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)

    def _draw_edge(self, ax, dart):
        alpha_values = np.linspace(0.0, 1.0, 10)
        p_start = dart.tail.aux_data.pos / np.linalg.norm(dart.tail.aux_data.pos)
        p_end = dart.head.aux_data.pos / np.linalg.norm(dart.head.aux_data.pos)
        p_last = p_start
        for i in range(1, len(alpha_values)):
            p = p_start * (1.0 - alpha_values[i]) + p_end * alpha_values[i]
            u_p = p / np.linalg.norm(p)
            ax.plot3D(np.array([p_last[0], u_p[0]]), np.array([p_last[1], u_p[1]]), np.array([p_last[2], u_p[2]]),
                      color="black", linewidth=1.0)
            p_last = u_p

    def _draw_edges(self, ax):
        dart_set = set()
        nv = self.polyhedron_repr.num_vertices
        for i in range(0, nv):
            out_edges = self.polyhedron_repr.get_out_darts(i)
            for dart in out_edges:
                if dart in dart_set:
                    continue
                else:
                    dart_set.add(dart)
                    dart_set.add(dart.twin)
                    self._draw_edge(ax, dart)

    def _draw_vertices(self, ax):
        for v in self.polyhedron_repr.vertices:
            # compute normalized vector
            u = v.aux_data.pos / np.linalg.norm(v.aux_data.pos)

            # plot vertex
            ax.scatter(np.array([u[0]]),
                       np.array([u[1]]),
                       np.array([u[2]]), color=np.array([0.6, 0, 1.0, 1.0]), marker="o", linewidth=0.5,
                       edgecolors="black")
