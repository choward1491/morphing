import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from .vertex import *
from .dart import Dart

'''
This class is designed to represent a polyhedron using a graph representation with a rotation system. 
This class is setup so that we can then do other operations on the polyhedron, like computing its dual.
'''


class Polyhedron:
    def __init__(self, num_vertices, connectivity, aux_data=None):
        """
        vertex_list should be a numpy array of dimensions d x n, where d is the
        dimension of the space and n is the number of vertices

        connectivity should be an adjacency list representation for the graph, where
        the order of the edges is expected to correspond to a clockwise rotation
        system. For example, if the out edges for a vertex v are [e1, e2, e3, e4]
        in that order, the expected rotation system is drawn below

                                      ▲ │
                                      │ │
                                      │ │
                                      │ │
                                      │ │
                                     e2 │                  ┌▶
                                      │ │                ┌─┘ ┌─
                                      │ │              ┌─┘ ┌─┘
                                      │ │           e3─┘ ┌─┘
                                      │ │          ┌─┘ ┌─┘
                                      │ ▼        ┌─┘ ┌─┘
                                              ┌──┘ ┌─┘
                                    .─────. ──┘  ┌─┘
                                   ;       :   ┌─┘
                  ┌──────────▶     :   v   ;  ◀┘
        ──────────┘┌──e1───────
         ◀─────────┘                 `───'  ──┐
                                           ◀┐ └──┐
                                            └──┐ └e4┐
                                               └──┐ └──┐
                                                  └──┐ └─┐
                                                     └── └▶

        """

        # build the primal graph
        self._build_primal_graph(num_vertices, connectivity, aux_data)

    def get_vertex(self, vid):
        return self.vertices[vid]

    def get_dart(self, vid1, vid2):
        for d in self.out_connectivity[vid1]:
            if d.head.id == vid2:
                return d
        return None

    def get_out_darts(self, vid):
        return self.out_connectivity[vid]

    def num_v(self):
        return self.num_vertices

    def num_e(self):
        return self.num_edges

    def out_degree(self, v: Vertex):
        # primal out degree of a primal vertex v
        id = v.id
        return len(self.out_connectivity[id])

    def in_degree(self, v: Vertex):
        # primal in degree of a primal vertex v
        # currently assume the same as out degree
        return self.out_degree(v)

    def build_dual_graph(self):

        # get the faces for the dual graph
        faces_list, edge2facet_idx = self._get_faces()

        # get the number of faces
        nf = len(faces_list)

        # construct the vertices for the dual graph
        dual_data = []
        for f in faces_list:
            dual_data.append(VertexData(face=f))

        # construct the edges for the dual graph
        dual_connectivity = []
        for i in range(0, nf):
            e_list = []
            for e in faces_list[i]:
                e_rev = e.rev()
                e_list.append(edge2facet_idx[e_rev])
            e_list.reverse()
            dual_connectivity.append(e_list)

        # construct the actual dual graph
        dual_graph: Polyhedron = Polyhedron(nf, dual_connectivity, dual_data)

        # connect dual edges and primal edges
        for i in range(0, nf):
            # faces_list[i].reverse()
            idx = 0
            for j in dual_connectivity[i]:
                e = faces_list[i][idx]
                idx = idx + 1

                # construct dual edge
                e_ij = dual_graph.get_dart(i, j)

                # connect primal edge with corresponding dual edge
                e_ij.set_dual_edge(e)
                e.set_dual_edge(e_ij)

        return dual_graph

    def build_polar(self):

        # get the faces for the dual graph
        faces_list, edge2facet_idx = self._get_faces()

        # get the number of faces
        nf = len(faces_list)

        # construct the vertices for the dual graph
        dual_data = []
        for f in faces_list:
            # compute the position of the dual point
            n = self._compute_polar_normal(f)

            # add the dual data
            dual_data.append(VertexData(face=f, position=n))

        # construct the edges for the dual graph
        dual_connectivity = []
        for i in range(0, nf):
            e_list = []
            for e in faces_list[i]:
                e_rev = e.rev()
                e_list.append(edge2facet_idx[e_rev])
            e_list.reverse()
            dual_connectivity.append(e_list)

        # construct the actual dual graph
        dual_graph: Polyhedron = Polyhedron(nf, dual_connectivity, dual_data)

        # connect dual edges and primal edges
        for i in range(0, nf):
            # faces_list[i].reverse()
            idx = 0
            for j in dual_connectivity[i]:
                e = faces_list[i][idx]
                idx = idx + 1

                # construct dual edge
                e_ij = dual_graph.get_dart(i, j)

                # connect primal edge with corresponding dual edge
                e_ij.set_dual_edge(e)
                e.set_dual_edge(e_ij)

        return dual_graph

    def get_points_coords(self):
        P = np.zeros((self.num_vertices, 3))
        for i in range(0, self.num_vertices):
            vi = self.vertices[i]
            P[i, :] = vi.aux_data.pos
        return P

    def recenter(self, new_center=np.zeros((3,))):
        delta = new_center - self.center
        for v in self.vertices:
            v.aux_data.pos += delta
        self.center = new_center

    def radius(self):
        r = 0
        for v in self.vertices:
            r = np.max([r, np.linalg.norm(v.aux_data.pos)])
        return r

    def plot_edges(self, color, handle=None):
        # set the figure handle
        fig_handle = None
        if handle is None:
            fig_handle = plt.figure()
        else:
            fig_handle = handle
        plt.figure(fig_handle.number)
        ax = plt.axes(projection='3d')

        # size of polyhedron graph
        ne = self.num_e()  # number of edges
        nv = self.num_v()  # number of vertices

        # plot the edges first
        for i in range(0, nv):
            out_edges = self.get_out_darts(i)
            for e in out_edges:
                # plot edge using coordinates of vertices
                p1 = e.tail.aux_data.pos
                p2 = e.head.aux_data.pos
                ax.plot3D(np.array([p1[0], p2[0]]),
                          np.array([p1[1], p2[1]]),
                          np.array([p1[2], p2[2]]), color=color, linewidth=2)

        # plot the vertices
        for i in range(0, nv):
            vi = self.get_vertex(i).aux_data

            # plot vertex
            ax.scatter(np.array([vi.pos[0]]),
                       np.array([vi.pos[1]]),
                       np.array([vi.pos[2]]), c=color)

        # return the figure handle
        return fig_handle

    def plot_surface(self, face_color, linewidth=1, handle=None):
        # set the figure handle
        fig_handle = None
        ax = None
        if handle is None:
            fig_handle = plt.figure()
        else:
            fig_handle = handle
        plt.figure(fig_handle.number)
        ax = fig_handle.gca(projection='3d')

        # get the faces of the polyhedron
        # faces has type [[Dart]]
        faces, edge2facet_map = self._get_faces()

        # set the vertex coordinates
        coords = []
        for v in self.vertices:
            coords.append([v.aux_data.pos[0], v.aux_data.pos[1], v.aux_data.pos[2]])
        vertex_coords = np.array(coords)

        # set the triangular faces
        triangles = []
        for face in faces:
            d0: Dart = face[0]
            v_rel = d0.tail
            for i in range(1, len(face) - 1):
                di: Dart = face[i]
                triangles.append([v_rel.id, di.tail.id, di.head.id])

        # plot the triangulation
        ax.plot_trisurf(vertex_coords[:, 0], vertex_coords[:, 1], triangles=triangles, linewidth=0, color=face_color,
                        Z=vertex_coords[:, 2])

        # size of polyhedron graph
        ne = self.num_e()  # number of edges
        nv = self.num_v()  # number of vertices

        # plot the edges first
        for i in range(0, nv):
            out_edges = self.get_out_darts(i)
            for e in out_edges:
                # plot edge using coordinates of vertices
                p1 = e.tail.aux_data.pos
                p2 = e.head.aux_data.pos
                ax.plot3D(np.array([p1[0], p2[0]]),
                          np.array([p1[1], p2[1]]),
                          np.array([p1[2], p2[2]]), color="black", linewidth=linewidth)

        # plot the vertices
        for i in range(0, nv):
            vi = self.get_vertex(i).aux_data

            # plot vertex
            ax.scatter(np.array([vi.pos[0]]),
                       np.array([vi.pos[1]]),
                       np.array([vi.pos[2]]), c="orange", marker="o", linewidth=0.5, edgecolors="black")

        # plot the edges for the graph on top of the triangulation
        return fig_handle

    """
    add private helper methods
    """

    def _build_primal_graph(self, num_vertices, connectivity, aux_data):
        # some parameters that will be good to store
        n = num_vertices
        self.num_vertices = n
        self.num_edges = 0
        self.center = np.zeros((3,))

        # construct vertices
        self.vertices = [None] * n
        for i in range(0, n):
            if aux_data is None:
                self.vertices[i] = Vertex(i)
            else:
                self.vertices[i] = Vertex(i, aux_data[i])
                self.center += aux_data[i].pos
        self.center /= n

        # construct list of edges in adjacency list format, where the
        # edges associated with vertex i are those where vertex i is the
        # tail of the edge
        self.out_connectivity = []
        for i in range(0, n):
            vi = self.vertices[i]
            edge_list = []
            for j in connectivity[i]:
                vj = self.vertices[j]
                self.num_edges += 1
                edge_list.append(Dart(tail=vi, head=vj))
            self.out_connectivity.append(edge_list)

        # setup the reverse of each edge
        for i in range(0, n):
            for e in self.out_connectivity[i]:
                (revt, revh) = e.rev_idx()
                for eprime in self.out_connectivity[revh]:
                    if eprime.head.id == revt:
                        e.set_twin(eprime)
                        break

    def _compute_polar_normal(self, edge_list):

        # grab distinct vertices
        v_set = set()
        for e in edge_list:
            v_set.add(e.tail)
            v_set.add(e.head)

        # compute center of the vertices
        center = np.zeros((3,))
        for v in v_set:
            center += v.aux_data.pos / len(v_set)

        # compute the normal
        e0 = edge_list[0]
        v1 = e0.tail.aux_data.pos - center
        v2 = e0.head.aux_data.pos - center
        n = np.cross(v1, v2)

        # compute normalization value
        normalize_val = np.dot(n, e0.tail.aux_data.pos)
        n = n / normalize_val
        return n

    def _rot_idx(self, e: Dart):
        # index of edge e in rotation system about its tail
        idx = 0
        for eprime in self.out_connectivity[e.tail.id]:
            if eprime.head.id == e.head.id:
                return idx
            else:
                idx += 1

    def _next(self, e: Dart):
        # get next dart in rotation system
        odeg = self.out_degree(e.tail)
        idx = self._rot_idx(e)
        return self.out_connectivity[e.tail.id][(idx + 1) % odeg]

    def _prev(self, e: Dart):
        # get next dart in rotation system
        odeg = self.out_degree(e.tail)
        idx = self._rot_idx(e)
        return self.out_connectivity[e.tail.id][(idx + (odeg - 1)) % odeg]

    def _get_faces(self):
        unvisited_edges = set()

        # init set of edges
        for i in range(0, self.num_vertices):
            for edge in self.out_connectivity[i]:
                unvisited_edges.add(edge)

        # construct the faces
        fidx = 0
        edge2facet_idx = dict()
        faces_list = []
        while len(unvisited_edges) > 0:
            # init a new face
            face = []

            # pop off an arbitrary edge
            e = unvisited_edges.pop()
            face.append(e)
            edge2facet_idx[e] = fidx

            # loop over cycle representing face
            # based on the rotation system
            et = self._next(e.rev())
            while et != e:
                face.append(et)
                edge2facet_idx[et] = fidx
                unvisited_edges.remove(et)
                et = self._next(et.rev());

            # add the face to the list
            faces_list.append(face)
            fidx = fidx + 1

        # return the final list of faces
        return faces_list, edge2facet_idx


def scale(P: Polyhedron, s):
    for v in P.vertices:
        v.aux_data.pos *= s
    P.center *= s


def translate(P: Polyhedron, t):
    for v in P.vertices:
        v.aux_data.pos += t
    P.center += t


def rotate(P: Polyhedron, R):
    for v in P.vertices:
        v.aux_data.pos = np.matmul(R, v.aux_data.pos)
    P.center = np.matmul(R, P.center)
