import polyhedron_repr as poly_repr
import pulp
from pulp import *
import time
import numpy as np
from scipy.linalg import null_space
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


class SphereEmbedding:
    def __init__(self, polyhedron: poly_repr.Polyhedron):
        self.polyhedron_repr = polyhedron
        self._normalize_internal_polygon()
        self._build_unit_sphere_triangulation()  # build unit triangulation for visualization

    # TODO(choward): Should probably separate any drawing routines out into a class so we can create different drawing implementations of the sphere, depending on the visualization library in question.
    def draw(self, do_draw_polyhedron=False, handle=None, az=None, el=None, other_vertices=None,
             sphere_color=np.array([0.9, 0.9, 0.9, 0.5])):
        """
        This method is for drawing a spherical embedding of a graph onto a sphere with an optimal flag for drawing the underlying polyhedron,
        :param do_draw_polyhedron: Boolean flag to decide whether a user wants to draw the underlying star-shaped polyhedron of the spherical embedding.
        :param handle: Figure handle for reuse purposes.
        :param az: Azimuth angle to position camera
        :param el: Elevation angle to position camera
        :param other_vertices: List of vertices that will be marked by another color when plotted, usually to emphasize some particularly localized aspect of the embedding.
        :return: (Figure Handle, Plot Axes)
        """
        # set the figure handle
        fig_handle = None
        ax = None
        if handle is None:
            fig_handle = plt.figure()
        else:
            fig_handle = handle
        plt.figure(fig_handle.number)
        # ax = fig_handle.gca(projection='3d')
        ax = plt.axes(projection='3d', computed_zorder=False)

        # set 3d camera
        if az is not None:
            ax.azim = az

        if el is not None:
            ax.elev = el

        # center and normalize the polyhedron
        # self.polyhedron_repr.recenter()
        poly_repr.scale(self.polyhedron_repr, 1.0 / self.polyhedron_repr.radius())

        # draw polyhedron if necessary
        if do_draw_polyhedron:
            self.polyhedron_repr.plot_surface(face_color=np.array([1.0, 0.0, 0.6, 0.5]), handle=fig_handle)

        # draw a transparent sphere
        ax.plot_trisurf(self.x, self.y, triangles=self.triangles, linewidth=0, color=sphere_color,
                        Z=self.z)

        # draw the edges of the embedding on the sphere
        self._draw_edges(ax)

        # draw the vertices of the embedding on the sphere
        self._draw_vertices(ax, other_vertices)

        return fig_handle, ax

    def draw_longitude_morph(self, north_pole_idx: int, new_coordinates, num_snapshots=50, save_file_pattern=None):
        """
        This method produces an animation for a longitudinal morph with snapshots that are saved to disk as images.
        :param north_pole_idx: Index for vertex that will be viewed as the north pole
        :param new_coordinates: New coordinates we will morph the embedding to, under the assumption they can be obtained by a longitudinal morph.
        :param num_snapshots: Number of animation frames we want to save to disk.
        :param save_file_pattern: Pattern for image filenames that we will save, if specified.
        :return: None
        """

        # draw figure first and let user tweak the orientation and size
        fig_handle = self.draw()
        plt.show()

        input("Set up the figure to the orientation and size you want and then hit enter:")

        # set the figure handle
        plt.figure(fig_handle.number)
        # ax = fig_handle.gca(projection='3d')
        ax = plt.axes(projection='3d', computed_zorder=False)
        az = ax.azim
        el = ax.elev

        # create interpolation values
        alpha_values = np.linspace(0.0, 1.0, num_snapshots)

        # save the original vertex values
        num_vertices = self.polyhedron_repr.num_vertices
        old_coordinates = []
        for i in range(0, num_vertices):
            old_coordinates.append(self.polyhedron_repr.get_vertex(i).aux_data.pos)

        # interpolate and optionally save snapshots using file pattern
        counter = 1
        for alpha in alpha_values:

            # interpolate between coordinate values
            for i in range(0, num_vertices):
                if i != north_pole_idx:
                    old_pos = old_coordinates[i]
                    new_pos = new_coordinates[i]
                    phi_old = self._phi_position(old_pos)
                    phi_new = self._phi_position(new_pos)
                    phi_alpha = np.arctan2(alpha * np.tan(phi_new) + (1.0 - alpha) * np.tan(phi_old), 1.0)
                    u = old_pos[:2]
                    u = np.cos(phi_alpha) * u / np.linalg.norm(u)
                    tmp_pos = np.array([u[0], u[1], np.sin(phi_alpha)])
                    self.polyhedron_repr.get_vertex(i).aux_data.pos = tmp_pos

            # plot the result
            plt.clf()
            fig, ax = self.draw(az=az, el=el)

            # save snapshot, if necessary
            if save_file_pattern is not None:
                fig.savefig(save_file_pattern.format(counter), dpi=300)
            else:
                time.sleep(0.1)

            # update counter
            print("Finished img {0}".format(counter))
            counter += 1

        # set coordinates back to the old ones
        for i in range(0, num_vertices):
            self.polyhedron_repr.get_vertex(i).aux_data.pos = old_coordinates[i]

    def draw_longitude_directed_morph(self, quad, new_coordinates, direction, num_snapshots=50, save_file_pattern=None):
        """
        This method produces an animation for a longitudinal morph with snapshots that are saved to disk as images.
        :param quad: List of 4 vertex IDs corresponding to a quad we want to visualize being morphed
        :param new_coordinates: New coordinates we will morph the embedding to, under the assumption they can be obtained by a longitudinal morph.
        :param direction: Unit direction we will view as the north pole unit normal. We will morph along this direction to the new coordinates.
        :param num_snapshots: Number of animation frames we want to save to disk.
        :param save_file_pattern: Pattern for image filenames that we will save, if specified.
        :return: None
        """

        # draw figure first and let user tweak the orientation and size
        fig_handle, ax = self.draw(other_vertices=quad)
        plt.show()

        input("Set up the figure to the orientation and size you want and then hit enter:")

        # set the figure handle
        plt.figure(fig_handle.number)
        # ax = fig_handle.gca(projection='3d')
        ax = plt.axes(projection='3d', computed_zorder=False)
        az = ax.azim
        az = -163
        el = ax.elev
        el = 50
        print(f"az = {az}, el = {el}")

        # create interpolation values
        alpha_values = np.linspace(0.0, 1.0, num_snapshots)

        # set rotation matrix to simplify work
        direction.resize((1, 3))
        U = np.zeros((3, 3))
        Uother = null_space(direction)  # complete a basis using direction as one of the (unit) basis vectors
        U[:2, :] = Uother.transpose()
        U[2, :] = direction  # make the unit direction vector correspond to the new z coordinate
        Ut = U.transpose()
        direction.resize((3,))

        # save the original vertex values
        num_vertices = self.polyhedron_repr.num_vertices
        old_coordinates = []
        for i in range(0, num_vertices):
            old_coordinates.append(self.polyhedron_repr.get_vertex(i).aux_data.pos)

        # interpolate and optionally save snapshots using file pattern
        counter = 1
        for alpha in alpha_values:

            # interpolate between coordinate values
            for i in range(0, num_vertices):
                old_pos = np.matmul(U, old_coordinates[i])  # get old position in same frame as new position
                new_pos = np.matmul(U, new_coordinates[i])
                phi_old = self._phi_position(old_pos)
                phi_new = self._phi_position(new_pos)
                phi_alpha = np.arctan2(alpha * np.tan(phi_new) + (1.0 - alpha) * np.tan(phi_old), 1.0)
                u = old_pos[:2]
                u = np.cos(phi_alpha) * u / np.linalg.norm(u)
                tmp_pos = np.array([u[0], u[1], np.sin(phi_alpha)])
                self.polyhedron_repr.get_vertex(i).aux_data.pos = np.matmul(Ut, tmp_pos)  # tranform into old frame
                # self.polyhedron_repr.get_vertex(i).aux_data.pos = tmp_pos  # tranform into old frame

            # plot the result
            plt.clf()
            fig, ax = self.draw(az=az, el=el, other_vertices=quad)
            ax.plot3D([-direction[0], direction[0]], [-direction[1], direction[1]], [-direction[2], direction[2]],
                      'red')

            # save snapshot, if necessary
            if save_file_pattern is not None:
                fig.savefig(save_file_pattern.format(counter), dpi=300)
            else:
                time.sleep(0.1)

            # update counter
            print("Finished img {0}".format(counter))
            counter += 1

        # set coordinates back to the old ones
        for i in range(0, num_vertices):
            self.polyhedron_repr.get_vertex(i).aux_data.pos = old_coordinates[i]

    def _phi_position(self, pos):
        return np.arctan2(pos[2], np.linalg.norm(pos[:2]))

    def try_quad_convexification(self, quad, tri1, tri2):
        num_vertices = self.polyhedron_repr.num_vertices
        if len(quad) != 4 or len(tri1) != 3 or len(tri2) != 3:
            raise AssertionError()
        else:

            # attempt to convexify the quad in some direction that makes sense
            return self._convexify_quad(quad, tri1, tri2)

    def _get_reflex_vertex_and_direction(self, quad):
        # assume the vertices are in some order, be it counter clockwise or clockwise
        # *current implementation is assuming triangles are based on shortest path geodesics*

        # identify the reflex vertex
        reflex_idx = -1
        direction = np.array([1, 0, 0])
        signs = np.array([0, 0, 0, 0])
        num_pos_signs = 0
        num_neg_signs = 0
        for i in range(0, 4):
            v_i = self.polyhedron_repr.get_vertex(quad[i]).aux_data.pos
            v_ip1 = self.polyhedron_repr.get_vertex(quad[(i + 1) % 4]).aux_data.pos
            v_im1 = self.polyhedron_repr.get_vertex(quad[(i + 3) % 4]).aux_data.pos
            d1 = v_ip1 - v_i
            d1 = d1 / np.linalg.norm(d1)
            d2 = v_i - v_im1
            d2 = d2 / np.linalg.norm(d2)

            # compute gross product and dot it with direction to center of sphere
            # to get the sign of everything
            signs[i] = np.sign(np.dot(-v_i, np.cross(d2, d1)))
            num_pos_signs += (signs[i] > 0)
            num_neg_signs += (signs[i] < 0)

        # check that either we have 3 positive or 3 negative signs
        if num_pos_signs == 3 or num_neg_signs == 3:
            if num_pos_signs == 3:
                reflex_idx = np.argmin(signs)
            else:
                reflex_idx = np.argmax(signs)

        # compute a direction if feasible
        if reflex_idx >= 0:
            # grab reflex vertex, the vertex before it, and vertex after it
            p = self.polyhedron_repr.get_vertex(quad[reflex_idx]).aux_data.pos
            q = self.polyhedron_repr.get_vertex(quad[(reflex_idx + 1) % 4]).aux_data.pos
            r = self.polyhedron_repr.get_vertex(quad[(reflex_idx + 3) % 4]).aux_data.pos
            t = self.polyhedron_repr.get_vertex(quad[(reflex_idx + 2) % 4]).aux_data.pos

            # get unit vector between q and r
            u = q - r
            u = u / np.linalg.norm(u)

            # project onto the line segment between q and r
            p_prime = r + u * np.dot(p - r, u)
            p_prime = p_prime / np.linalg.norm(p_prime)

            # set direction as unit vector between p' and p
            direction = p_prime - p
            direction = p - t

            # now remove portion of direction directed towards the center of the sphere
            direction = direction - p * np.dot(direction, p)

            # normalize
            direction = direction / np.linalg.norm(direction)

        # return index in quad array for reflex vertex, something in set {0, 1, 2, 3}
        return reflex_idx, direction

    def _are_triangles_adjacent(self, tri1, tri2):
        set1 = set(tri1)
        set2 = set(tri2)
        intersection = set1.intersection(set2)
        return len(intersection) == 2

    def _form_quad_adjacent_triangles(self, tri1, tri2):
        set1 = set(tri1)
        set2 = set(tri2)
        intersection = set1.intersection(set2)
        sidx = 0
        eidx = 0
        for i in range(0, 3):
            if tri1[i] not in intersection:
                sidx = i
            if tri2[i] not in intersection:
                eidx = i
        return [tri1[sidx], tri1[(sidx + 1) % 3], tri2[eidx], tri1[(sidx + 2) % 3]]

    def _is_quad_nonconvex(self, quad):
        # identify the reflex vertex
        reflex_idx, direction = self._get_reflex_vertex_and_direction(quad)
        return reflex_idx > -1

    def _is_union_nonconvex(self, tri1, tri2):
        # check that the two share two vertices\
        if not self._are_triangles_adjacent(tri1, tri2):
            raise AssertionError

        # merge the triangles into a quad
        quad = self._form_quad_adjacent_triangles(tri1, tri2)

        # check if it is nonconvex
        return self._is_quad_nonconvex(quad)

    def try_find_nonconvex_quad(self):
        face_id_list = self._get_facet_id_lists()
        num_faces = len(face_id_list)
        for i in range(0, num_faces):
            tri_i = face_id_list[i]
            for j in range(i + 1, num_faces):
                tri_j = face_id_list[j]
                if self._are_triangles_adjacent(tri_i, tri_j):
                    if self._is_union_nonconvex(tri_i, tri_j):
                        return True, self._form_quad_adjacent_triangles(tri_i, tri_j), tri_i, tri_j

        return False, None, None, None

    def try_find_all_nonconvex_quads(self):
        list_of_quads = []
        face_id_list = self._get_facet_id_lists()
        num_faces = len(face_id_list)
        for i in range(0, num_faces):
            tri_i = face_id_list[i]
            for j in range(i + 1, num_faces):
                tri_j = face_id_list[j]
                if self._are_triangles_adjacent(tri_i, tri_j):
                    if self._is_union_nonconvex(tri_i, tri_j):
                        list_of_quads.append((self._form_quad_adjacent_triangles(tri_i, tri_j), tri_i, tri_j))
        return list_of_quads

    def _form_simple_baseline_embedding_LP(self, lp_name: str, north_pole_dir: np.array):
        num_vertices = self.polyhedron_repr.num_vertices

        # extract the old coordinates
        old_coordinates = []
        for i in range(0, num_vertices):
            p_i = self.polyhedron_repr.get_vertex(i).aux_data.pos
            old_coordinates.append(np.copy(p_i))

        # use north pole direction to form rotation matrix
        north_pole_dir = north_pole_dir / np.linalg.norm(north_pole_dir)  # ensure normalized
        north_pole_dir.resize((1, 3))
        R = np.zeros((3, 3))
        Rother = null_space(north_pole_dir)  # complete a basis using direction as one of the (unit) basis vectors
        R[:2, :] = Rother.transpose()
        R[2, :] = north_pole_dir  # make the unit direction vector correspond to the new z coordinate

        # rotate the polyhedron's vertices
        poly_repr.rotate(self.polyhedron_repr, R)

        # start forming the linear program
        model = LpProblem(lp_name, LpMinimize)

        # define the variables and constraints we definitely need for the baseline LP.

        # variables representing the "z" coordinates in the rotated frame that we want to assign to each vertex,
        # keeping the other x and y components fixed.
        z_vars = LpVariable.matrix("z", indices=[str(i) for i in range(0, num_vertices)], lowBound=-1, upBound=1)
        # z_vars = LpVariable.matrix("z", indices=[str(i) for i in range(0, num_vertices)])

        # for each triangle, add a constraint to maintain the same signed volume for the induced tetrahedron with
        # the origin as one of its vertices

        # set the area constraints for all faces
        faces_id_list = self._get_facet_id_lists()
        face_constraint_cntr = 0
        for face in faces_id_list:

            # assert the face is a triangle
            if len(face) != 3:
                print(face)
            assert (len(face) == 3)

            # form the constraint for the triangle
            face_orientation = self._orientation_simplex(face)
            # if face_orientation < 0:
            #    print("is this expected?")
            c = np.zeros((3,))
            for i in range(0, 3):
                c[i] = face_orientation * self._xy_determinant(face, i)
            model += lpDot(c, [z_vars[j] for j in face]) >= 0, "Face Constraint " + str(face_constraint_cntr)
            face_constraint_cntr += 1

        # fix vertices to their old values
        for i in range(0, num_vertices):
            self.polyhedron_repr.get_vertex(i).aux_data.pos = old_coordinates[i][:]

        return model, R, z_vars

    def _form_baseline_embedding_LP(self, lp_name: str, north_pole_dir: np.array, use_vertex_constraints: bool = True):
        num_vertices = self.polyhedron_repr.num_vertices

        # extract the old coordinates
        old_coordinates = []
        for i in range(0, num_vertices):
            p_i = self.polyhedron_repr.get_vertex(i).aux_data.pos
            old_coordinates.append(np.copy(p_i))

        # use north pole direction to form rotation matrix
        north_pole_dir = north_pole_dir / np.linalg.norm(north_pole_dir)  # ensure normalized
        north_pole_dir.resize((1, 3))
        R = np.zeros((3, 3))
        Rother = null_space(north_pole_dir)  # complete a basis using direction as one of the (unit) basis vectors
        R[:2, :] = Rother.transpose()
        R[2, :] = north_pole_dir  # make the unit direction vector correspond to the new z coordinate

        # rotate the polyhedron's vertices
        poly_repr.rotate(self.polyhedron_repr, R)

        # start forming the linear program
        model = LpProblem(lp_name, LpMaximize)

        # define the variables and constraints we definitely need for the baseline LP.

        # variable that will be used to maximize the minimum signed volume of each triangle
        # to try and keep things from becoming degenerate
        svol_var = LpVariable(name="svol", lowBound=1e-15)

        # a variable that will try to keep all vertices relatively close to their starting z coordinate value,
        # if it makes sense.
        # delta_var = LpVariable(name="d", lowBound=0)
        delta_vars = LpVariable.matrix("d", indices=[str(i) for i in range(0, num_vertices)], lowBound=0, upBound=20)

        # variables representing the "z" coordinates in the rotated frame that we want to assign to each vertex,
        # keeping the other x and y components fixed.
        z_vars = LpVariable.matrix("z", indices=[str(i) for i in range(0, num_vertices)], lowBound=-10, upBound=10)

        # for each vertex, add in a constraint to attempt to keep each vertex relatively close to where it started
        if use_vertex_constraints:

            # add the constraints
            vertex_constraint_cntr = 0
            for i in range(0, num_vertices):
                p_i = self.polyhedron_repr.get_vertex(i).aux_data.pos
                model += z_vars[i] - p_i[2] <= delta_vars[i], "Vertex Constraint " + str(vertex_constraint_cntr)
                model += z_vars[i] - p_i[2] >= -delta_vars[i], "Vertex Constraint " + str(vertex_constraint_cntr + 1)
                vertex_constraint_cntr += 2

        # for each triangle, add a constraint to maintain the same signed volume for the induced tetrahedron with
        # the origin as one of its vertices

        # set the area constraints for all faces
        faces_id_list = self._get_facet_id_lists()
        face_constraint_cntr = 0
        for face in faces_id_list:

            # assert the face is a triangle
            if len(face) != 3:
                print(face)
            assert (len(face) == 3)

            # form the constraint for the triangle
            face_orientation = self._orientation_simplex(face)
            # print(f"face orientation = {face_orientation}")
            # if face_orientation < 0:
            #    print("is this expected?")
            c = np.zeros((3,))
            for i in range(0, 3):
                c[i] = face_orientation * self._xy_determinant(face, i)
            model += lpDot(c, [z_vars[j] for j in face]) >= svol_var, "Face Constraint " + str(face_constraint_cntr)
            face_constraint_cntr += 1

        # fix vertices to their old values
        for i in range(0, num_vertices):
            self.polyhedron_repr.get_vertex(i).aux_data.pos = old_coordinates[i][:]

        return model, R, svol_var, z_vars, delta_vars

    def _unidirectional_untwist_version1(self, direction: np.array, phi_max=np.pi / 2):
        # will attempt to untwist an embedding along `direction` by trying to make
        # coordinates of triangles with at least two vertices with latitude between [phi_max, -phi_max]
        # as close to the average as possible

        num_vertices = self.polyhedron_repr.num_vertices

        # extract the old coordinates
        old_coordinates = []
        for i in range(0, num_vertices):
            p_i = self.polyhedron_repr.get_vertex(i).aux_data.pos
            old_coordinates.append(np.copy(p_i))

        # grab baseline embedding LP with primary variables
        model, R, svol_var, z_vars, delta_var = self._form_baseline_embedding_LP(lp_name="v1-untwist",
                                                                                 north_pole_dir=direction,
                                                                                 use_vertex_constraints=True)

        # rotate the polygon
        poly_repr.rotate(self.polyhedron_repr, R)

        # add the extra variables and constraints
        faces_id_list = self._get_facet_id_lists()
        eps_vars = LpVariable.matrix("eps", indices=[str(i) for i in range(0, len(faces_id_list))], lowBound=0,
                                     upBound=100.0)

        # add constraints to fix the s and t vertex
        s_idx = num_vertices - 2
        t_idx = num_vertices - 1
        model += z_vars[s_idx] == self.polyhedron_repr.get_vertex(s_idx).aux_data.pos[2], "S Pole Fixed"
        model += z_vars[t_idx] == self.polyhedron_repr.get_vertex(t_idx).aux_data.pos[2], "N Pole Fixed"

        # add constraints on faces
        face_constraint_cntr = 0
        face_index = 0
        for face in faces_id_list:

            # grab the coordinates of the triangle
            vertices = [self.polyhedron_repr.get_vertex(j).aux_data.pos for j in face]

            # check that at least two of the vertices are within the window of latitude values
            num_vertices_within = 0
            for v in vertices:
                phi_val = self._phi_position(v)
                if abs(phi_val) <= phi_max:
                    num_vertices_within += 1

            if num_vertices_within < 2:
                continue

            # add constraints for triangle, given it is within the desired latitude window
            for j in face:
                model += z_vars[j] - (z_vars[face[0]] + z_vars[face[1]] + z_vars[face[2]]) / 3.0 <= eps_vars[
                    face_index], "Avg-Limiting-" + str(face_constraint_cntr)
                model += z_vars[j] - (z_vars[face[0]] + z_vars[face[1]] + z_vars[face[2]]) / 3.0 >= -eps_vars[
                    face_index], "Avg-Limiting-" + str(face_constraint_cntr + 1)
                face_constraint_cntr += 2
            face_index += 1

        # add the objective function
        obj_func = 1e6 * svol_var - 1.0 * lpSum(eps_vars) - 0.0 * delta_var
        model += obj_func
        print(model)

        # try to solve the LP
        model.solve()

        # construct output coordinates
        output_coordinates = []
        if model.status == 1:  # successful case
            print(f"svol_val = {svol_var.value()}")
            Rt = R.transpose()
            for i in range(0, num_vertices):
                pos = self.polyhedron_repr.get_vertex(i).aux_data.pos
                print("{0} = {1}".format(z_vars[i].name, z_vars[i].value()))
                tmp_pos = np.array([pos[0], pos[1], z_vars[i].value()])
                print(tmp_pos)
                if np.linalg.norm(tmp_pos) == 0.0:
                    tmp_pos = np.array([pos[0], pos[1], 1e-4])
                proj_pos = np.matmul(Rt, (tmp_pos / np.linalg.norm(tmp_pos)))
                output_coordinates.append(proj_pos)

        # reset values for position back to normal
        for i in range(0, num_vertices):
            self.polyhedron_repr.get_vertex(i).aux_data.pos[:] = old_coordinates[i][:]

        # return results
        return model.status == 1, output_coordinates, direction

    def _convexify_quad2(self, quad, tri1, tri2):
        # output: (did_succeed, new_coordinates, direction)
        num_vertices = self.polyhedron_repr.num_vertices

        # identify the reflex vertex
        reflex_idx, direction = self._get_reflex_vertex_and_direction(quad)

        # extract the old coordinates
        old_coordinates = []
        for i in range(0, num_vertices):
            p_i = self.polyhedron_repr.get_vertex(i).aux_data.pos
            old_coordinates.append(np.copy(p_i))

        # if did not find any reflex vertex, then implies the quad is convex already
        # so we can return the default result
        if reflex_idx == -1:
            return True, old_coordinates, direction

        direction = old_coordinates[quad[(reflex_idx + 2) % 4]]

        # If the quad is non-convex, then we attempt to find a morph via linear programming to fix it
        # transform coordinates using the new direction
        '''direction.resize((1, 3))
        U = np.zeros((3, 3))
        Uother = null_space(direction)  # complete a basis using direction as one of the (unit) basis vectors
        U[:2, :] = Uother.transpose()
        U[2, :] = direction  # make the unit direction vector correspond to the new z coordinate
        for i in range(0, num_vertices):
            p_i = self.polyhedron_repr.get_vertex(i).aux_data.pos
            self.polyhedron_repr.get_vertex(i).aux_data.pos = np.matmul(U, p_i)
            '''

        # get the initial linear program
        model, R, svol_var, z_vars, delta_vars = self._form_baseline_embedding_LP(lp_name="v2-convexify",
                                                                                  north_pole_dir=direction,
                                                                                  use_vertex_constraints=True)

        # this fixed a bug
        for i in range(0, num_vertices):
            p_i = self.polyhedron_repr.get_vertex(i).aux_data.pos
            self.polyhedron_repr.get_vertex(i).aux_data.pos = np.matmul(R, p_i)

        # set the signed volume constraint for the non-existent triangular face q[i+1]q[i]q[i-1] to ensure
        # that its signed area changes sign as much as possible
        key_face = [quad[(reflex_idx + 3) % 4], quad[reflex_idx], quad[(reflex_idx + 1) % 4]]
        face_orientation = self._orientation_simplex(key_face)
        c = np.zeros((3,))
        for i in range(0, 3):
            c[i] = face_orientation * self._xy_determinant(key_face, i)
        model += -lpDot(c, [z_vars[j] for j in key_face]) >= svol_var, "Convexify Face Constraint "
        model += z_vars[quad[(reflex_idx + 2) % 4]] == 1.0, "Opposite must be north pole eq"

        # add the objective function
        obj_func = 1.0 * svol_var - (1e-1 / num_vertices) * lpSum(delta_vars)
        model += obj_func

        # try to solve the LP
        glpk_solver: pulp.apis.glpk_api.GLPK_CMD = GLPK_CMD()
        model.solve(glpk_solver)

        # construct output coordinates
        output_coordinates = []
        if model.status == 1:  # successful case
            # print(f"svol_val = {svol_var.value()}")
            Rt = R.transpose()
            for i in range(0, num_vertices):
                pos = self.polyhedron_repr.get_vertex(i).aux_data.pos
                # print("{0} = {1}".format(z_vars[i].name, z_vars[i].value()))
                tmp_pos = np.array([pos[0], pos[1], z_vars[i].value()])
                # print(tmp_pos)
                if np.linalg.norm(tmp_pos) == 0.0:
                    tmp_pos = np.array([pos[0], pos[1], 1e-4])
                proj_pos = np.matmul(Rt, (tmp_pos / np.linalg.norm(tmp_pos)))
                output_coordinates.append(proj_pos)

        # reset values for position back to normal
        for i in range(0, num_vertices):
            self.polyhedron_repr.get_vertex(i).aux_data.pos[:] = old_coordinates[i][:]

        # return results
        return (model.status == 1), (svol_var.value() > 0.0), output_coordinates, direction, R

    def _convexify_quad2_threefixed(self, quad, tri1, tri2):
        # output: (did_succeed, new_coordinates, direction)
        num_vertices = self.polyhedron_repr.num_vertices

        # identify the reflex vertex
        reflex_idx, direction = self._get_reflex_vertex_and_direction(quad)

        # extract the old coordinates
        old_coordinates = []
        for i in range(0, num_vertices):
            p_i = self.polyhedron_repr.get_vertex(i).aux_data.pos
            old_coordinates.append(np.copy(p_i))

        # if did not find any reflex vertex, then implies the quad is convex already
        # so we can return the default result
        if reflex_idx == -1:
            return True, old_coordinates, direction

        direction = old_coordinates[quad[(reflex_idx + 2) % 4]]

        # If the quad is non-convex, then we attempt to find a morph via linear programming to fix it
        # transform coordinates using the new direction
        '''direction.resize((1, 3))
        U = np.zeros((3, 3))
        Uother = null_space(direction)  # complete a basis using direction as one of the (unit) basis vectors
        U[:2, :] = Uother.transpose()
        U[2, :] = direction  # make the unit direction vector correspond to the new z coordinate
        for i in range(0, num_vertices):
            p_i = self.polyhedron_repr.get_vertex(i).aux_data.pos
            self.polyhedron_repr.get_vertex(i).aux_data.pos = np.matmul(U, p_i)
            '''

        # get the initial linear program
        model, R, svol_var, z_vars, delta_var = self._form_baseline_embedding_LP(lp_name="v2-convexify",
                                                                                 north_pole_dir=direction,
                                                                                 use_vertex_constraints=False)

        # this fixed a bug
        for i in range(0, num_vertices):
            p_i = self.polyhedron_repr.get_vertex(i).aux_data.pos
            self.polyhedron_repr.get_vertex(i).aux_data.pos = np.matmul(R, p_i)

        # set the signed volume constraint for the non-existent triangular face q[i+1]q[i]q[i-1] to ensure
        # that its signed area changes sign as much as possible
        key_face = [quad[(reflex_idx + 3) % 4], quad[reflex_idx], quad[(reflex_idx + 1) % 4]]
        face_orientation = self._orientation_simplex(key_face)
        c = np.zeros((3,))
        for i in range(0, 3):
            c[i] = face_orientation * self._xy_determinant(key_face, i)
        model += -lpDot(c, [z_vars[j] for j in key_face]) >= svol_var, "Convexify Face Constraint "
        model += z_vars[quad[(reflex_idx + 2) % 4]] == 1.0, "Opposite must be north pole eq"
        model += z_vars[quad[(reflex_idx + 1) % 4]] == 0.0, "Neighbor 1 must lie on equator eq"
        model += z_vars[quad[(reflex_idx + 3) % 4]] == 0.0, "Neighbor 2 must lie on equator eq"

        # add the objective function
        obj_func = 1.0 * svol_var - 0 * delta_var
        model += obj_func

        # try to solve the LP
        model.solve()

        # construct output coordinates
        output_coordinates = []
        if model.status == 1:  # successful case
            print(f"svol_val = {svol_var.value()}")
            Rt = R.transpose()
            for i in range(0, num_vertices):
                pos = self.polyhedron_repr.get_vertex(i).aux_data.pos
                print("{0} = {1}".format(z_vars[i].name, z_vars[i].value()))
                tmp_pos = np.array([pos[0], pos[1], z_vars[i].value()])
                print(tmp_pos)
                if np.linalg.norm(tmp_pos) == 0.0:
                    tmp_pos = np.array([pos[0], pos[1], 1e-4])
                proj_pos = np.matmul(Rt, (tmp_pos / np.linalg.norm(tmp_pos)))
                output_coordinates.append(proj_pos)

        # reset values for position back to normal
        for i in range(0, num_vertices):
            self.polyhedron_repr.get_vertex(i).aux_data.pos[:] = old_coordinates[i][:]

        # return results
        return model.status == 1, output_coordinates, direction, R

    def _convexify_quad3(self, quad, tri1, tri2):
        # output: (did_succeed, new_coordinates, direction)
        num_vertices = self.polyhedron_repr.num_vertices

        # identify the reflex vertex
        reflex_idx, direction = self._get_reflex_vertex_and_direction(quad)

        # extract the old coordinates
        old_coordinates = []
        for i in range(0, num_vertices):
            p_i = self.polyhedron_repr.get_vertex(i).aux_data.pos
            old_coordinates.append(np.copy(p_i))

        # if did not find any reflex vertex, then implies the quad is convex already
        # so we can return the default result
        if reflex_idx == -1:
            return True, old_coordinates, direction

        north_pole = old_coordinates[quad[(reflex_idx + 2) % 4]]

        # If the quad is non-convex, then we attempt to find a morph via linear programming to fix it
        # transform coordinates using the new direction
        # north_pole.resize((1, 3))
        # U = np.zeros((3, 3))
        # Uother = null_space(north_pole)  # complete a basis using direction as one of the (unit) basis vectors
        # U[:2, :] = Uother.transpose()
        # U[2, :] = north_pole  # make the unit direction vector correspond to the new z coordinate
        # for i in range(0, num_vertices):
        #    p_i = self.polyhedron_repr.get_vertex(i).aux_data.pos
        #    self.polyhedron_repr.get_vertex(i).aux_data.pos = np.matmul(U, p_i)

        # get the initial linear program
        model, R, z_vars = self._form_simple_baseline_embedding_LP(lp_name="v3-convexify", north_pole_dir=north_pole)

        model += z_vars[quad[reflex_idx]] <= 0.0, "Reflex Vertex Max constraint"
        model += z_vars[quad[reflex_idx]] >= -10, "Reflect Vertex Min constraint"
        model += z_vars[quad[(reflex_idx + 1) % 4]] == 0.0, "Neighbor 1 must lie on equator eq"
        model += z_vars[quad[(reflex_idx + 3) % 4]] == 0.0, "Neighbor 2 must lie on equator eq"
        model += z_vars[quad[(reflex_idx + 2) % 4]] == 1.0, "Opposite must be north pole eq"

        for i in range(0, num_vertices):
            p_i = self.polyhedron_repr.get_vertex(i).aux_data.pos
            self.polyhedron_repr.get_vertex(i).aux_data.pos = np.matmul(R, p_i)

        p_r = self.polyhedron_repr.get_vertex(quad[reflex_idx]).aux_data.pos
        r_r = np.linalg.norm(p_r[:2])
        p_left = self.polyhedron_repr.get_vertex(quad[(reflex_idx + 1) % 4]).aux_data.pos
        r_left = np.linalg.norm(p_left[:2])
        p_right = self.polyhedron_repr.get_vertex(quad[(reflex_idx + 3) % 4]).aux_data.pos
        r_right = np.linalg.norm(p_right[:2])
        # model += z_vars[quad[reflex_idx]] <= -1e-2 + z_vars[quad[(reflex_idx + 1) % 4]]*(r_r / r_left), "Neighbor 1 constraint"
        # model += z_vars[quad[reflex_idx]] <= -1e-2 + z_vars[quad[(reflex_idx + 3) % 4]] * (r_r / r_right), "Neighbor 2 constraint"

        # set the signed volume constraint for the non-existent triangular face q[i+1]q[i]q[i-1] to ensure
        # that its signed area changes sign as much as possible
        # key_face = [quad[(reflex_idx + 3) % 4], quad[reflex_idx], quad[(reflex_idx + 1) % 4]]
        # face_orientation = self._orientation_simplex(key_face)
        # c = np.zeros((3,))
        # for i in range(0, 3):
        #    c[i] = face_orientation * self._xy_determinant(key_face, i)
        # model += -lpDot(c, [z_vars[j] for j in key_face]) >= 1e-1, "Convexify Face Constraint "

        # add the objective function
        obj_func = -z_vars[quad[reflex_idx]]
        model += obj_func

        # print(model)

        # try to solve the LP
        model.solve()

        # construct output coordinates
        output_coordinates = []
        if model.status == 1:  # successful case
            print(z_vars[quad[reflex_idx]].value())
            Rt = R.transpose()
            for i in range(0, num_vertices):
                pos = self.polyhedron_repr.get_vertex(i).aux_data.pos
                print("{0} = {1}".format(z_vars[i].name, z_vars[i].value()))
                if z_vars[i].value() is None:
                    tmp_pos = np.array([pos[0], pos[1], 0.0])
                else:
                    tmp_pos = np.array([pos[0], pos[1], z_vars[i].value()])
                print(tmp_pos)
                if np.linalg.norm(tmp_pos) == 0.0:
                    tmp_pos = np.array([pos[0], pos[1], 1e-4])
                proj_pos = np.matmul(Rt, (tmp_pos / np.linalg.norm(tmp_pos)))
                output_coordinates.append(proj_pos)
        else:
            print(f"Not successful! Output status is {model.status}")

        # reset values for position back to normal
        for i in range(0, num_vertices):
            self.polyhedron_repr.get_vertex(i).aux_data.pos[:] = old_coordinates[i][:]

        # return results
        return (model.status == 1) and (z_vars[quad[reflex_idx]].value() > 0), output_coordinates, north_pole, R

    def _convexify_quad(self, quad, tri1, tri2):
        # output: (did_succeed, new_coordinates, direction)
        num_vertices = self.polyhedron_repr.num_vertices

        # identify the reflex vertex
        reflex_idx, direction = self._get_reflex_vertex_and_direction(quad)

        # extract the old coordinates
        old_coordinates = []
        for i in range(0, num_vertices):
            p_i = self.polyhedron_repr.get_vertex(i).aux_data.pos
            old_coordinates.append(np.copy(p_i))

        # if did not find any reflex vertex, then implies the quad is convex already
        # so we can return the default result
        if reflex_idx == -1:
            return True, old_coordinates, direction

        # If the quad is non-convex, then we attempt to find a morph via linear programming to fix it
        # transform coordinates using the new direction
        direction.resize((1, 3))
        U = np.zeros((3, 3))
        Uother = null_space(direction)  # complete a basis using direction as one of the (unit) basis vectors
        U[:2, :] = Uother.transpose()
        U[2, :] = direction  # make the unit direction vector correspond to the new z coordinate
        for i in range(0, num_vertices):
            p_i = self.polyhedron_repr.get_vertex(i).aux_data.pos
            self.polyhedron_repr.get_vertex(i).aux_data.pos = np.matmul(U, p_i)

        # built the linear program instance
        model = LpProblem("Quad Convexification Morph", LpMaximize)

        # for this linear program, we want to enforce that area constraints for the simplicial faces
        # maintain the same signed area even as we slide things along the specified direction
        # additionally,
        # if i is the reflex vertex index in the quad array q[0..3], then we want the triangle q[i+1]q[i]q[i-1] to have
        # its signed area flip. Note that this triangle should not actually exist in the graph and if it does, then
        # this LP naturally cannot have a solution

        # setup variables for the LP
        # setup the variables for the linear program
        A_var = LpVariable(name="A", lowBound=1e-15)
        delta_var = LpVariable(name="d", lowBound=0)
        z_vars = LpVariable.matrix("z", indices=[str(i) for i in range(0, num_vertices)], lowBound=-10, upBound=10)

        # set objective function
        obj_func = A_var - 1000 * delta_var
        model += obj_func

        # set the area constraints for all faces
        faces_id_list = self._get_facet_id_lists()

        # vertex constraints, to limit motion of z coordinates if possible
        vertex_constraint_cntr = 0
        for i in range(0, num_vertices):
            p_i = self.polyhedron_repr.get_vertex(i).aux_data.pos
            model += z_vars[i] - p_i[2] <= delta_var, "Vertex Constraint " + str(vertex_constraint_cntr)
            model += z_vars[i] - p_i[2] >= -delta_var, "Vertex Constraint " + str(vertex_constraint_cntr + 1)
            vertex_constraint_cntr += 2

        face_constraint_cntr = 0
        for face in faces_id_list:
            if len(face) != 3:
                continue
            face_set = set(face)
            if face_set == set(tri1) or face_set == set(tri2):
                continue
            # form the constraint for the triangle
            face_orientation = self._orientation_simplex(face)
            if face_orientation < 0:
                print("is this expected?")
            c = np.zeros((3,))
            for i in range(0, 3):
                c[i] = face_orientation * self._xy_determinant(face, i)
            model += lpDot(c, [z_vars[j] for j in face]) >= 1e-10, "Face Constraint " + str(face_constraint_cntr)
            face_constraint_cntr += 1

        # set the area constraint for the non-existent triangular face q[i+1]q[i]q[i-1] to ensure
        # that its signed area changes sign as much as possible
        key_face = [quad[(reflex_idx + 3) % 4], quad[reflex_idx], quad[(reflex_idx + 1) % 4]]
        face_orientation = self._orientation_simplex(key_face)
        c = np.zeros((3,))
        for i in range(0, 3):
            c[i] = face_orientation * self._xy_determinant(key_face, i)
        model += -lpDot(c, [z_vars[j] for j in key_face]) >= A_var, "Face Constraint " + str(face_constraint_cntr)
        face_constraint_cntr += 1

        # save the LP
        model.writeLP("conv_quad_morph.lp")

        # print(model)

        # try to solve the LP
        model.solve()

        # construct output coordinates
        output_coordinates = []
        if model.status == 1:  # successful case
            Ut = U.transpose()
            for i in range(0, num_vertices):
                pos = self.polyhedron_repr.get_vertex(i).aux_data.pos
                print("{0} = {1}".format(z_vars[i].name, z_vars[i].value()))
                tmp_pos = np.array([pos[0], pos[1], z_vars[i].value()])
                if np.linalg.norm(tmp_pos) == 0.0:
                    tmp_pos = np.array([pos[0], pos[1], 1e-4])
                proj_pos = np.matmul(Ut, (tmp_pos / np.linalg.norm(tmp_pos)))
                output_coordinates.append(proj_pos)

        # reset values for position back to normal
        for i in range(0, num_vertices):
            self.polyhedron_repr.get_vertex(i).aux_data.pos[:] = old_coordinates[i][:]

        # return results
        direction.resize((3,))
        return model.status == 1, output_coordinates, direction

    def try_morph_into_southern_hemisphere(self, north_pole_idx: int = 0):
        num_vertices = self.polyhedron_repr.num_vertices
        if north_pole_idx < 0 or north_pole_idx >= num_vertices:
            raise IndexError()
        else:

            # attempt to morph all of the embedding except for the north pole into the southern hemisphere
            # such that the subcomplex is convex, returning new coordinate positions if possible and other
            # returning it was not successful
            (did_succeed, new_coordinates, Aval) = self._morph_into_shemisphere_with_convex_boundary(north_pole_idx)

            # if the above not successful, then slide the vertices along longitudes as much as possible
            # if not did_succeed:
            #    new_coordinates = self._morph_along_longitudes_with_north_pole(north_pole_idx)

            return (did_succeed, new_coordinates)

    def _all_signed_vols_have_same_sign(self):
        facet_id_lists = self._get_facet_id_lists()
        base_orientation = self._orientation_simplex(facet_id_lists[0])
        for face in facet_id_lists:
            face_orientation = self._orientation_simplex(face)
            if face_orientation * base_orientation < 1.0:
                return False
        return True

    def _normalize_internal_polygon(self):
        num_vertices = self.polyhedron_repr.num_vertices
        for i in range(0, num_vertices):
            p_i = self.polyhedron_repr.get_vertex(i).aux_data.pos
            p_i = p_i / np.linalg.norm(p_i)
            self.polyhedron_repr.get_vertex(i).aux_data.pos = p_i

    def reorient_with_chosen_north_pole(self, north_pole_idx):
        num_vertices = self.polyhedron_repr.num_vertices
        if north_pole_idx < 0 or north_pole_idx >= num_vertices:
            raise IndexError()
        else:
            pass

    def _morph_into_shemisphere_with_convex_boundary(self, north_pole_idx):
        num_vertices = self.polyhedron_repr.num_vertices
        theta = -np.pi / 10.0

        # built the linear program instance
        model = LpProblem("South Hemisphere Morph", LpMaximize)

        # compute the faces of our sphere
        vertices_in_link = [False] * num_vertices
        zTheta = np.zeros((num_vertices,))
        faces_id_list = self._get_facet_id_lists()

        # figure out which vertices are not in the link of N
        for face in faces_id_list:
            if len(face) != 3:
                raise AssertionError

            if north_pole_idx in face:
                # mark all vertices except for north pole since in link of north pole
                for i in range(0, 3):
                    vid = face[i]
                    if not vertices_in_link[vid]:
                        vertices_in_link[vid] = True

        # extract the old coordinates
        old_coordinates = []
        for i in range(0, num_vertices):
            p_i = self.polyhedron_repr.get_vertex(i).aux_data.pos
            old_coordinates.append(np.copy(p_i))

        # If the quad is non-convex, then we attempt to find a morph via linear programming to fix it
        # transform coordinates using the new direction
        direction = np.copy(self.polyhedron_repr.get_vertex(north_pole_idx).aux_data.pos)
        direction.resize((1, 3))
        U = np.zeros((3, 3))
        Uother = null_space(direction)  # complete a basis using direction as one of the (unit) basis vectors
        U[:2, :] = Uother.transpose()
        U[2, :] = direction  # make the unit direction vector correspond to the new z coordinate
        for i in range(0, num_vertices):
            p_i = self.polyhedron_repr.get_vertex(i).aux_data.pos
            self.polyhedron_repr.get_vertex(i).aux_data.pos = np.matmul(U, p_i)

        # compute z(v,theta) for each v != N
        for i in range(0, num_vertices):
            if i != north_pole_idx:
                zTheta[i] = self._compute_ztheta(self.polyhedron_repr.get_vertex(i).aux_data.pos, theta)

        # setup the variables for the linear program
        A_var = LpVariable(name="A", lowBound=0)
        z_vars = LpVariable.matrix("z", indices=[str(i) for i in range(0, num_vertices)])

        # set objective function
        obj_func = A_var + 0 * lpDot(z_vars, np.ones((num_vertices,))) / len(z_vars)
        model += obj_func

        # set constraints
        # compute the faces of our sphere
        vertices_in_link = [False] * num_vertices

        # add constraints on all triangles
        face_constraint_cntr = 1
        for face in faces_id_list:
            if len(face) != 3:
                raise AssertionError

            if north_pole_idx in face:
                # mark all vertices except for north pole since in link of north pole
                vertices_in_link[face[0]] = True
                vertices_in_link[face[1]] = True
                vertices_in_link[face[2]] = True

            else:
                # form the constraint for the triangle
                face_orientation = self._orientation_simplex(face)
                if face_orientation < 0:
                    print("is this expected?")
                c = np.zeros((3,))
                for i in range(0, 3):
                    c[i] = face_orientation * self._xy_determinant(face, i)
                model += lpDot(c, [z_vars[j] for j in face]) >= A_var, "Face Constraint " + str(face_constraint_cntr)
                face_constraint_cntr += 1

        # add constraints for the vertices
        link_constraint_counter = 1
        non_link_constraint_counter = 1
        for i in range(0, num_vertices):

            if i != north_pole_idx:

                model += z_vars[i] >= -10, "Vertex Z_{0} lower bound ".format(i)

                # if link vertex
                if vertices_in_link[i]:
                    model += z_vars[i] == zTheta[i], "Link Constraint " + str(link_constraint_counter)
                    link_constraint_counter += 1

                # if not link vertex
                # else:
                #    model += z_vars[i] <= zTheta[i], "Non-link Constraint " + str(non_link_constraint_counter)
                #    non_link_constraint_counter += 1

        # add constraints for the north pole vertex
        model += z_vars[north_pole_idx] == 1.0, "North pole fixed Constraint "

        # save the LP
        model.writeLP("sh_morph.lp")

        # print(model)

        # try to solve the LP
        model.solve()

        # construct output coordinates
        output_coordinates = []
        if model.status == 1:  # successful case
            for i in range(0, num_vertices):
                pos = self.polyhedron_repr.get_vertex(i).aux_data.pos
                print("{0} == {1}".format(z_vars[i].name, z_vars[i].value()))
                tmp_pos = np.array([pos[0], pos[1], z_vars[i].value()])
                if np.linalg.norm(tmp_pos) == 0.0:
                    tmp_pos = np.array([pos[0], pos[1], 1e-4])
                output_coordinates.append(tmp_pos / np.linalg.norm(tmp_pos))

        # reset values for position back to normal
        for i in range(0, num_vertices):
            self.polyhedron_repr.get_vertex(i).aux_data.pos[:] = old_coordinates[i][:]

        # return results
        return model.status == 1, output_coordinates, A_var.value()

    def _morph_along_longitudes_with_north_pole(self, north_pole_idx):
        # setup the variables for the linear program
        # set objective function
        # set constraints
        pass

    def _compute_ztheta(self, pos, theta):
        return np.sqrt(pos[0] ** 2 + pos[1] ** 2) * np.tan(theta)

    def _xy_determinant(self, face_ids, idx):
        # face_ids is a list of three IDs for vertices that are a triangular face of the embedding
        # assume order is [v0, v1, v2]

        # extract vertices from face
        v0 = self.polyhedron_repr.get_vertex(face_ids[0]).aux_data.pos
        v1 = self.polyhedron_repr.get_vertex(face_ids[1]).aux_data.pos
        v2 = self.polyhedron_repr.get_vertex(face_ids[2]).aux_data.pos

        X = np.zeros((2, 2))
        coef = (-1) ** idx
        if idx == 0:
            X = np.array([[v1[0], v1[1]], [v2[0], v2[1]]])
        elif idx == 1:
            X = np.array([[v0[0], v0[1]], [v2[0], v2[1]]])
        else:  # idx == 2
            X = np.array([[v0[0], v0[1]], [v1[0], v1[1]]])
        return coef * np.linalg.det(X)

    def _orientation_simplex(self, face_ids):
        # face_ids is a list of three IDs for vertices that are a triangular face of the embedding

        # extract vertices from face
        v0 = self.polyhedron_repr.get_vertex(face_ids[0]).aux_data.pos
        v1 = self.polyhedron_repr.get_vertex(face_ids[1]).aux_data.pos
        v2 = self.polyhedron_repr.get_vertex(face_ids[2]).aux_data.pos

        # form matrix X where each row is a point in R^3 for a triangular face of the embedding
        X = np.array([[v0[0], v0[1], v0[2]],
                      [v1[0], v1[1], v1[2]],
                      [v2[0], v2[1], v2[2]]])

        # computing the sign of the determinant will give the orientation for the face
        return np.sign(np.linalg.det(X))

    def _get_facet_id_lists(self):
        faces_list, edge2facet_idx = self.polyhedron_repr._get_faces()
        nf = len(faces_list)
        facet_id_lists = []
        for i in range(0, nf):
            face = faces_list[i]
            id_list = []
            for dart in face:
                id_list.append(dart.tail.id)
            facet_id_lists.append(id_list)
        return facet_id_lists

    def _build_unit_sphere_triangulation(self):
        (n, m) = (30, 50)

        # Meshing a unit sphere according to n, m
        theta = np.linspace(0, 2 * np.pi, num=n, endpoint=False)
        phi = np.linspace(np.pi * (-0.5 + 1. / (m + 1)), np.pi * 0.5, num=m, endpoint=False)
        theta, phi = np.meshgrid(theta, phi)
        theta, phi = theta.ravel(), phi.ravel()
        theta = np.append(theta, [0.])  # Adding the north pole...
        phi = np.append(phi, [np.pi * 0.5])
        radius_plt = 1.
        mesh_x, mesh_y = (
            radius_plt * (np.pi * 0.5 - phi) * np.cos(theta), radius_plt * (np.pi * 0.5 - phi) * np.sin(theta))
        self.triangles = mtri.Triangulation(mesh_x, mesh_y).triangles
        self.x, self.y, self.z = radius_plt * np.cos(phi) * np.cos(theta), radius_plt * np.cos(phi) * np.sin(
            theta), radius_plt * np.sin(phi)

    def _draw_edge(self, ax, dart):
        # TODO: use unit quaternions to compute locations on the sphere for use in visualizing edges
        alpha_values = np.linspace(0.0, 1.0, 20)
        p_start = dart.tail.aux_data.pos / np.linalg.norm(dart.tail.aux_data.pos)
        p_end = dart.head.aux_data.pos / np.linalg.norm(dart.head.aux_data.pos)
        p_last = p_start
        for i in range(1, len(alpha_values)):
            p = p_start * (1.0 - alpha_values[i]) + p_end * alpha_values[i]
            u_p = p / np.linalg.norm(p)
            ax.plot3D(np.array([p_last[0], u_p[0]]), np.array([p_last[1], u_p[1]]), np.array([p_last[2], u_p[2]]),
                      color="black", linewidth=1.0, zorder=3.0)
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

    def _draw_vertices(self, ax, other_vertices=None):
        default_color = (0.6, 0, 1.0)
        identifying_color = (1.0, 0, 0.5)
        id1 = 0
        id2 = 0
        id = 0
        n = self.polyhedron_repr.num_vertices
        num_others = 0
        if other_vertices is not None:
            num_others = len(other_vertices)
        x_vals = np.zeros((n - num_others,))
        y_vals = np.zeros((n - num_others,))
        z_vals = np.zeros((n - num_others,))
        x_vals_other = np.zeros((num_others,))
        y_vals_other = np.zeros((num_others,))
        z_vals_other = np.zeros((num_others,))
        for v in self.polyhedron_repr.vertices:
            # compute normalized vector
            u = v.aux_data.pos / np.linalg.norm(v.aux_data.pos)
            if other_vertices is not None:
                if id in other_vertices:
                    x_vals_other[id2] = u[0]
                    y_vals_other[id2] = u[1]
                    z_vals_other[id2] = u[2]
                    id2 += 1
                else:
                    x_vals[id1] = u[0]
                    y_vals[id1] = u[1]
                    z_vals[id1] = u[2]
                    id1 += 1
            else:
                x_vals[id1] = u[0]
                y_vals[id1] = u[1]
                z_vals[id1] = u[2]
                id1 += 1

            # update the vertex index
            id += 1

        # plot the vertices
        ax.scatter(x_vals,
                   y_vals,
                   z_vals, marker="o", linewidth=0.5,
                   color=default_color,
                   edgecolors="black",
                   zorder=3.1)

        ax.scatter(x_vals_other,
                   y_vals_other,
                   z_vals_other, marker="o", linewidth=0.5,
                   color=identifying_color,
                   edgecolors="black",
                   zorder=3.1)

