import numpy as np


def _normalize_edge_vertices(edge: [np.array]):
    edge[0] = edge[0] / np.linalg.norm(edge[0])
    edge[1] = edge[1] / np.linalg.norm(edge[1])


def _input_verification(initial_vertices: [np.array], initial_triangles: [[int]], new_edge):
    # check that all vertices are 3-dim with (approximately) unit length
    for v in initial_vertices:
        if len(v) != 3:
            raise Exception("Error: At least one polyhedron vertex is not 3-dimensional")

        if np.abs(1.0 - np.linalg.norm(v)) > 1e-6:
            raise Exception(
                f"Error: At least one polyhedron vertex is not (approximately) unit length ({np.linalg.norm(v)})")

    # Check that all faces in our list of triangles have exactly three entries with integers within the allowed range
    # based on the number of vertices we were given.
    num_vertices = len(initial_vertices)
    for tri in initial_triangles:
        if len(tri) != 3:
            raise Exception("Error: At least one face is not comprised of three vertices")

        if len(set(tri)) < 3:
            raise Exception("Error: At least one face has one vertex listed at least 2 times")

        for index in tri:
            if index < 0 or index >= num_vertices:
                raise Exception("Error: At least one face is using an invalid vertex index")

        # check that the volume is positive
        A = np.zeros((3, 3))
        A[0, :] = initial_vertices[tri[0]]
        A[1, :] = initial_vertices[tri[1]]
        A[2, :] = initial_vertices[tri[2]]
        volume = np.abs(np.linalg.det(A))
        if volume <= 0.0:
            raise Exception("Error: an input triangle does not have the right orientation")

    # check that the edge has two elements
    if len(new_edge) != 2:
        raise Exception("Error: New edge is not comprised of exactly two vertices")

    # check that the elements take on reasonable values
    edge_vertices = []
    for i in range(0, 2):
        if type(new_edge[i]) is int:
            if new_edge[i] < 0 or new_edge[i] >= num_vertices:
                raise Exception(f"Error: new_edge[{i}] has an invalid vertex ID of {new_edge[i]}")
            else:
                edge_vertices.append(initial_vertices[new_edge[i]])
        elif type(new_edge[i]) is np.ndarray:
            if len(new_edge[i]) != 3:
                raise Exception(f"Error: new_edge[{i}] is not 3-dimensional")
            else:
                new_edge[i] = new_edge[i] / np.linalg.norm(new_edge[i])
                edge_vertices.append(new_edge[i])

    # Check that the two vertices of the edge are not actually the same.
    if type(new_edge[0]) is int and type(new_edge[1]) is int:
        if new_edge[0] == new_edge[1]:
            raise Exception("Error: both vertices of new_edge have the same vertex ID")

    # Check that the edge is not a single point.
    if np.linalg.norm(edge_vertices[0] - edge_vertices[1]) < 1e-10:
        raise Exception("Error: both vertices of new_edge have the same 3-d coordinate, making this edge a point.")

    # Check that the edge is not a long edge
    if np.linalg.norm(edge_vertices[0] - edge_vertices[1]) > 2.0 - 1e-10:
        raise Exception("Error: the new edge being added is being found to correspond to a long geodesic")


def _get_edge_normal(edge: [np.array]):
    return np.cross(edge[0], edge[1])


def _point_in_geodesic(p: np.array, e: [np.array]):
    # Assumes p is a unit vector and vertices of e are unit vectors
    eps = 1e-10
    n = _get_edge_normal(e)
    if np.abs(np.dot(p, n)) > eps:
        return False
    return np.dot(n, np.cross(e[0], p)) >= 0.0 and np.dot(n, np.cross(p, e[1])) >= 0.0


def _point_in_great_circle_of_edge(p: np.array, e: [np.array]):
    # Assumes p is a unit vector and vertices of e are unit vectors
    eps = 1e-10
    n = _get_edge_normal(e)
    return np.abs(np.dot(p, n)) <= eps


def _point_above_great_circle_of_edge(p: np.array, e: [np.array]):
    # Assumes p is a unit vector and vertices of e are unit vectors
    eps = 1e-10
    n = _get_edge_normal(e)
    return np.dot(p, n) > eps


def _point_below_great_circle_of_edge(p: np.array, e: [np.array]):
    # Assumes p is a unit vector and vertices of e are unit vectors
    eps = 1e-10
    n = _get_edge_normal(e)
    return np.dot(p, n) < -eps


def _edges_cross(e1: [np.array], e2: [np.array]):
    # Output a mark that denotes if the edges cross and any extra info.
    # If return -1, then e2 is below great circle that contains e1
    # If return 0, then e2 intersects e1
    # If return 1, then e2 is above great circle that contains e1
    # If return 2, then e2 crosses great circle that contains e1 but does not intersect e1 and has vertex e2[1]
    # above the great circle
    # If return -2, then e2 crosses great circle that contains e1 but does not intersect e1 and has vertex e2[1]
    # below the great circle
    # If return 3, then none of the above (i.e. e2 lives on the great circle that e1 is on, but does not intersect e1)

    # Check if either vertex of e2 is contained in e1, implying the two edges intersect
    if _point_in_geodesic(e2[0], e1) or _point_in_geodesic(e2[1], e1):
        return 0

    # If e1 is contained in e2, then e2 must be removed
    if _point_in_geodesic(e1[0], e2) and _point_in_geodesic(e1[1], e2):
        return 0

    # Handle all edges marked as "completely above" the great circle containing e1
    # Note that if one vertex of e2 lives on the great circle of e1 and the other is above the great circle, then we
    # still consider this segment as "above"
    for i in range(0, 2):
        if (_point_in_great_circle_of_edge(e2[i], e1) or _point_above_great_circle_of_edge(e2[i], e1)) \
                and _point_above_great_circle_of_edge(e2[(i + 1) % 2], e1):
            return 1

    # Handle all edges marked as "completely below" the great circle containing e1
    # Note that if one vertex of e2 lives on the great circle of e1 and the other is below the great circle, then we
    # still consider this segment as "below"
    for i in range(0, 2):
        if (_point_in_great_circle_of_edge(e2[i], e1) or _point_below_great_circle_of_edge(e2[i], e1)) \
                and _point_below_great_circle_of_edge(e2[(i + 1) % 2], e1):
            return -1

    # Investigate other possible intersections.
    # We say e2 crosses e1 if there exists some a in [0, 1] such that the normalization of
    # e2[0]*a + e2[1]*(1 - a) to the unit sphere is contained in the geodesic corresponding to e1.
    # Check that both points are on different sides of the great circle
    n1 = _get_edge_normal(e1)
    s_e20 = np.sign(np.dot(e2[0], n1))
    s_e21 = np.sign(np.dot(e2[1], n1))
    if s_e20 != s_e21:

        # let p = e2[0] and p' = e2[1]
        # let p_c = p * alpha + p' * (1 - alpha) be the vector on the line segment between p and p' that lives in the
        # plane that e1 lives on. compute p_c.
        alpha = -np.dot(n1, e2[1]) / np.dot(n1, e2[0] - e2[1])
        p_c = e2[0] * alpha + e2[1] * (1 - alpha)

        # If p_c is within the geodesic of e1, then return 0
        if _point_in_geodesic(p_c, e1):
            return 0

        # Otherwise e2 only intersects the great circle of e1, so we need to identify the type of intersection we  have.
        if _point_below_great_circle_of_edge(e2[1], e1):
            return -2

        if _point_above_great_circle_of_edge(e2[1], e1):
            return 2

    # If e2 lives on the same great circle as e1 but does not intersect e1, then return 3
    return 3


def _edges_cross2(e1_in, e2_in: [int], vertices: [np.array]):
    # Output a mark that denotes if the edges cross and any extra info.
    # If return -1, then e2 is below great circle that contains e1
    # If return 0, then e2 intersects e1
    # If return 1, then e2 is above great circle that contains e1
    # If return 2, then e2 crosses great circle that contains e1 but does not intersect e1 and has vertex e2[1]
    # above the great circle
    # If return -2, then e2 crosses great circle that contains e1 but does not intersect e1 and has vertex e2[1]
    # below the great circle
    # If return 3, then none of the above (i.e. e2 lives on the great circle that e1 is on, but does not intersect e1)

    # get the 3-d representation of e1
    e1 = []
    is_id = []
    for i in range(0, 2):
        if type(e1_in[i]) is int:
            e1.append(vertices[e1_in[i]])
            is_id.append(True)
        elif type(e1_in[i]) is np.ndarray:
            e1.append(e1_in[i])
            is_id.append(False)

    # get the 3-d representation of e2
    e2 = [vertices[e2_in[0]], vertices[e2_in[1]]]

    # handle some cases in the event e2 and e1 share some of the same vertices
    for i in range(0, 2):
        if is_id[i] and (e1_in[i] == e2_in[0]):
            if _point_above_great_circle_of_edge(e2[1], e1):
                return 1
            elif _point_below_great_circle_of_edge(e2[1], e1):
                return -1
            elif _point_in_geodesic(e2[1], e1) or _point_in_geodesic(e1[(i + 1) % 2], e2):
                return 0
            else:
                return 3
        if is_id[i] and (e1_in[i] == e2_in[1]):
            if _point_above_great_circle_of_edge(e2[0], e1):
                return 1
            elif _point_below_great_circle_of_edge(e2[0], e1):
                return -1
            elif _point_in_geodesic(e2[0], e1) or _point_in_geodesic(e1[(i + 1) % 2], e2):
                return 0
            else:
                return 3

    # Now given they don't share any of the same vertices, perform the usual checks
    return _edges_cross(e1, e2)


def _get_edge_coordinates_of_triangle(triangle: [int], vertices: [np.array]):
    return [[vertices[triangle[0]], vertices[triangle[1]]], [vertices[triangle[1]], vertices[triangle[2]]],
            [vertices[triangle[2]], vertices[triangle[0]]]]


def _get_edge_ids_of_triangle(triangle: [int]):
    return [[triangle[0], triangle[1]], [triangle[1], triangle[2]], [triangle[2], triangle[0]]]

def _is_convex_vertex(vertex_idx: int, vertex_cycle: [int], new_vertices: [np.array]):
    """
    :param vertex_idx: Index for the vertex within the cycle that you want to test for being a reflex vertex.
    :param vertex_cycle: List representing vertex indices with respect to `new_vertices` input. Assumed to be ordered in counter-clockwise order.
    :param new_vertices: List of 3-dimensional representation for vertices in the embedding
    :return: True if the desired vertex is a convex vertex with respect to the cycle
    """
    # Get the number of vertices in the cycle.
    num_cycle_vertices = len(vertex_cycle)

    # Compute vertex indices _before_ and _after_ `vertex_idx` in the cycle.
    vertex_before_idx = (vertex_idx + num_cycle_vertices - 1) % num_cycle_vertices
    vertex_after_idx = (vertex_idx + 1) % num_cycle_vertices

    # Grab all the vertices.
    v = new_vertices[vertex_cycle[vertex_idx]]
    v_b = new_vertices[vertex_cycle[vertex_before_idx]]
    v_a = new_vertices[vertex_cycle[vertex_after_idx]]

    # Use a mix of cross products and dot products to check for if v is a reflex vertex.
    #                     .─────────────.
    #                _.──'               `───.
    #             ,─'                         '─.
    #          ,─'                               '─.
    #        ,'   .─.                               `.
    #       ╱    (v_b)                                ╲
    #     ,'      `─'                                  `.
    #    ╱           ▲                                   ╲
    #   ╱             ╲                                   ╲
    #  ;               ╲                                   :
    #  ;                ╲                                  :
    # ;                  ╲                                  :
    # │                   ╲                                 │
    # │                    ╲                                │
    # │                     ╲                               │
    # │                      ╲ .─.                    .─.   │
    # :            ───────────( v )────────────────▶─(v_a)─────────▷
    #  :                       `─'                    `─'  ;
    #  :                                                   ;
    #   ╲                                                 ╱
    #    ╲                                               ╱
    #     ╲                                             ╱
    #      `.                                         ,'
    #        ╲                                       ╱
    #         `.                                   ,'
    #           '─.                             ,─'
    #              '─.                       ,─'
    #                 `───.             _.──'
    #                      `───────────'
    # v is a convex vertex wrt the cycle iff dot(v_b, cross(v, v_a)) > 0
    return np.dot(v_b, np.cross(v, v_a)) < 0.0

def _is_reflex_vertex(vertex_idx: int, vertex_cycle: [int], new_vertices: [np.array]):
    """
    :param vertex_idx: Index for the vertex within the cycle that you want to test for being a reflex vertex.
    :param vertex_cycle: List representing vertex indices with respect to `new_vertices` input. Assumed to be ordered in counter-clockwise order.
    :param new_vertices: List of 3-dimensional representation for vertices in the embedding
    :return: True if the desired vertex is a reflex vertex with respect to the cycle
    """
    # Get the number of vertices in the cycle.
    num_cycle_vertices = len(vertex_cycle)

    # Compute vertex indices _before_ and _after_ `vertex_idx` in the cycle.
    vertex_before_idx = (vertex_idx + num_cycle_vertices - 1) % num_cycle_vertices
    vertex_after_idx = (vertex_idx + 1) % num_cycle_vertices

    # Grab all the vertices.
    v = new_vertices[vertex_cycle[vertex_idx]]
    v_b = new_vertices[vertex_cycle[vertex_before_idx]]
    v_a = new_vertices[vertex_cycle[vertex_after_idx]]

    # Use a mix of cross products and dot products to check for if v is a reflex vertex.
    #                     .─────────────.
    #                _.──'               `───.
    #             ,─'                         '─.
    #          ,─'                               '─.
    #        ,'   .─.                               `.
    #       ╱    (v_b)                                ╲
    #     ,'      `─'                                  `.
    #    ╱           ▲                                   ╲
    #   ╱             ╲                                   ╲
    #  ;               ╲                                   :
    #  ;                ╲                                  :
    # ;                  ╲                                  :
    # │                   ╲                                 │
    # │                    ╲                                │
    # │                     ╲                               │
    # │                      ╲ .─.                    .─.   │
    # :            ───────────( v )────────────────▶─(v_a)─────────▷
    #  :                       `─'                    `─'  ;
    #  :                                                   ;
    #   ╲                                                 ╱
    #    ╲                                               ╱
    #     ╲                                             ╱
    #      `.                                         ,'
    #        ╲                                       ╱
    #         `.                                   ,'
    #           '─.                             ,─'
    #              '─.                       ,─'
    #                 `───.             _.──'
    #                      `───────────'
    # v is a reflex vertex wrt the cycle iff dot(v_b, cross(v, v_a)) > 0
    return np.dot(v_b, np.cross(v, v_a)) > 0.0


def _vertex_pair_are_short_geodesic(v1: np.array, v2: np.array):
    eps = 1e-4
    n1 = v1 / np.linalg.norm(v1)
    n2 = v2 / np.linalg.norm(v2)
    return np.linalg.norm(n1 - n2) + eps <= 2.0


def _triangle_not_degenerate(v1: np.array, v2: np.array, v3: np.array):
    eps = 1e-6
    A = np.zeros((3, 3))
    A[0, :] = v1
    A[1, :] = v2
    A[2, :] = v3
    volume = np.abs(np.linalg.det(A))
    return volume >= eps


def _build_cycle_with_directed_edges(edge_lst: [[int]], num_vertices: int):
    next_vertex = [None] * num_vertices
    prev_vertex = [None] * num_vertices
    for edge in edge_lst:
        next_vertex[edge[0]] = edge[1]
        prev_vertex[edge[1]] = edge[0]

    # Populate the cycle starting with the first vertex
    v = edge_lst[0][0]
    v_start = v
    cycle = [v]
    while next_vertex[v] != v_start:
        u = next_vertex[v]
        cycle.append(u)
        v = u

    # Reverse the cycle order so it is consistent with a counter-clockwise rotation
    cycle.reverse()

    # Return the cycle
    return cycle


def _all_faces_positive_volume(vertices: [np.array], triangles: [[int]]):
    all_positive = True
    for tri in triangles:
        A = np.zeros((3, 3))
        A[0, :] = vertices[tri[0]]
        A[1, :] = vertices[tri[1]]
        A[2, :] = vertices[tri[2]]
        volume = np.abs(np.linalg.det(A))
        if volume <= 0.0:
            all_positive = False
    return all_positive

def _form_top_and_bottom_cycles2(new_edge: [int], vertices: [np.array], triangles: [[int]]):
    nv = len(vertices)
    nf = len(triangles)
    faces_to_remove = []
    top_cycle = []
    bottom_cycle = []

    v_start = vertices[new_edge[0]]
    v_end = vertices[new_edge[1]]
    n = np.cross(v_start, v_end)
    n = n / np.linalg.norm(n)

    # obtain data structures for finding twins of edges and their corresponding triangles
    face_of_edge = dict()
    index_of_edge_in_parent_face = dict()
    for i in range(0, nf):
        tri = triangles[i]
        for j in range(0, 3):
            edge = (tri[j], tri[(j+1)%3])
            face_of_edge[edge] = i
            index_of_edge_in_parent_face[edge] = j

    # identify the triangle containing new_edge[0] and the edge opposite to this vertex
    sliced_face = None
    sliced_edge = (0, 0)
    for i in range(0, nf):
        tri = triangles[i]
        if new_edge[0] in tri:
            idx = -1
            for j in range(0, 3):
                if tri[j] == new_edge[0]:
                    idx = j
                    break
            assert idx >= 0
            v1_idx = tri[(idx+1)%3]
            v2_idx = tri[(idx+2)%3]
            if np.dot(vertices[v1_idx], n) < 0.0 < np.dot(vertices[v2_idx], n):
                faces_to_remove.append(i)
                sliced_face = tri
                sliced_edge = (v1_idx, v2_idx)
                break

    if sliced_face is None:
        raise Exception("No face found that has new_edge[0] as a vertex and has an edge sliced by new_edge")

    top_cycle.append(sliced_edge[1])
    bottom_cycle.append(sliced_edge[0])

    # given current triangle that is split by `new_edge`, identify next triangle split by edge as we work
    # from new_edge[0] to new_edge[1]. terminate after finding triangle that contains new_edge[1].
    complete: bool = False
    for i in range(0, nf):

        # identify next sliced triangle and next sliced edge
        twin_sliced_edge: (int, int) = (sliced_edge[1], sliced_edge[0])
        twin_sliced_tri_idx: int = face_of_edge[twin_sliced_edge]
        sliced_face: [int] = triangles[twin_sliced_tri_idx]
        local_idx: int = index_of_edge_in_parent_face[twin_sliced_edge]
        faces_to_remove.append(twin_sliced_tri_idx)

        # return early if our newly sliced face contains the east-most endpoint
        if new_edge[1] in sliced_face:
            top_cycle.append(new_edge[1])
            top_cycle.append(new_edge[0])
            bottom_cycle.reverse()
            bottom_cycle.append(new_edge[0])
            bottom_cycle.append(new_edge[1])
            complete = True
            break

        # given i is the local index of the vertex starting the sliced edge in the new sliced triangle,
        # if dot(vertices[new_sliced_triangle[i-1]], n) > 0,
        # then new sliced edge is [new_sliced_triangle[i], new_sliced_triangle[i+1]].
        # otherwise, new sliced edge is [new_sliced_triangle[i-1], new_sliced_triangle[i]].
        if np.dot(n, vertices[sliced_face[(local_idx+2)%3]]) > 0:
            sliced_edge = (sliced_face[(local_idx+1)%3], sliced_face[(local_idx+2)%3])
            top_cycle.append(sliced_edge[1])
        else:
            sliced_edge = (sliced_face[(local_idx + 2) % 3], sliced_face[local_idx])
            bottom_cycle.append(sliced_edge[0])

    # check for an error case that hopefully is never triggered
    if not complete:
        raise Exception("Something went wrong in identifying all faces sliced by the new edge.")

    # return the results
    return top_cycle, bottom_cycle, faces_to_remove



def _form_top_and_bottom_cycles(new_edge: [int], new_vertices: [np.array], initial_triangles: [[int]]):
    num_vertices = len(new_vertices)
    faces_to_remove = []

    # loop over all the triangles and identify pairs of triangles that share the same edge
    twin_map = dict()
    for i in range(0, len(initial_triangles)):
        tri = initial_triangles[i]
        for edge in _get_edge_ids_of_triangle(tri):
            key = tuple(sorted(edge))
            if key not in twin_map:
                twin_map[key] = [i]
            else:
                twin_map[key].append(i)

    # Forms the cycles that will need to be triangulated.
    # Assumes there do not exist any vertices that live on the great circle that `new_edge` lives on, aside from
    # new_edge[0] and new_edge[1]
    top_cycle = []
    bottom_cycle = []
    v = new_edge[0]

    # edge segment for new_edge
    ne_coords = [new_vertices[new_edge[0]], new_vertices[new_edge[1]]]

    # identify the first triangle that contains new_edge[0] and intersects the segment
    # from new_edge[0] to new_edge[1]
    t1 = -1
    e1 = -1
    for i in range(len(initial_triangles)):
        tri = initial_triangles[i]
        if v in tri:
            edges = _get_edge_ids_of_triangle(tri)
            for j in range(3):
                e = edges[j]
                if v not in e and _edges_cross2(new_edge, e, new_vertices) == 0:
                    t1 = i
                    e1 = e
                    break
        if t1 >= 0:
            break

    # Check that we do not hit an error condition
    if t1 == -1:
        for i in range(len(initial_triangles)):
            tri = initial_triangles[i]
            if v in tri:
                edges = _get_edge_ids_of_triangle(tri)
                for j in range(3):
                    e = edges[j]
                    if v not in e and _edges_cross2(new_edge, e, new_vertices) == 0:
                        t1 = i
                        e1 = e
                        break
            if t1 >= 0:
                break
        raise Exception("Error: could not find starting triangle incident to new_edge[0]")

    # add this first triangle as one that needs to be removed
    faces_to_remove.append(t1)

    # assign initial cycle
    for u in e1:
        if _point_above_great_circle_of_edge(new_vertices[u], ne_coords):
            top_cycle.append(u)
        else:
            bottom_cycle.append(u)

    # get the twin of t1
    for tri_idx in twin_map[tuple(sorted(e1))]:
        if tri_idx != t1:
            t1 = tri_idx
            break

    # mark this as the last edge that was crossed
    last_edge = e1

    # iterate finding new triangles that intersect the segment represented by new_edge
    # until we reach the triangle that has new_edge[1] as a vertex.
    while new_edge[1] not in initial_triangles[t1]:

        # identify the next edge getting crossed
        edges = _get_edge_ids_of_triangle(initial_triangles[t1])
        e1 = None
        for i in range(3):
            e = edges[i]
            if _edges_cross2(new_edge, e, new_vertices) == 0 and not (last_edge[0] in e and last_edge[1] in e):
                e1 = e
                break
        if e1 is None:
            raise Exception("Error: no edge in current triangle was found to cross the line segment being inserted")

        # add current triangle as one to be removed
        faces_to_remove.append(t1)

        # update cycles as necessary
        for u in e1:
            if _point_above_great_circle_of_edge(new_vertices[u], ne_coords):
                if top_cycle[len(top_cycle) - 1] != u:
                    top_cycle.append(u)
            else:
                if bottom_cycle[len(bottom_cycle) - 1] != u:
                    bottom_cycle.append(u)

        # identify the twin triangle
        for tri_idx in twin_map[tuple(sorted(e1))]:
            if tri_idx != t1:
                t1 = tri_idx
                break

        # update the last edge crossed
        last_edge = e1

    # remove the final triangle that contains new_edge[1]
    faces_to_remove.append(t1)

    # complete the top and bottom cycles
    top_cycle.append(new_edge[1])
    top_cycle.append(new_edge[0])
    bottom_cycle.reverse()
    bottom_cycle.append(new_edge[0])
    bottom_cycle.append(new_edge[1])

    # Return the resulting cycles and remaining faces
    return top_cycle, bottom_cycle, faces_to_remove


def _triangulate_cycles_with_new_edge_as_base(cycle: [int], new_vertices: [np.array]):
    # Assumes the cycle is in clockwise order
    new_faces = []

    # get number of vertices in the cycle
    num_cycle_vertices = len(cycle)

    # Assumes the base edge `new_edge` uses up the last two vertices listed in the cycle
    v_start = num_cycle_vertices - 2
    start_value = cycle[v_start]
    v_i = v_start - 1
    degenerate_tris = set()
    while len(cycle) > 3:

        # Indices of vertices to the left and right of v_i, wrt the counter-clockwise cycle order
        vl = (v_i + len(cycle) - 1) % len(cycle)
        vr = (v_i + 1) % len(cycle)

        # Compute some quantities to decide if the current vertex and its neighbors can form a triangle
        vertex_not_reflex = _is_convex_vertex(vertex_idx=v_i, vertex_cycle=cycle, new_vertices=new_vertices)
        edge_short = _vertex_pair_are_short_geodesic(v1=new_vertices[cycle[vr]], v2=new_vertices[cycle[vl]])
        triangle_not_degenerate = _triangle_not_degenerate(v1=new_vertices[cycle[vr]], v2=new_vertices[cycle[v_i]],
                                                           v3=new_vertices[cycle[vl]])

        # It has been seen that sometimes a triangle can form but when there's 4 vertices left, this can _force_ the
        # remaining 3 vertices to be a degenerate triangle. We check for this and if the remaining triangle would be
        # degenerate, we update the `triangle_not_degenerate` flag with the hope that other possible triangle will
        # avoid this situation.
        last_triangle_not_degenerate = True
        if len(cycle) == 4:
            tmp_cycle = [cycle[0], cycle[1], cycle[2], cycle[3]]
            tmp_cycle.pop(v_i)
            last_triangle_not_degenerate = _triangle_not_degenerate(v1=new_vertices[tmp_cycle[2]],
                                                                    v2=new_vertices[tmp_cycle[1]],
                                                                    v3=new_vertices[tmp_cycle[0]])
        triangle_not_degenerate = triangle_not_degenerate and last_triangle_not_degenerate
        new_tri = [cycle[vr], cycle[v_i], cycle[vl]]

        # This is book-keeping to break any infinite loop that could occur by allowing for a degenerate triangle if the
        # only way to triangulate things requires it
        seen_before = False
        if not triangle_not_degenerate:
            tri = tuple(new_tri)
            if tri in degenerate_tris:
                seen_before = True
            else:
                degenerate_tris.add(tri)

        # if v_i is not a reflex vertex and if the neighbors of v_i can form a short geodesic, and if the produced
        # triangle is not degenerate, then form a triangle with its two neighbors and remove it from the cycle.
        # Note: if the triangle has been passed over before as being degenerate, we assume we need to accept it
        # otherwise we end up in an infinite loop.
        if vertex_not_reflex and edge_short and (triangle_not_degenerate or seen_before):
            new_faces.append(new_tri)
            cycle.pop(v_i)
            if cycle[v_i] == start_value:
                v_i -= 1

        else:  # Otherwise, decrement v_i's index by 1
            v_i -= 1

            # For simplicity, we want to maintain the `new_edge` vertices always being present in the cycle,
            # so if v_i < 0, this implies we are considering removal of one of these vertices.
            # We reset the index to the vertex just before the new edge vertices in the cycle's order.
            if v_i < 0:
                v_i = len(cycle) - 3

    # if we are down to 3 vertices, add this as the remaining triangle
    if len(cycle) == 3:
        new_faces.append([cycle[2], cycle[1], cycle[0]])

    # return all the found triangle faces
    return new_faces


def _update_embedding_with_disconnected_vertices_removed(vertices: [np.array], triangle_faces: [[int]]):
    """
    :param vertices: initial vertices for embedding
    :param triangle_faces: initial triangle faces for embedding
    :return: (new_vertices, triangle_faces) such that `new_vertices` removes all vertices from `vertices` that are no
    longer being used, if any such vertices exist. If there do exist such vertices, then `triangle_faces` are updated
    with a renumbering of surviving vertices.
    """
    num_vertices = len(vertices)
    new_vertex_ids = [None] * num_vertices

    # Loop over vertices present in the faces and assign a unique ID to them, which will be their new ID in the
    # modified embedding.
    id: int = 0
    for face in triangle_faces:
        for v in face:
            if new_vertex_ids[v] is None:
                new_vertex_ids[v] = id
                id += 1

    # if no vertices need to be removed then just return the result as-is
    if id == num_vertices:
        return vertices, triangle_faces

    # Loop over triangle faces and update the IDs for each vertex that was assigned a new vertex.
    for i in range(0, len(triangle_faces)):
        triangle_faces[i][0] = new_vertex_ids[triangle_faces[i][0]]
        triangle_faces[i][1] = new_vertex_ids[triangle_faces[i][1]]
        triangle_faces[i][2] = new_vertex_ids[triangle_faces[i][2]]

    # Build the new list of vertices to only contain vertices that were assigned a new ID.
    num_new_vertices = id
    new_vertices = [None] * num_new_vertices
    for i in range(0, num_vertices):
        if new_vertex_ids[i] is not None:
            new_vertices[new_vertex_ids[i]] = vertices[i]

    # Return the new embedding.
    return new_vertices, triangle_faces


def _point_is_in_triangle(p: np.array, triangle: [int], vertices: [np.array]):
    edges = _get_edge_coordinates_of_triangle(triangle, vertices)
    is_in_tri = True
    for edge in edges:
        p0 = edge[0]# - p
        p1 = edge[1]# - p
        is_in_tri = is_in_tri and np.dot(p, np.cross(p0, p1)) > 0
    return is_in_tri


def _insert_edge_and_triangulate_existing_vertices(initial_vertices: [np.array], initial_triangles: [[int]],
                                                   new_edge: [int]):
    # check if the desired edge already exists
    for tri in initial_triangles:
        if new_edge[0] in tri and new_edge[1] in tri:
            return initial_vertices, initial_triangles

    # if there are any vertices that lie on the new edge, we will delete them and
    ne_coords = [initial_vertices[new_edge[0]], initial_vertices[new_edge[1]]]
    for i in range(len(initial_vertices)):
        if i not in new_edge and _point_in_geodesic(initial_vertices[i], ne_coords):
            raise Exception("Error: a vertex that is not the vertices of `new_edge` lies on the geodesic of `new_edge`")

    # construct the top and bottom cycles and identify faces that will be removed.
    top_cycle, bottom_cycle, faces_to_remove = _form_top_and_bottom_cycles2(new_edge, initial_vertices,
                                                                           initial_triangles)

    # Initialize the new vertices and triangles that will be returned
    new_vertices = []
    for v in initial_vertices:
        new_vertices.append(v)

    new_faces = []
    for face_index in range(0, len(initial_triangles)):
        if face_index not in faces_to_remove:
            new_faces.append(initial_triangles[face_index])

    # triangulate the weakly simple cycles
    top_new_triangles = _triangulate_cycles_with_new_edge_as_base(top_cycle, new_vertices)
    bottom_new_triangles = _triangulate_cycles_with_new_edge_as_base(bottom_cycle, new_vertices)

    # Add the new faces to the embedding
    for tri in top_new_triangles:
        new_faces.append(tri)
    for tri in bottom_new_triangles:
        new_faces.append(tri)

    # Check that all triangles have correct volume
    if not _all_faces_positive_volume(new_vertices, new_faces):
        print("Not all produces triangles have positive orientation!")

    # Return the new vertices and faces of the embedding
    return new_vertices, new_faces


def _triangulate_point(initial_vertices: [np.array], initial_triangles: [[int]], point: np.ndarray):
    new_vertices = initial_vertices
    new_triangles = []
    new_point_idx = len(new_vertices)

    # loop through and find the triangle that contains this point
    never_found = True
    for i in range(len(initial_triangles)):
        tri = initial_triangles[i]
        if _point_is_in_triangle(point, tri, initial_vertices):
            if not never_found:
                print("already found a point? Let's investigate")
                _point_is_in_triangle(point, tri, initial_vertices)
            new_triangles.append([tri[0], tri[1], new_point_idx])
            new_triangles.append([tri[1], tri[2], new_point_idx])
            new_triangles.append([tri[2], tri[0], new_point_idx])
            never_found = False
        else:
            new_triangles.append(tri)

    if never_found:
        for i in range(len(initial_triangles)):
            tri = initial_triangles[i]
            if _point_is_in_triangle(point, tri, initial_vertices):
                new_triangles.append([tri[0], tri[1], new_point_idx])
                new_triangles.append([tri[1], tri[2], new_point_idx])
                new_triangles.append([tri[2], tri[0], new_point_idx])
                never_found = False
            else:
                new_triangles.append(tri)

    # add the new vertex
    new_vertices.append(point)

    # return the new vertices and triangles
    return new_vertices, new_triangles


def _triangulate_if_intersect_edge(initial_vertices: [np.array], initial_triangles: [[int]], point: np.ndarray):
    new_vertices = initial_vertices
    new_triangles = []
    new_point_idx = len(new_vertices)
    did_intersect = False
    for tri in initial_triangles:
        edges = _get_edge_coordinates_of_triangle(tri, initial_vertices)
        edges_ids = _get_edge_ids_of_triangle(tri)
        intersect_edge = -1
        for i in range(3):
            e = edges[i]
            if _point_in_geodesic(point, e):
                intersect_edge = i
        if intersect_edge >= 0:
            did_intersect = True
            for i in range(3):
                e = edges_ids[i]
                if i != intersect_edge:
                    new_triangles.append([e[0], e[1], new_point_idx])
        else:
            new_triangles.append(tri)

    if did_intersect:
        new_vertices.append(point)

    return did_intersect, new_vertices, new_triangles

def check_for_existing_point_to_snap_to(point: np.array, point_list: [np.array], epsilon=1e-5):
    for i in range(len(point_list)):
        if np.linalg.norm(point - point_list[i]) < epsilon:
            return i
    return None

def insert_edge_and_triangulate(initial_vertices: [np.array], initial_triangles: [[int]], new_edge):
    """
    :param initial_vertices: List of n vertices in 3D, assumed to be coordinates on the unit sphere
    :param initial_triangles: List of all triangle faces in the embedding, where each face is represented by a list of
    vertex IDs with the vertices ordered so the triangle has an outward normal (relative to the sphere center).
    Assumes that all edges correspond to geodesics with length smaller than a half great circle.
    :param new_edge: List of two vertices, each either represented with a 3-d vector or vertex ID.
    Assumes that all edges correspond to geodesics with length smaller than a half great circle.
    :return: Return tuple (new_vertices: [np.array], initial_triangles: [[int]]) representing the new embedding, where
    all faces should have vertices ordered in counter-clockwise order so they have outward normals.

    Note: "top" and "bottom" are relative to the (normalized) direction n = cross(new_edge[0], new_edge[1]) being viewed
    as the z direction, even though the code does not rotate the sphere so that this is true.
    """
    vertices = []
    triangles = []
    for i in range(len(initial_vertices)):
        vertices.append(initial_vertices[i])
    for i in range(len(initial_triangles)):
        triangles.append(initial_triangles[i])

    # Check that the inputs satisfy expectations
    _input_verification(vertices, triangles, new_edge)

    # if the new edge is comprised of vertices that are new, triangulate any triangles they are in and then proceed with
    # the triangulation with them as existing vertices in the embedding.
    tmp_vertices = vertices
    tmp_triangles = triangles
    edge_ids = [0, 0]
    info1 = None
    if type(new_edge[0]) is np.ndarray:
        info1 = check_for_existing_point_to_snap_to(new_edge[0], vertices)
        if info1 is None:
            did_intersect, tmp_vertices, tmp_triangles = _triangulate_if_intersect_edge(tmp_vertices, tmp_triangles,
                                                                                        new_edge[0])
            if not did_intersect:
                tmp_vertices, tmp_triangles = _triangulate_point(tmp_vertices, tmp_triangles, new_edge[0])
            edge_ids[0] = len(tmp_vertices) - 1
        else:
            edge_ids[0] = info1
    else:
        edge_ids[0] = new_edge[0]

    info2 = None
    if type(new_edge[1]) is np.ndarray:
        info2 = check_for_existing_point_to_snap_to(new_edge[1], vertices)
        if info2 is None:
            did_intersect, tmp_vertices, tmp_triangles = _triangulate_if_intersect_edge(tmp_vertices, tmp_triangles,
                                                                                        new_edge[1])
            if not did_intersect:
                tmp_vertices, tmp_triangles = _triangulate_point(tmp_vertices, tmp_triangles, new_edge[1])
            edge_ids[1] = len(tmp_vertices) - 1
        else:
            edge_ids[1] = info2
    else:
        edge_ids[1] = new_edge[1]

    # triangulate the result by using the newly created but existing vertices within the embedding.
    return _insert_edge_and_triangulate_existing_vertices(initial_vertices=tmp_vertices,
                                                          initial_triangles=tmp_triangles, new_edge=edge_ids)


def _identify_dependency_arc(vertex_id: int, face: [int]):
    arc = [0, 0]
    for i in range(0, len(face)):
        e_i = [face[i], face[(i + 1) % len(face)]]
        if e_i[0] == vertex_id:
            arc[1] = e_i[1]
        if e_i[1] == vertex_id:
            arc[0] = e_i[0]
    return arc


def convert_repr_to_adj_list(vertices: [np.array], faces: [[int]]):
    # Assumes face vertices are oriented clockwise on the surface of the sphere, so has outward normals
    num_vertices = len(vertices)
    adj_list: [[int]] = []
    adj_dep: [[[int]]] = []

    for i in range(0, num_vertices):
        adj_list.append([])
        adj_dep.append([])

    for face in faces:
        for i in range(0, len(face)):
            adj_dep[face[i]].append(_identify_dependency_arc(face[i], face))

    # Construct the adjacency lists
    for v in range(0, num_vertices):
        adj_list[v] = _build_cycle_with_directed_edges(edge_lst=adj_dep[v], num_vertices=num_vertices)

    # return the vertices and adjacency list representation
    return vertices, adj_list


def test_edges_cross(e1: [np.array], e2: [np.array]):
    return _edges_cross(e1, e2)
