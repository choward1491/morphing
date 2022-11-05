# class representing data that might be stored in our vertices for this problem
class VertexData:
    def __init__(self, position=None, face=None):
        self.face = face
        self.pos = position

    # get all vertex IDs on the boundary of the face
    # and return them as a set
    def get_vertex_id_set(self):
        vid_set = set()
        for d in self.face:
            vid_set.add(d.tail.id)
            vid_set.add(d.head.id)
        return vid_set


# general vertex class with the ability to add
# auxiliary data for the vertex to store
class Vertex:
    def __init__(self, id, aux_data=None):
        self.id = id
        self.aux_data = aux_data

    def __hash__(self):
        return hash(id)

    def __eq__(self, u):
        return self.id == u.id

    def __ne__(self, u):
        return not self.__eq__(u)
