from .vertex import Vertex


# general dart
class Dart:
    def __init__(self, tail: Vertex, head: Vertex):
        self.tail = tail
        self.head = head
        self.twin = None
        self.dual_edge = None

    def __hash__(self):
        return hash((self.tail.id, self.head.id))

    def __eq__(self, e):
        return (self.tail == e.tail) and (self.head == e.head)

    def __ne__(self, e):
        return not self.__eq__(e)

    def __repr__(self):
        return "({0}->{1})".format(self.tail.id, self.head.id)

    def rev(self):
        # get the reversal edge
        return self.twin

    def rev_idx(self):
        # get the reversal vertex indices
        return self.tail.id, self.head.id

    def dual(self):
        # get the dual edge
        return self.dual_edge

    def set_twin(self, twin):
        self.twin = twin

    def set_dual_edge(self, dual_e):
        self.dual_edge = dual_e
