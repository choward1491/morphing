/*
** Created by Christian Howard on 4/7/23.
*/
#include "primitives.h"
#include <cassert>

namespace morphing {

bool Vertex::operator!=(const Vertex &v) const {
  return v.id != id;
}
bool Vertex::operator==(const Vertex &v) const {
  return v.id == id;
}
bool Vertex::operator<(const Vertex &v) const {
  return (v.id < id) or (v.type < type);
}

Dart::Dart(Vertex *head, Vertex *tail):dual_dart_(nullptr), twin_(nullptr) {
  set_head(head);
  set_tail(tail);
}
void Dart::set_head(Vertex *head) {
  head_ = head;
}
void Dart::set_tail(Vertex *tail) {
  tail_ = tail;
}
void Dart::set_twin(Dart *twin) {
  twin_ = twin;
}
void Dart::set_dual_dart(Dart *dual) {
  dual_dart_ = dual;
}
Dart *Dart::dual() const {
  return dual_dart_;
}
Dart *Dart::rev() const {
  return twin_;
}
Dart *Dart::twin() const {
  return twin_;
}
std::pair<id_type, id_type> Dart::vertex_ids() const {
  using id_pair_t = std::pair<id_type, id_type>;
  return id_pair_t(tail()->id, head()->id);
}
bool Dart::operator!=(const Dart &d) const {
  return d.head() != head_ or d.tail() != tail_ or d.rev() != twin_ or d.dual() != dual_dart_;
}
bool Dart::operator==(const Dart &d) const {
  return not operator!=(d);
}
bool Dart::operator<(const Dart &d) const {
  return (head_->id < d.head()->id) or (head_->id == d.head()->id and tail_->id < d.tail()->id);
}
Vertex *Dart::head() const {
  return head_;
}
Vertex *Dart::tail() const {
  return tail_;
}
}
