/*
** Created by Christian Howard on 4/7/23.
*/
#ifndef MORPHING__PRIMITIVES_H_
#define MORPHING__PRIMITIVES_H_

#include <vector>
#include <cstdint>

#include <armadillo>

namespace morphing {

  using id_type = std::uint64_t;

  class Dart;

  enum class VertexType: int { Primal = 0, Dual};
  struct Vertex {

    // data
    VertexType type;
    id_type id;
    arma::vec3 pos;
    std::vector<Dart*> face;

    // operators for data structures
    bool operator!=(const Vertex& v) const;
    bool operator==(const Vertex& v) const;
    bool operator<(const Vertex& v) const;

  };

  class Dart {
   public:

    // constructor
    explicit Dart(Vertex* head = nullptr, Vertex* tail = nullptr);
    ~Dart() = default;

    // setters/getters
    void set_head(Vertex *head);
    void set_tail(Vertex *tail);
    void set_twin(Dart* twin);
    void set_dual_dart(Dart* dual);

    [[nodiscard]] Vertex* head() const;
    [[nodiscard]] Vertex* tail() const;
    [[nodiscard]] Dart* dual() const;
    [[nodiscard]] Dart* rev() const;
    [[nodiscard]] Dart* twin() const;
    [[nodiscard]] std::pair<id_type, id_type> vertex_ids() const;

    // operators for data structures
    bool operator!=(const Dart& d) const;
    bool operator==(const Dart& d) const;
    bool operator<(const Dart& d) const;

   private:
    Vertex *head_;
    Vertex *tail_;
    Dart* twin_;
    Dart* dual_dart_;
  };

}

#endif //MORPHING__PRIMITIVES_H_
