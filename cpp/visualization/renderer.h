#ifndef SPH_MORPH_VISUALIZATION_RENDERER_H_
#define SPH_MORPH_VISUALIZATION_RENDERER_H_

#include "protos/morph_sequence.pb.h"

namespace morphing {
namespace visualization {

struct RenderConfig {
  bool draw_spherical_grid = true;
  float vertex_size = 5.0f;
  float edge_width = 1.5f;
};

class Renderer {
 public:
  virtual ~Renderer() = default;

  virtual void Initialize() = 0;
  virtual void Clear() = 0;
  virtual void RenderFrame(const protos::MorphKeyframeProto& keyframe, const RenderConfig& config) = 0;
  virtual void Shutdown() = 0;
};

} // namespace visualization
} // namespace morphing

#endif // SPH_MORPH_VISUALIZATION_RENDERER_H_
