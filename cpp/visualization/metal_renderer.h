#ifndef SPH_MORPH_VISUALIZATION_METAL_RENDERER_H_
#define SPH_MORPH_VISUALIZATION_METAL_RENDERER_H_

#include "visualization/renderer.h"

namespace morphing {
namespace visualization {

class MetalRenderer : public Renderer {
 public:
  MetalRenderer() = default;
  ~MetalRenderer() override = default;

  void Initialize() override;
  void Clear() override;
  void RenderFrame(const protos::MorphKeyframeProto& keyframe, const RenderConfig& config) override;
  void Shutdown() override;
};

} // namespace visualization
} // namespace morphing

#endif // SPH_MORPH_VISUALIZATION_METAL_RENDERER_H_
