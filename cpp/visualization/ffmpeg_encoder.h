#ifndef SPH_MORPH_VISUALIZATION_FFMPEG_ENCODER_H_
#define SPH_MORPH_VISUALIZATION_FFMPEG_ENCODER_H_

#include <string>
#include <cstdio>
#include <vector>

namespace morphing {
namespace visualization {

class FFmpegEncoder {
 public:
  FFmpegEncoder() = default;
  ~FFmpegEncoder();

  bool Start(const std::string& output_path, int width, int height, int fps);
  bool WriteFrame(const std::vector<uint8_t>& rgb_buffer);
  void Stop();

 private:
  std::FILE* ffmpeg_pipe_ = nullptr;
  int width_ = 0;
  int height_ = 0;
};

} // namespace visualization
} // namespace morphing

#endif // SPH_MORPH_VISUALIZATION_FFMPEG_ENCODER_H_
