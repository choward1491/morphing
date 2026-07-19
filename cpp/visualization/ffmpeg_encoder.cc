#include "visualization/ffmpeg_encoder.h"
#include <iostream>
#include <sstream>

namespace morphing {
namespace visualization {

FFmpegEncoder::~FFmpegEncoder() {
  Stop();
}

bool FFmpegEncoder::Start(const std::string& output_path, int width, int height, int fps) {
  width_ = width;
  height_ = height;

  std::stringstream cmd;
  cmd << "ffmpeg -y -f rawvideo -pix_fmt rgb24 -s " << width << "x" << height
      << " -r " << fps << " -i - -an -vcodec libx264 -pix_fmt yuv420p " << output_path;

  std::cout << "[FFmpegEncoder] Running command: " << cmd.str() << std::endl;
  ffmpeg_pipe_ = popen(cmd.str().c_str(), "w");
  if (!ffmpeg_pipe_) {
    std::cerr << "[FFmpegEncoder] Failed to open FFmpeg pipe!" << std::endl;
    return false;
  }
  return true;
}

bool FFmpegEncoder::WriteFrame(const std::vector<uint8_t>& rgb_buffer) {
  if (!ffmpeg_pipe_) {
    return false;
  }
  size_t expected_size = static_cast<size_t>(width_ * height_ * 3);
  if (rgb_buffer.size() < expected_size) {
    std::cerr << "[FFmpegEncoder] Buffer size is too small!" << std::endl;
    return false;
  }
  size_t written = std::fwrite(rgb_buffer.data(), 1, expected_size, ffmpeg_pipe_);
  return written == expected_size;
}

void FFmpegEncoder::Stop() {
  if (ffmpeg_pipe_) {
    std::fflush(ffmpeg_pipe_);
    pclose(ffmpeg_pipe_);
    ffmpeg_pipe_ = nullptr;
    std::cout << "[FFmpegEncoder] FFmpeg pipe closed." << std::endl;
  }
}

} // namespace visualization
} // namespace morphing
