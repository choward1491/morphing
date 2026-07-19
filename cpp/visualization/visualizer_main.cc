#include <iostream>
#include <vector>
#include "visualization/metal_renderer.h"
#include "visualization/ffmpeg_encoder.h"

int main(int argc, char* argv[]) {
  std::cout << "========================================" << std::endl;
  std::cout << "Spherical Graph Morph Visualizer" << std::endl;
  std::cout << "========================================" << std::endl;

  // Instantiate Metal Renderer
  morphing::visualization::MetalRenderer renderer;
  renderer.Initialize();

  // Test FFmpeg encoder pipe functionality
  morphing::visualization::FFmpegEncoder encoder;
  const std::string test_output = "morph_output.mp4";
  int width = 800;
  int height = 600;
  int fps = 30;

  if (encoder.Start(test_output, width, height, fps)) {
    std::cout << "Headless rendering started. Saving to " << test_output << std::endl;
    
    // Generate 60 mock frames (2 seconds)
    std::vector<uint8_t> dummy_frame(width * height * 3, 128); // medium gray background
    for (int i = 0; i < 60; ++i) {
      // Modify buffer slightly to simulate visual movement
      dummy_frame[i * 100 % dummy_frame.size()] = 255; 
      encoder.WriteFrame(dummy_frame);
    }
    encoder.Stop();
    std::cout << "Mock animation exported successfully to " << test_output << std::endl;
  }

  renderer.Shutdown();
  return 0;
}
