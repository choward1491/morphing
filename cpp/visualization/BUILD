load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary")

cc_library(
    name = "renderer",
    srcs = [
        "renderer.cc",
        "metal_renderer.mm",
    ],
    hdrs = [
        "renderer.h",
        "metal_renderer.h",
    ],
    copts = ["-std=c++17"],
    linkopts = [
        "-framework Metal",
        "-framework Cocoa",
        "-framework QuartzCore",
    ],
    deps = [
        "//protos:morph_sequence_cc_proto",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ffmpeg_encoder",
    srcs = ["ffmpeg_encoder.cc"],
    hdrs = ["ffmpeg_encoder.h"],
    copts = ["-std=c++17"],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "morph_visualizer",
    srcs = ["visualizer_main.cc"],
    copts = ["-std=c++17"],
    deps = [
        ":renderer",
        ":ffmpeg_encoder",
    ],
)
