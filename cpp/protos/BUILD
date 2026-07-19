load("@rules_proto//proto:defs.bzl", "proto_library")
load("@protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")

proto_library(
    name = "geometry_proto",
    srcs = ["geometry.proto"],
    visibility = ["//visibility:public"],
)

cc_proto_library(
    name = "geometry_cc_proto",
    deps = [":geometry_proto"],
    visibility = ["//visibility:public"],
)

proto_library(
    name = "morph_sequence_proto",
    srcs = ["morph_sequence.proto"],
    deps = [":geometry_proto"],
    visibility = ["//visibility:public"],
)

cc_proto_library(
    name = "morph_sequence_cc_proto",
    deps = [":morph_sequence_proto"],
    visibility = ["//visibility:public"],
)
