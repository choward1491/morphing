load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cc_library(
    name = "primitives",
    srcs = ["primitives.cc"],
    hdrs = ["primitives.h"],
    deps = [
        "//protos:geometry_cc_proto",
        "@boost//:multiprecision",
        "@eigen",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "dcel",
    srcs = ["dcel.cc"],
    hdrs = ["dcel.h"],
    deps = [
        ":primitives",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "linear_program",
    srcs = ["linear_program.cc"],
    hdrs = ["linear_program.h"],
    deps = [
        ":primitives",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "pseudomorph_builder",
    srcs = ["pseudomorph_builder.cc"],
    hdrs = ["pseudomorph_builder.h"],
    deps = [
        ":dcel",
        ":linear_program",
        "//protos:morph_sequence_cc_proto",
        "@abseil-cpp//absl/status",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "morph_compiler",
    srcs = ["morph_compiler.cc"],
    hdrs = ["morph_compiler.h"],
    deps = [
        ":dcel",
        ":pseudomorph_builder",
        "//protos:morph_sequence_cc_proto",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "dcel_test",
    srcs = ["dcel_test.cc"],
    deps = [
        ":dcel",
        "@googletest//:gtest_main",
    ],
)
