load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cc_library(
    name = "spherical-pseudomorph",
    srcs = ["spherical-pseudomorph.cc"],
    hdrs = ["spherical-pseudomorph.h"],
    deps = [
        "//protos:geometry_cc_proto",
        "//protos:morph_sequence_cc_proto",
        "@boost//:multiprecision",
        "@eigen",
    ],
    visibility = ["//visibility:public"],
)

