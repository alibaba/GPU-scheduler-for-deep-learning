# Description:
# A library to do dynamic resource management.

package(default_visibility = [
    "//tensorflow:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cuda_library",
    "tf_cc_test",
    "tf_copts",
    "tf_cuda_cc_test",
)

# For platform specific build config
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")

cc_library(
    name = "file_listener",
    srcs = ["file_listener.cc"],
    hdrs = ["file_listener.h"],
    copts = ["-I/usr/include/"],
    linkopts = ["-L/usr/lib64/", "-lcrypto"],
    deps = [
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "file_listener_test",
    size = "small",
    srcs = ["file_listener_test.cc"],
    deps = [
        ":file_listener",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
