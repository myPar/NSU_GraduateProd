cmake_minimum_required(VERSION 3.9)

# common options

enable_language(CXX)

# set(CMAKE_C_COMPILER   "/usr/bin/clang")
# set(CMAKE_CXX_COMPILER "/usr/bin/clang++")

set(CMAKE_CXX_STANDARD 17)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
      -W -Wall -Wextra -Wpedantic \
      -Waggressive-loop-optimizations \
      -Wcast-align -Wcast-qual \
      -Wdouble-promotion -Wduplicated-branches -Wduplicated-cond \
      -Wfloat-equal -Wformat=2 -Wformat-signedness -Wframe-larger-than=32768 \
      -Wlogical-op \
      -Wnull-dereference \
      -Wodr -Wold-style-cast \
      -Wshadow=local -Wshift-overflow=2 -Wstrict-aliasing=2 -Wsuggest-final-methods -Wsuggest-final-types -Wsync-nand \
      -Wtrampolines \
      -Wuseless-cast -Wno-unused-but-set-parameter \
      -Wwrite-strings"
      CACHE INTERNAL ""
  )
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
      -Weverything \
      -Wno-class-varargs -Wno-padded \
      -Wc++17-compat \
      -Wno-c++98-compat -Wno-c++98-compat-pedantic"
      # -Wno-c++14-extensions
      # -Wno-c++17-extensions
      # -Wno-switch-enum -Wno-unused-macros
      CACHE INTERNAL ""
  )
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g"
    CACHE INTERNAL ""
)

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} \
    -g -pg -no-pie \
    -O3 -ffast-math -fno-rtti"
    CACHE INTERNAL ""
)
set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO
    "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} -pg"
    CACHE INTERNAL ""
)

# rolled loops works faster here
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} \
    -O3 -ffast-math -fno-rtti"
    CACHE INTERNAL ""
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -Os"
      CACHE INTERNAL "")
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -Oz"
      CACHE INTERNAL "")
endif()

set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} \
    -O3 -ffast-math -fno-math-errno -fno-unroll-loops \
    -fno-rtti -fno-stack-protector -fno-ident \
    -fomit-frame-pointer -ffunction-sections -fdata-sections \
    -fmerge-all-constants"
    CACHE INTERNAL ""
)

# project configuration

# project(1D_3L_chart_test)

# include_directories(1D_3L_chart_test PRIVATE "./include")

# add_executable(1D_3L_chart_test "")
# target_sources(1D_3L_chart_test PRIVATE
#     keras2cpp/utils.cc
#     keras2cpp/baseLayer.cc
#     keras2cpp/layers/activation.cc
#     keras2cpp/layers/conv1d.cc
#     keras2cpp/layers/conv2d.cc
#     keras2cpp/layers/dense.cc
#     keras2cpp/layers/elu.cc
#     keras2cpp/layers/leaky_relu.cpp
#     keras2cpp/layers/embedding.cc
#     keras2cpp/layers/flatten.cc
#     keras2cpp/layers/lstm.cc
#     keras2cpp/layers/locally1d.cc
#     keras2cpp/layers/locally2d.cc
#     keras2cpp/layers/maxPooling2d.cc
#     keras2cpp/layers/batchNormalization.cc
#     keras2cpp/model.cc
#     keras2cpp/tensor.cc
#     1D_3L_chart/model_inference.cpp
#     1D_3L_chart_test.cpp
# )

# execution time benchmark project:
project(1D_3L_chart_benchmark)

include_directories(1D_3L_chart_benchmark PRIVATE "./include")

add_executable(1D_3L_chart_benchmark "")
target_sources(1D_3L_chart_benchmark PRIVATE
    keras2cpp/utils.cc
    keras2cpp/baseLayer.cc
    keras2cpp/layers/activation.cc
    keras2cpp/layers/conv1d.cc
    keras2cpp/layers/conv2d.cc
    keras2cpp/layers/dense.cc
    keras2cpp/layers/elu.cc
    keras2cpp/layers/leaky_relu.cpp
    keras2cpp/layers/embedding.cc
    keras2cpp/layers/flatten.cc
    keras2cpp/layers/lstm.cc
    keras2cpp/layers/locally1d.cc
    keras2cpp/layers/locally2d.cc
    keras2cpp/layers/maxPooling2d.cc
    keras2cpp/layers/batchNormalization.cc
    keras2cpp/model.cc
    keras2cpp/tensor.cc
    1D_3L_chart/model_inference.cpp
    test_inference_time.cpp
)
