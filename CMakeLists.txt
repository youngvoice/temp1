cmake_minimum_required(VERSION 3.14)
project(reranker1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ONNX Runtime
# Users can specify the ONNX Runtime installation path via ONNXRUNTIME_ROOT
if(NOT DEFINED ONNXRUNTIME_ROOT)
    set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime installation")
endif()

# Try to find ONNX Runtime headers and libraries
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS
        ${ONNXRUNTIME_ROOT}/include
        /usr/local/include
        /usr/include
        /usr/local/include/onnxruntime
        /usr/include/onnxruntime
)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS
        ${ONNXRUNTIME_ROOT}/lib
        /usr/local/lib
        /usr/lib
        /usr/lib/x86_64-linux-gnu
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR
        "ONNX Runtime not found. Please install it or set ONNXRUNTIME_ROOT.\n"
        "Example: cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime ..\n"
        "Download ONNX Runtime from: https://github.com/microsoft/onnxruntime/releases"
    )
endif()

message(STATUS "Found ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "Found ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")

# Create the executable
add_executable(reranker1 reranker1.cpp)

target_include_directories(reranker1 PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(reranker1 PRIVATE ${ONNXRUNTIME_LIBRARY})

# Platform-specific settings
if(UNIX AND NOT APPLE)
    target_link_libraries(reranker1 PRIVATE pthread)
endif()

# Installation
install(TARGETS reranker1 DESTINATION bin)
