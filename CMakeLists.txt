cmake_minimum_required(VERSION 3.28)
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architecture")

project(GraphFEX LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/libtorch")

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")

find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

pybind11_add_module(GraphFEX
    src/bindings.cpp
)
target_include_directories(GraphFEX PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(GraphFEX PRIVATE "${TORCH_LIBRARIES}")
target_compile_options(GraphFEX PRIVATE ${TORCH_CXX_FLAGS})