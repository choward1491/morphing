cmake_minimum_required(VERSION 3.25)
project(morphing)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(matplotplusplus
                     GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
                     GIT_TAG origin/master)
FetchContent_GetProperties(matplotplusplus)
if(NOT matplotplusplus_POPULATED)
    FetchContent_Populate(matplotplusplus)
    add_subdirectory(${matplotplusplus_SOURCE_DIR} ${matplotplusplus_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
find_package(Armadillo)
find_library(GLPK glpk)

add_library(sphere_morph polyhedron.cpp polyhedron.h sphere_embedding.cpp sphere_embedding.h primitives.cpp primitives.h linear_program.hpp linear_program.cpp morph_visualizer.cpp morph_visualizer.h morph_step.h longitudinal_morph_step.cpp longitudinal_morph_step.h glp3r_morph_step.cpp glp3r_morph_step.h hemisphere_morph_step.cpp hemisphere_morph_step.h morph.cpp morph.h)
target_include_directories(sphere_morph PUBLIC /usr/local/include ${ARMADILLO_INCLUDE_DIRS})
if (${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    find_library(LIBBLAS Accelerate)
    target_link_libraries(sphere_morph PUBLIC matplot ${GLPK} ${ARMADILLO_LIBRARIES} ${LIBBLAS})
endif ()

add_executable(visualizer main.cpp)
target_link_libraries(visualizer PUBLIC sphere_morph)