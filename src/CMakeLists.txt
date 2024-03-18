cmake_minimum_required(VERSION 3.0)

# Define project name
project(drell_yan_ang_coef CXX)

# Find ROOT package
find_package(ROOT REQUIRED COMPONENTS RIO Hist Tree)

# Set include directories
include_directories(${ROOT_INCLUDE_DIRS})

# Set source files
set(SOURCES
    reweighting_plots.cc
    reweighting_mc.cc
)

# Add a shared library target
add_library(drell_yan_ang_coef SHARED ${SOURCES})

# Link against ROOT libraries
target_link_libraries(drell_yan_ang_coef ${ROOT_LIBRARIES})

# Generate dictionary for ROOT classes
ROOT_GENERATE_DICTIONARY(dic_drell_yan_ang_coef ${CMAKE_CURRENT_SOURCE_DIR}/reweighting_plots.hh ${CMAKE_CURRENT_SOURCE_DIR}/reweighting_mc.hh LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/LinkDef.hh)

# Include directory with generated dictionary
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Link dictionary object file to library
target_sources(drell_yan_ang_coef PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/dic_drell_yan_ang_coef.cxx)

# Set output directory for library
set_target_properties(drell_yan_ang_coef PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
