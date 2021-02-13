cmake_minimum_required(VERSION 3.10)

project("TCLDetection")

# Find Required Libraries
find_package( HDF5 REQUIRED )
find_package( OpenCV REQUIRED )

# Set Include Directories
include_directories(${OpenCV_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})

# Gather the source files in one variable
set( SOURCES main.cpp TCLManager.cpp featureExtractor.cpp BSIFFilter.cpp )

# Define the executable
add_executable(tclDetect ${SOURCES})

# Set target options
target_compile_options(tclDetect PRIVATE ${HDF5_DEFINITIONS})
target_link_libraries(tclDetect ${OpenCV_LIBS} ${HDF5_HL_LIBRARIES} ${HDF5_LIBRARIES})
