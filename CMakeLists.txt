cmake_minimum_required(VERSION 2.6)

project(camera-self-calibration)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_DEBUG "-g -std=c++11 -stdlib=libc++")

set(CMAKE_C_FLAGS_RELEASE "-W -Wall -O3")
set(CMAKE_CXX_FLAGS_RELEASE "-W -Wall -O3 -std=c++11 -stdlib=libc++")

include_directories(${PROJECT_SOURCE_DIR}/include /usr/include /usr/local/include /usr/local/Cellar/opencv3/3.0.0/include)
link_directories(/usr/lib /usr/local/lib /usr/local/Cellar/opencv3/3.0.0/lib)

file(GLOB_RECURSE SRCS src/*.cpp)
set(OPENCV_LIBRARIES opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_ml opencv_video opencv_features2d opencv_calib3d opencv_objdetect opencv_flann opencv_viz)

link_libraries(selfcalib ${OPENCV_LIBRARIES})
add_executable(selfcalib ${SRCS})
