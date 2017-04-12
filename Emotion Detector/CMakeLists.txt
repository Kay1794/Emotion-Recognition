cmake_minimum_required(VERSION 3.6)
project(Dataset_making)

set(CMAKE_CXX_STANDARD 11)


set(SOURCE_FILES_CAMSHOT camera_shot.cpp)
set(SOURCE_FILES_FACE face_detection.cpp)
set(SOURCE_FILE_CROP crop_picture.cpp)
set(SOURCE_FILE_EXTEND extend_dbset.cpp)
# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)
find_package(caffe REQUIRED)
find_package(Caffe)
set(Caffe_INCLUDE_DIRS "/Users/zhongyiqi/Documents/code/caffe/include")
if (Caffe_FOUND)
    message(STATUS "Caffe is found => caffeCoder will be compiled")
endif()
message(STATUS "Caffe library status:")
message(STATUS "    libraries: ${Caffe_LIBRARIES}")
message(STATUS "    include path: ${Caffe_INCLUDE_DIRS}")
# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them



message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Add OpenCV headers location to your include paths
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(camera_shot ${SOURCE_FILES_CAMSHOT})
add_executable(face_detection ${SOURCE_FILES_FACE})
add_executable(crop_picture ${SOURCE_FILE_CROP})
add_executable(extend_dbset ${SOURCE_FILE_EXTEND})
add_executable(classification classification.cpp)
add_executable(emotion_detector emotion_detector.cpp)
add_executable(CKplus_Edit CKplus_Edit.cpp)
add_executable(txtmake txtmake.cpp)

# Link your application with OpenCV libraries

target_link_libraries(camera_shot ${OpenCV_LIBS})
target_link_libraries(face_detection ${OpenCV_LIBS})
target_link_libraries(crop_picture ${OpenCV_LIBS})
target_link_libraries(extend_dbset ${OpenCV_LIBS})
target_link_libraries(classification ${Caffe_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(emotion_detector ${Caffe_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(CKplus_Edit ${OpenCV_LIBS})