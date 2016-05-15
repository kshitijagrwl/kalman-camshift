cmake_minimum_required(VERSION 2.8)
project( tracking )
find_package( OpenCV REQUIRED )
add_executable(tracking camshift.cpp)
target_link_libraries( tracking ${OpenCV_LIBS})


