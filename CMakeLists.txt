cmake_minimum_required(VERSION 2.8)
project( tracking )
find_package( OpenCV REQUIRED )
add_executable(tracking main.cpp tracker.cpp tracker.hpp functions.hpp)
target_link_libraries(tracking ${OpenCV_LIBS})


