cmake_minimum_required(VERSION 3.0)
project(cl-hello)

find_package(OpenCL REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL)
