cmake_minimum_required(VERSION 3.24)
project(sycl_profile_tracy)

set(CMAKE_CXX_STANDARD 17)

find_package(hipSYCL REQUIRED)
find_package(fmt REQUIRED)

set(BUILD_SHARED_LIBS ON)
set(TRACY_FIBERS ON CACHE BOOL "" FORCE)
add_subdirectory(contrib/tracy)

add_executable(sycl_profile_tracy main.cpp)
add_sycl_to_target(TARGET sycl_profile_tracy SOURCES main.cpp)
target_link_libraries(sycl_profile_tracy PUBLIC TracyClient fmt)
