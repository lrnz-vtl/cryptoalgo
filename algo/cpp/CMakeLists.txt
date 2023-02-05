cmake_minimum_required(VERSION 3.13)  # CMake version check
project(cseries)               # Create project "simple_example"
set(CMAKE_CXX_STANDARD 14)            # Enable c++14 standard

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})


include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

find_package(GTest REQUIRED)


# Add executable target with source files listed in SOURCE_FILES variable
add_executable(series series.cpp series_main.cpp)
target_link_libraries(series ${PYTHON_LIBRARIES})


enable_testing()

add_executable(
  series_tests
  series_tests.cpp
  series.cpp
)

target_link_libraries(
    series_tests
    # ${GTEST_LIBRARIES}
    gtest_main
    ${PYTHON_LIBRARIES}
)

include(GoogleTest)
gtest_discover_tests(series_tests)

# add_test(series_tests series_tests)
