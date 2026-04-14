cmake_minimum_required(VERSION 4.1)
project(SIMD_Optimized_Financial_Calculator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
find_package(Threads REQUIRED)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
	benchmark
	GIT_REPOSITORY https://github.com/google/benchmark.git
	GIT_TAG v1.9.1
)
FetchContent_MakeAvailable(benchmark)

add_executable(SIMD_Optimized_Financial_Calculator main.cpp)
target_link_libraries(SIMD_Optimized_Financial_Calculator PRIVATE benchmark::benchmark Threads::Threads)

target_compile_options(SIMD_Optimized_Financial_Calculator PRIVATE
	$<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -mavx2 -mfma>
	$<$<CXX_COMPILER_ID:MSVC>:/O2 /arch:AVX2>
)
