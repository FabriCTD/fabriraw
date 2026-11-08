cmake_minimum_required(VERSION 3.24)

project(fraw VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(FRAW_BUILD_TESTS "Build the FabriRAW core tests" ON)

# MinGW: link the GCC runtime statically so binaries do not pick up a
# different libstdc++-6.dll from PATH (fails with 0xC0000139 otherwise).
if(MINGW)
  add_link_options(-static -static-libgcc -static-libstdc++)
endif()
option(FRAW_FETCH_ZLIB "Always build zlib from source instead of using a system zlib" OFF)

# ---------------------------------------------------------------------------
# zlib: system package if available, otherwise madler/zlib v1.3.1 built static
# ---------------------------------------------------------------------------
if(NOT FRAW_FETCH_ZLIB)
  find_package(ZLIB QUIET)
endif()

if(ZLIB_FOUND)
  message(STATUS "fraw: using system zlib ${ZLIB_VERSION_STRING}")
  set(FRAW_ZLIB_TARGET ZLIB::ZLIB)
else()
  message(STATUS "fraw: system zlib not found, fetching madler/zlib v1.3.1")
  include(FetchContent)
  # SOURCE_SUBDIR points to a non-existent directory so zlib's own
  # CMakeLists.txt is NOT used; we compile the needed sources ourselves.
  FetchContent_Declare(fraw_zlib_src
    GIT_REPOSITORY https://github.com/madler/zlib.git
    GIT_TAG v1.3.1
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR _fraw_no_cmake_)
  FetchContent_MakeAvailable(fraw_zlib_src)
  set(_z ${fraw_zlib_src_SOURCE_DIR})
  add_library(fraw_zlib STATIC
    ${_z}/adler32.c ${_z}/compress.c ${_z}/crc32.c ${_z}/deflate.c ${_z}/infback.c
    ${_z}/inffast.c ${_z}/inflate.c ${_z}/inftrees.c ${_z}/trees.c ${_z}/uncompr.c ${_z}/zutil.c)
  target_include_directories(fraw_zlib SYSTEM PUBLIC ${_z})
  set_target_properties(fraw_zlib PROPERTIES POSITION_INDEPENDENT_CODE ON C_STANDARD 99)
  if(MSVC)
    target_compile_definitions(fraw_zlib PRIVATE _CRT_SECURE_NO_DEPRECATE _CRT_NONSTDC_NO_DEPRECATE)
  endif()
  set(FRAW_ZLIB_TARGET fraw_zlib)
endif()

# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------
add_library(fraw_warnings INTERFACE)
if(MSVC)
  target_compile_options(fraw_warnings INTERFACE /W4 /permissive- /utf-8)
else()
  target_compile_options(fraw_warnings INTERFACE -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion)
endif()

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
add_library(fraw STATIC
  src/chunk.cpp
  src/container.cpp
  src/fset.cpp
  src/golden.cpp
  src/json.cpp
  src/zlib_codec.cpp)
target_include_directories(fraw PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(fraw PRIVATE ${FRAW_ZLIB_TARGET} fraw_warnings)
target_compile_features(fraw PUBLIC cxx_std_20)

# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------
add_executable(fraw-info tools/fraw_info.cpp)
target_link_libraries(fraw-info PRIVATE fraw fraw_warnings)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(FRAW_BUILD_TESTS)
  enable_testing()
  add_executable(fraw_tests
    tests/harness.cpp
    tests/test_basics.cpp
    tests/test_errors.cpp
    tests/test_fset.cpp
    tests/test_golden.cpp)
  target_link_libraries(fraw_tests PRIVATE fraw ${FRAW_ZLIB_TARGET} fraw_warnings)
  target_include_directories(fraw_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  get_filename_component(FRAW_FIXTURE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tests/fixtures" ABSOLUTE)
  target_compile_definitions(fraw_tests PRIVATE FRAW_FIXTURE_DIR="${FRAW_FIXTURE_DIR}")

  # One CTest entry per test group (prefix filter understood by the harness).
  foreach(group layout crc json roundtrip errors truncation unknown fuzz update_dev fset golden)
    add_test(NAME fraw.${group} COMMAND fraw_tests ${group})
  endforeach()
  add_test(NAME fraw-info.fixture
    COMMAND fraw-info --make-fixture ${CMAKE_CURRENT_BINARY_DIR}/fixture_cli.fraw)
  add_test(NAME fraw-info.inspect COMMAND fraw-info ${FRAW_FIXTURE_DIR}/golden.fraw)
  set_tests_properties(fraw-info.inspect PROPERTIES PASS_REGULAR_EXPRESSION "\nOK\n")
endif()
