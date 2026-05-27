cmake_minimum_required(VERSION 3.20)

project(BackTester LANGUAGES CXX)

# Use modern C++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# BackTester executable
add_executable(BackTester
    Source.cpp
)

# Dependencies via vcpkg
find_package(cpr CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

target_link_libraries(BackTester PRIVATE
    cpr::cpr
    nlohmann_json::nlohmann_json
)

# ── Tests ────────────────────────────────────────────────────────────────────
enable_testing()

# Locate Google Test (installed via libgtest-dev)
find_package(GTest REQUIRED)

add_executable(BacktesterTests
    tests/test_backtester.cpp
)

target_include_directories(BacktesterTests PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(BacktesterTests PRIVATE
    GTest::GTest
    GTest::Main
)

include(GoogleTest)
gtest_discover_tests(BacktesterTests)
