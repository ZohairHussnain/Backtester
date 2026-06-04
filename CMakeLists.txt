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

# Test executable
enable_testing()

add_executable(BackTesterTests
    Tests.cpp
)

target_link_libraries(BackTesterTests PRIVATE
    nlohmann_json::nlohmann_json
)

add_test(NAME BackTesterTests COMMAND BackTesterTests)
