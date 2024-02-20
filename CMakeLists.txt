cmake_minimum_required(VERSION 3.16.3)

set(PATCH_VERSION "1" CACHE INTERNAL "Patch version")
set(PROJECT_VERSION 0.0.${PATCH_VERSION})

project(fashio_mnist VERSION ${PROJECT_VERSION})

configure_file(version.h.in version.h)

find_package(Boost CONFIG COMPONENTS program_options REQUIRED)

include(${PROJECT_SOURCE_DIR}/cmake/conan.cmake)
conan_cmake_run(
  CONANFILE conanfile.txt
  BASIC_SETUP
  CMAKE_TARGETS # individual targets to link to
  BUILD
    missing 
)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/mlp_classifier.cpp
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    CONAN_PKG::eigen
    Boost::program_options
)

target_include_directories(${PROJECT_NAME} PRIVATE
    "${CMAKE_BINARY_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

find_package(GTest REQUIRED)

add_executable(${PROJECT_NAME}_test 
    tests/main.cpp
    tests/mlp_test.cpp
    src/mlp_classifier.cpp
)

target_link_libraries(${PROJECT_NAME}_test GTest::gtest  CONAN_PKG::eigen)

target_include_directories(${PROJECT_NAME}_test PRIVATE
    "${CMAKE_BINARY_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
    
set_target_properties(${PROJECT_NAME} ${PROJECT_NAME}_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_test RUNTIME DESTINATION bin)

set(CPACK_GENERATOR DEB)

set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")

set(CPACK_PACKAGE_CONTACT example@example.com)

include(CPack)

enable_testing()