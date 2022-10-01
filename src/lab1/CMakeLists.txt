set(LAB_TARGET_PREFIX ${PROJECT_TARGET_PREFIX}lab1_)

add_executable(${LAB_TARGET_PREFIX}task1 task1.cpp)
add_executable(${LAB_TARGET_PREFIX}task2 task2.cpp)

if (CMAKE_CXX_COMPILER_ID MATCHES MSVC)
    target_compile_definitions(${LAB_TARGET_PREFIX}task2 PRIVATE LAB1_HAVE_OMP)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES Clang)
    target_compile_definitions(${LAB_TARGET_PREFIX}task2 PRIVATE LAB1_CXX_INLINE_ASSEMBLY_SUPPORTED)
endif ()

add_executable(${LAB_TARGET_PREFIX}task3_4 task3_4.cpp)

add_subdirectory(matrix)

add_executable(${LAB_TARGET_PREFIX}task5 task5.cpp)
target_link_libraries(${LAB_TARGET_PREFIX}task5 PRIVATE ${LAB_TARGET_PREFIX}matrix)
