set(LAB_TARGET_PREFIX ${PROJECT_TARGET_PREFIX}lab5_)
set(TARGET_NAME ${LAB_TARGET_PREFIX}tasks_all)

add_subdirectory(matrix)

add_executable(${TARGET_NAME} main.cpp)
target_compile_options(${TARGET_NAME} PRIVATE /openmp)
target_link_libraries(${TARGET_NAME} PRIVATE ${LAB_TARGET_PREFIX}matrix)
