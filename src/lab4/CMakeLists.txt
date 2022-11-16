set(LAB_TARGET_PREFIX ${PROJECT_TARGET_PREFIX}lab4_)
set(TARGET_NAME ${LAB_TARGET_PREFIX}tasks_all)

add_executable(${TARGET_NAME} main.cpp)
target_compile_options(${TARGET_NAME} PRIVATE /openmp)
