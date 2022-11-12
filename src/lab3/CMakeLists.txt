set(LAB_TARGET_PREFIX ${PROJECT_TARGET_PREFIX}lab3_)
set(TARGET_NAME ${LAB_TARGET_PREFIX}tasks_all)

add_executable(${TARGET_NAME} main.cpp)

add_subdirectory(cpuinfo)

target_link_libraries(${TARGET_NAME} PRIVATE ${LAB_TARGET_PREFIX}cpu_info)
