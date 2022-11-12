set(TARGET_NAME ${LAB_TARGET_PREFIX}cpu_info)

add_library(${TARGET_NAME}
        include/cpuinfo.h
        src/cpuinfo.cpp
        )

target_include_directories(${TARGET_NAME} PUBLIC include)
