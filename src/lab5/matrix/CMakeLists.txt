add_library(${LAB_TARGET_PREFIX}matrix STATIC
        include/matrix.h
        src/matrix.cpp
        )

target_include_directories(${LAB_TARGET_PREFIX}matrix PUBLIC include)
