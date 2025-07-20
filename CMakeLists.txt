cmake_minimum_required(VERSION 3.15)
project(neural_network)

# Set C++ standard to 17 for filesystem
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find and require OpenMP
find_package(OpenMP REQUIRED)

# Source files
set(SOURCES
    main.cpp
    src/DenseLayer.cpp
    src/OutputLayer.cpp
    src/NeuralNetwork.cpp
    src/InitializationFunctions.cpp
    src/MemoryFunctions.cpp
    src/MenuFunctions.cpp
    src/StatisticsFunctions.cpp
    src/SavedStateLoader.cpp
    src/NNInitAndValidation.cpp
    src/TrainingLogAndList.cpp
)

# Header files (for IDE organization)
set(HEADERS
    include/DenseLayer.h
    include/OutputLayer.h
    include/NeuralNetwork.h
    include/InitializationFunctions.h
    include/MemoryFunctions.h
    include/MenuFunctions.h
    include/StatisticsFunctions.h
    include/Constants.h
    include/TrainingLogAndList.h
)

# For relative access to the nn saved state and dataset files
add_compile_definitions(PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link OpenMP (guaranteed to exist due to REQUIRED)
target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)

# For older GCC versions that need explicit filesystem linking
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    target_link_libraries(${PROJECT_NAME} stdc++fs)
endif()
