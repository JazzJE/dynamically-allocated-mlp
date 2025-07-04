cmake_minimum_required(VERSION 3.15)
project(neural_network)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
    src/BestStateLoader.cpp
    src/NNInitAndValidation.cpp
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
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Copy executable to project root after building
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PROJECT_NAME}> ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Copying executable to project root"
)
