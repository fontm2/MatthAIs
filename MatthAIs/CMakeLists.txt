cmake_minimum_required(VERSION 2.8)

project(MatthAIs)


# activate latest c++ compiler version
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

# set flags to configure the warning settings
# Note: warnings are compiler specific
if( CMAKE_COMPILER_IS_GNUCC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()

# set build type to Debug/Release
set(CMAKE_BUILD_TYPE "Debug")


# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed
set(CMAKE_AUTOMOC ON)
# Create code from a list of Qt designer ui files
set(CMAKE_AUTOUIC ON)


######finding/importing 3rd Party packages
# Find the QtWidgets library
find_package(Qt5Widgets CONFIG REQUIRED)
# Find the QtPrintSupport library
find_package(Qt5PrintSupport REQUIRED)
# Find the Threads Library
find_package(Threads)


#include the directoreis
set(APP_INCLUDE_DIRS
    src
    src/ui
    src/ui/qcustomplot-source
    src/Eigen
)
INCLUDE_DIRECTORIES(${APP_INCLUDE_DIRS})


# Populate a CMake variable with the sources
set(APP_SOURCE
    src/main.cpp
    src/layer.h
    src/linearlayer.h
    src/linearinputlayer.h
    src/linearoutputlayer.h
    src/network.h
    src/ui/widget.cpp
    src/ui/widget.h
    src/ui/widget.ui
)
# Add additional APP_SOURCES
set(APP_SOURCE ${APP_SOURCE}
    src/ui/qcustomplot-source/qcustomplot.h
    src/ui/qcustomplot-source/qcustomplot.cpp
    )



# Tell CMake to create the ${PROJECT_NAME} executable and connect the APP_SOURCES
add_executable(${PROJECT_NAME} WIN32 ${APP_SOURCE})


########Linking the found packages in the executable
# Use the Widgets module from Qt 5
target_link_libraries(${PROJECT_NAME} Qt5::Widgets)
# Use the QtPrintSupport module from Qt 5
target_link_libraries(${PROJECT_NAME} Qt5::PrintSupport)

#copy data from the source to the build directory
file(COPY facedata.txt DESTINATION ${PROJECT_BINARY_DIR})
file(COPY Patterns.txt DESTINATION ${PROJECT_BINARY_DIR})
