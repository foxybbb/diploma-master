cmake_minimum_required(VERSION 3.8)
project(radar_bgt60tr13c_driver)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable warnings for our code, but not for external code
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find ROS 2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Configure SDK paths and options
set(SDK_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/radar_sdk)
set(SDK_LIB_DIR1 ${CMAKE_CURRENT_SOURCE_DIR}/radar_sdk/libs/linux_x64)
set(SDK_LIB_DIR2 ${CMAKE_CURRENT_SOURCE_DIR}/radar_sdk/build/bin/lib)



# Define your ROS 2 node executable
add_executable(radar_publisher_node src/radar_node.cpp)

target_include_directories(radar_publisher_node PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${SDK_ROOT}/sdk/c
    ${SDK_ROOT}/sdk/c/ifxAvian
    ${SDK_ROOT}/sdk/c/ifxFmcw
    ${SDK_ROOT}/sdk/c/ifxBase
    ${CMAKE_CURRENT_SOURCE_DIR}/radar_sdk/examples/c/BGT60TR13C/range_doppler_map
)
link_directories(${SDK_LIB_DIR1} ${SDK_LIB_DIR2})


# Link against ALL required SDK libraries
target_link_libraries(radar_publisher_node PRIVATE
    ${SDK_LIB_DIR2}/libsdk_algo.so
    ${SDK_LIB_DIR2}/libsdk_base.so
    ${SDK_LIB_DIR1}/libsdk_avian.so
    ${SDK_LIB_DIR1}/liblib_avian.so
    ${SDK_LIB_DIR2}/libsdk_fmcw.so
    ${SDK_LIB_DIR2}/libsdk_radar.so
    ${SDK_LIB_DIR2}/libstrata_shared.so

    ${SDK_LIB_DIR2}/libsdk_radar_device_common.so

    rclcpp::rclcpp
    ${std_msgs_TARGETS}
    ${sensor_msgs_TARGETS}
)



# Install rules
install(TARGETS radar_publisher_node
    RUNTIME DESTINATION lib/${PROJECT_NAME}
)


# Install the pre-built SDK libraries
install(FILES
    ${SDK_LIB_DIR2}/libsdk_base.so
    ${SDK_LIB_DIR2}/libsdk_algo.so
    ${SDK_LIB_DIR2}/libstrata_shared.so
    ${SDK_LIB_DIR1}/libsdk_avian.so
    ${SDK_LIB_DIR1}/liblib_avian.so
    ${SDK_LIB_DIR2}/libsdk_fmcw.so
    ${SDK_LIB_DIR2}/libsdk_radar.so
    ${SDK_LIB_DIR1}/libradar_sdk.so
    ${SDK_LIB_DIR2}/libsdk_radar_device_common.so
    DESTINATION lib
)

# Install launch files
install(DIRECTORY
    launch
    DESTINATION share/${PROJECT_NAME}
)
# After existing install rules
install(PROGRAMS
    radar_bgt60tr13c_driver/radar_visualizer.py
    DESTINATION lib/${PROJECT_NAME}
)
# Create environment hook for library path using DSV file
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/library_path.dsv" "prepend-non-duplicate;LD_LIBRARY_PATH;lib\n")
ament_environment_hooks("${CMAKE_CURRENT_BINARY_DIR}/library_path.dsv")

ament_package()