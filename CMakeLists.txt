cmake_minimum_required(VERSION 2.8.3)
project(object_tracking)

find_package(catkin REQUIRED COMPONENTS
    rospy
    object_detector_msgs
    )

catkin_python_setup()
catkin_package(
  CATKIN_DEPENDS object_detector_msgs
)

install(PROGRAMS
   bin/object_tracking
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)