# ============================================================================ #
#  César Godinho                                                               #
#  13/05/2023                                                                  #
# ============================================================================ #

# This CMake file creates a shared library that side-loads the window wrapper for DCScan-GUI
# Its sole porpose is to create a bridge to the official C++ port both for the simulation 
# and configure windows

cmake_minimum_required(VERSION 3.5)

project(DCSsimu VERSION 1.0)

cmake_policy(SET CMP0100 NEW)
cmake_policy(SET CMP0090 OLD)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Lets cheat and supress all warnings...
# Just let the original port authors fix them eventually
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /permissive- /D_USE_MATH_DEFINES /W0")

list(APPEND CMAKE_PREFIX_PATH "C:/Qt/5.15.2/msvc2019_64")

find_package(OpenGL REQUIRED)
find_package(QT NAMES Qt6 Qt5 COMPONENTS Widgets OpenGL Core Gui PrintSupport REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets OpenGL Core Gui PrintSupport REQUIRED)

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# add_definitions(-DQT_DISABLE_DEPRECATED_BEFORE=0x060000)    # disables all the APIs deprecated before Qt 6.0.0

set(SOURCE
    CheckInputSpectrum.cpp
    analysie_voigt.cpp
    calculate_geo_corr.cpp
    dcs_montecarlo.cpp
    double_crystal_diffraction.cpp
    gaussi_rockin.cpp
    get_new_temp_for_bin.cpp
    intensitysource.cpp
    limits_horizontal_diverge.cpp
    dcs_gui.cpp
    make_graph_imageplates.cpp
    make_plot_profiles.cpp
    make_pointcryst.cpp
    makefitvoigt.cpp
    mask_test_c2.cpp
    obtain_curve_responce.cpp
    obtain_curved_hor_corr.cpp
    obtain_data_fitting.cpp
    obtain_energyspectrum.cpp
    obtain_misalign.cpp
    obtain_time.cpp
    obtainvertical.cpp
    qcustomplot.cpp
    qdcs3dvis.cpp
    set_ang_rad.cpp
    simulationmain.cpp
    source_complex.cpp
    source_simple.cpp
    test_input.cpp

    Boxmuller.hh
    CheckInputSpectrum.hh
    FindLoc.hh
    First_crystal_angle.hh
    First_crystal_angle_1_Approximation.hh
    First_crystal_angle_Full.hh
    Obtain_d_Lattice_temp.hh
    Obtain_ypzp.hh
    PseuVoigt.hh
    Reach_detector.hh
    Second_crystal_angle.hh
    Second_crystal_angle_Full.hh
    Second_crystal_angle_aproximation.hh
    TrueVoig.hh
    analysie_voigt.hh
    calculate_geo_corr.hh
    covsrt.hh
    cw.hh
    dcs_gui.h
    dcs_montecarlo.hh
    double_crystal_diffraction.hh
    gaussi_rockin.hh
    gaussj.hh
    get_new_temp_for_bin.hh
    intensitysource.hh
    limits_horizontal_diverge.hh
    make_graph_imageplates.hh
    make_plot_profiles.hh
    make_pointcryst.hh
    makefitvoigt.hh
    mask_test_c2.hh
    mrqcof.hh
    mrqmin.hh
    obtain_curve_responce.hh
    obtain_curved_hor_corr.hh
    obtain_data_fitting.hh
    obtain_energyspectrum.hh
    obtain_misalign.hh
    obtain_time.hh
    obtainvertical.hh
    qcustomplot.h
    qdcs3dvis.h
    set_ang_rad.hh
    simuGlobals.hh
    simulationmain.h
    source_complex.hh
    source_simple.hh
    spline.hh
    splint.hh
    test_input.hh

    dcs_gui.ui
    simulationmain.ui
)

# Just using a static lib (improves the "portability" of the final executable)
add_library(DCSsimuWrapper STATIC ${SOURCE})

target_link_libraries(DCSsimuWrapper PRIVATE
    Qt${QT_VERSION_MAJOR}::Widgets
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Gui
    Qt${QT_VERSION_MAJOR}::OpenGL
    Qt${QT_VERSION_MAJOR}::PrintSupport
    "${OPENGL_LIBRARIES}"
)

# We could either use this as a submodule and everything would be simpler (but not if we are using a private repo)
# So it is better to install on the system and allow for find_package() to do its magic

# Set default install dir to the install dir
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE STRING " " FORCE)
# Offer the user the choice of overriding the installation directories
set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files")
if(WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKE_DIR CMake)
else()
  set(DEF_INSTALL_CMAKE_DIR lib/CMake/DCSsimuWrapper)
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH "Installation directory for CMake files")

set(INSTALL_HEADERS
    simulationmain.h
    simuGlobals.hh
)

file(REAL_PATH "${CMAKE_INSTALL_PREFIX}/${INSTALL_INCLUDE_DIR}" ABS_INCLUDE_DIR)
set(CONF_INCLUDE_DIRS "${ABS_INCLUDE_DIR}")
configure_file(cmake/DCSsimuWrapperConfig.cmake.in "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/DCSsimuWrapperConfig.cmake" @ONLY)

install(TARGETS DCSsimuWrapper EXPORT DCSsimuWrapper DESTINATION ${INSTALL_LIB_DIR})
install(FILES ${CMAKE_SOURCE_DIR}/${INSTALL_HEADERS} DESTINATION ${INSTALL_INCLUDE_DIR}/DCSsimu)
install(FILES ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/DCSsimuWrapperConfig.cmake DESTINATION ${INSTALL_CMAKE_DIR})
install(EXPORT DCSsimuWrapper DESTINATION ${INSTALL_CMAKE_DIR})

# TODO : Fix target exporting to the registry (avoiding setting the CMAKE_PREFIX_PATH on the client)
export(EXPORT DCSsimuWrapper)
export(PACKAGE DCSsimuWrapper)

# Keep the original target here, in case we want to build standalone with cmake 
# no need to use qmake (but useful if using qtcreator)
# Keep it a console application just like in the base port qmake file
add_executable(DCSsimu ${SOURCE})

target_link_libraries(DCSsimu PRIVATE
    Qt${QT_VERSION_MAJOR}::Widgets
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Gui
    Qt${QT_VERSION_MAJOR}::OpenGL
    Qt${QT_VERSION_MAJOR}::PrintSupport
    "${OPENGL_LIBRARIES}"
)
