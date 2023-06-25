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

# removing the required tag and adding non required to compile for only console in headless clusters
#find_package(OpenGL REQUIRED)
#find_package(QT NAMES Qt6 Qt5 COMPONENTS Widgets OpenGL Core Gui PrintSupport REQUIRED)
#find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets OpenGL Core Gui PrintSupport REQUIRED)

find_package(OpenGL)
find_package(QT NAMES Qt6 Qt5 COMPONENTS Widgets OpenGL Core Gui PrintSupport)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets OpenGL Core Gui PrintSupport)

# set variable to install with or without interface
if(OpenGL_FOUND AND (QT_FOUND OR Qt${QT_VERSION_MAJOR}_FOUND))
    set(INSTALL_GUI TRUE)

    set(SOURCE
        dcs_montecarlo.cpp
        simu_starter.cpp
        make_plot_profiles.cpp
        mask_test_c2.cpp
        obtain_time.cpp
        qcustomplot.cpp
        qdcs3dvis.cpp
        simuGlobals.cpp
        simulation_interface.cpp
        source_complex.cpp
        source_simple.cpp
        Util.cpp

        simuConsts.h
        Util.h

        dcs_montecarlo.hh
        simu_starter.hh
        make_plot_profiles.hh
        mask_test_c2.hh
        obtain_time.hh
        qcustomplot.h

        qdcs3dvis.h
        simuGlobals.hh
        simulation_interface.h

        source_complex.hh
        source_simple.hh

        parameter_settings.ui
        simulation_window.ui
        input_settings_prompt.ui
        
        sserializer.inl
        configure.inl

        parameter_settings.h
        parameter_settings.cpp

        input_settings.h
        input_settings.cpp
    )
else()
    set(INSTALL_GUI FALSE)

    set(SOURCE
        dcs_montecarlo.cpp
        simu_starter.cpp
        mask_test_c2.cpp
        obtain_time.cpp
        simuGlobals.cpp
        simulation_interface.cpp
        source_complex.cpp
        source_simple.cpp
        Util.cpp

        simuConsts.h
        Util.h

        dcs_montecarlo.hh
        simu_starter.hh
        mask_test_c2.hh
        obtain_time.hh
        
        simuGlobals.hh
        simulation_interface.h

        source_complex.hh
        source_simple.hh

        sserializer.inl
        configure.inl

        input_settings.h
        input_settings.cpp
    )
endif()


# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# add_definitions(-DQT_DISABLE_DEPRECATED_BEFORE=0x060000)    # disables all the APIs deprecated before Qt 6.0.0

#images for the interface
#qt5_add_resources(RESFILES res/res.qrc)


if(WIN32)
    set(app_WINRC ${CMAKE_SOURCE_DIR}/app_win32.rc)
    set_property(SOURCE app_WINRC APPEND PROPERTY
            OBJECT_DEPENDS ${CMAKE_SOURCE_DIR}/res/icons/simu_icon.ico
            )
endif()


# only install the lib wrapper if we have qt in the system
if(INSTALL_GUI)
    # We could either use this as a submodule and everything would be simpler (but not if we are using a private repo)
    # So it is better to install on the system and allow for find_package() to do its magic

    # Set default install dir to the install dir
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE STRING " " FORCE)
    # Offer the user the choice of overriding the installation directories
    set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
    set(INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files")
    set(INSTALL_DATA_DIR data CACHE PATH "Installation directory for data files")
    if(WIN32 AND NOT CYGWIN)
    set(DEF_INSTALL_CMAKE_DIR CMake)
    else()
    set(DEF_INSTALL_CMAKE_DIR lib/CMake/DCSsimuWrapper)
    endif()
    set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH "Installation directory for CMake files")

    set(INSTALL_HEADERS
        simulation_interface.h
        simuGlobals.hh
        parameter_settings.h
        simuConsts.h
    )

    file(REAL_PATH "${CMAKE_INSTALL_PREFIX}/${INSTALL_INCLUDE_DIR}" ABS_INCLUDE_DIR)
    set(CONF_INCLUDE_DIRS "${ABS_INCLUDE_DIR}")
    configure_file(cmake/DCSsimuWrapperConfig.cmake.in "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/DCSsimuWrapperConfig.cmake" @ONLY)

    # Just using a static lib (improves the "portability" of the final executable)
    add_library(DCSsimuWrapper STATIC ${SOURCE})
    target_compile_definitions(DCSsimuWrapper PRIVATE -DENTRYPOINT=dummy -DLIB_DEF -DQT_EXISTS -DLIB_DATA_DIR="${CMAKE_INSTALL_PREFIX}/${INSTALL_DATA_DIR}")

    target_link_libraries(DCSsimuWrapper PRIVATE
        Qt${QT_VERSION_MAJOR}::Widgets
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Gui
        Qt${QT_VERSION_MAJOR}::OpenGL
        Qt${QT_VERSION_MAJOR}::PrintSupport
        "${OPENGL_LIBRARIES}"
    )

    install(TARGETS DCSsimuWrapper EXPORT DCSsimuWrapper DESTINATION ${INSTALL_LIB_DIR})
    install(FILES ${CMAKE_SOURCE_DIR}/${INSTALL_HEADERS} DESTINATION ${INSTALL_INCLUDE_DIR}/DCSsimu)
    install(FILES ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/DCSsimuWrapperConfig.cmake DESTINATION ${INSTALL_CMAKE_DIR})
    install(EXPORT DCSsimuWrapper DESTINATION ${INSTALL_CMAKE_DIR})

    # TODO : Fix target exporting to the registry (avoiding setting the CMAKE_PREFIX_PATH on the client)
    export(EXPORT DCSsimuWrapper)
    export(PACKAGE DCSsimuWrapper)

    # Install aditional files into the data/ directory
    # Crystal profiles
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/simuWorkspace/Crystal_Profiles DESTINATION ${CMAKE_INSTALL_PREFIX}/data)

    # OpenGL Models
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/simuWorkspace/DCSModels DESTINATION ${CMAKE_INSTALL_PREFIX}/data)

    # Copy files for the build dir
    add_custom_target(copy-workspace-build ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/simuWorkspace ${CMAKE_BINARY_DIR}/simuWorkspace
    )
    # file(COPY ${CMAKE_SOURCE_DIR}/simuWorkspace/ DESTINATION ${CMAKE_BINARY_DIR}/)
endif()


# Keep the original target here, in case we want to build standalone with cmake 
# no need to use qmake (but useful if using qtcreator)
# Keep it a console application just like in the base port qmake file
add_executable(DCSsimu ${SOURCE} ${app_WINRC})

if(INSTALL_GUI)
    # Disable console if in release mode and with the interface installed
    set_target_properties(DCSsimu PROPERTIES WIN32_EXECUTABLE $<CONFIG:Release>)

    target_compile_definitions(DCSsimu PRIVATE -DENTRYPOINT=main -DQT_EXISTS)

    target_link_libraries(DCSsimu PRIVATE
    Qt${QT_VERSION_MAJOR}::Widgets
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Gui
    Qt${QT_VERSION_MAJOR}::OpenGL
    Qt${QT_VERSION_MAJOR}::PrintSupport
    "${OPENGL_LIBRARIES}"
    )
else()
    target_compile_definitions(DCSsimu PRIVATE -DENTRYPOINT=main)
endif()