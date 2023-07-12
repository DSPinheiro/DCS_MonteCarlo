# ============================================================================ #
#  César Godinho             Daniel Pinheiro                                   #
#  13/05/2023                02/07/2023                                        #
# ============================================================================ #

# This CMake file creates a shared library that side-loads the window wrapper for DCScan-GUI
# Its sole porpose is to create a bridge to the official C++ port both for the simulation 
# and configure windows

cmake_minimum_required(VERSION 3.18)

project(DCS_Simulation VERSION 1.0)


# SETUP THE CXX STANDARD AND FLAGS

cmake_policy(SET CMP0100 NEW)
cmake_policy(SET CMP0090 OLD)
cmake_policy(SET CMP0104 NEW)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Lets cheat and supress all warnings...
# Just let the original port authors fix them eventually
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /permissive- /D_USE_MATH_DEFINES /W0")

list(APPEND CMAKE_PREFIX_PATH "C:/Qt/5.15.2/msvc2019_64")

# include_directories("include/")
set(CMAKE_AUTOUIC_SEARCH_PATHS "ui/")

# set(CMAKE_VERBOSE_MAKEFILE ON)

include_directories(include src)


# SETUP THE QT AND OPENGL LIBRARY SUPPORT

# NOT required to compile for only console in headless clusters
find_package(OpenGL)
find_package(QT NAMES Qt6 Qt5 COMPONENTS Widgets OpenGL Core Gui PrintSupport)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets OpenGL Core Gui PrintSupport)

# set variable to install with or without interface
if(OpenGL_FOUND AND (QT_FOUND OR Qt${QT_VERSION_MAJOR}_FOUND))
    set(INSTALL_GUI TRUE)
else()
    set(INSTALL_GUI FALSE)
endif()


# SETUP THE OPENMP LIBRARY

# load the openMP package for multiprocessing
find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    add_compile_options($<$<COMPILE_LANG_AND_ID:CXX,GNU>:-openmp:experimental>)
endif()


# SETUP THE CUDA LANGUAGE FOR GPU ACCELERATION

# load CUDA for GPU processing
include(CheckLanguage)
check_language(CUDA)

# TODO: check if this works in a system without CUDA and/or without CUB
if(CMAKE_CUDA_COMPILER)
    if(NOT EXISTS "${CUB_INCLUDE_DIR}")
        message(STATUS "Searching for the required CUB CUDA library...")
        find_path(
            CUB_INCLUDE_DIR 
            cub/cub.cuh
            HINTS
                ${CUDA_INCLUDE_DIRS}
                ${CMAKE_SOURCE_DIR}/include
                ${CMAKE_SOURCE_DIR}
                ${PROJECT_SOURCE_DIR}
                ${PROJECT_SOURCE_DIR}/include 
                /opt 
                $ENV{HOME}/opt 
                ENV CUB_DIR 
                ENV CUB_INCLUDE_DIR 
                ENV CUB_PATH
            DOC "nVIDIA CUB GPU primitives header-only CUDA library"
            PATH_SUFFIXES cub libcub nvidia-cub 
        )
    endif()

    if(EXISTS ${CUB_INCLUDE_DIR})
        message(STATUS "Found CUB library at: ${CUB_INCLUDE_DIR}")
        message(STATUS "Enabling CUDA language")

        enable_language(CUDA)

        set(CMAKE_CUDA_STANDARD 17)
        # add_link_options(/NODEFAULTLIB:LIBCMT.LIB)
        
        find_package(CUDAToolkit)

        include(FindCUDA/select_compute_arch)
        CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
        string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
        string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
        string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
        SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})

        add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-openmp:experimental>)
        add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--disable-warnings>)
        add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-dc>)
        add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)
    else()
        message(STATUS "Could not find CUB library. Disabling CUDA language")
        set(CMAKE_CUDA_COMPILER FALSE)
    endif()
endif()



# MANUAL TOGGLES FOR THE LIBRARIES
# TODO: make these as build targets from the cmd line
# set(INSTALL_GUI FALSE) # uncomment to disable the QT GUI
# message(NOTICE "QT GUI interface was manually disabled!") # uncomment to disable the QT GUI
# set(OpenMP_CXX_FOUND FALSE) # uncomment to disable the OpenMP support
# message(NOTICE "OpenMP support was manually disabled!") # uncomment to disable the OpenMP support
# set(CMAKE_CUDA_COMPILER FALSE) # uncomment to disable the CUDA support
# message(NOTICE "CUDA support was manually disabled!") # uncomment to disable the CUDA support


# SETUP THE PROPER SOURCE FILES TO COMPILE

set(SOURCE_CLI
    dcs_montecarlo.cpp
    simu_starter.cpp
    mask_test_c2.cpp
    obtain_time.cpp
    simuGlobals.cpp
    simulation_interface.cpp
    source_complex.cpp
    source_simple.cpp
    Util.cpp
    
    sserializer.inl
    configure.inl

    input_settings.cpp
)

if(INSTALL_GUI)
    set(SOURCE_GUI ${SOURCE_CLI})
    list(APPEND SOURCE_GUI
        make_plot_profiles.cpp
        qcustomplot.cpp
        qdcs3dvis.cpp

        parameter_settings.cpp
    )
endif()

list(TRANSFORM SOURCE_CLI PREPEND "src/")
list(TRANSFORM SOURCE_GUI PREPEND "src/")


# NEED TO MANUALLY ADD INCLUDE FILES FOR THE SIMULATION IN CLI MODE!!!!
file(GLOB SOURCES_CLI
        "include/dcs_montecarlo.hh"
        "include/simu_starter.hh"
        "include/mask_test_c2.hh"
        "include/obtain_time.hh"
        "include/simuGlobals.hh"
        "include/simuConsts.h"
        "include/simulation_interface.h"
        "include/source_complex.hh"
        "include/source_simple.hh"
        "include/Util.h"
        "include/input_settings.h"
        ${SOURCE_CLI}
)

file(GLOB SOURCES_GUI
        "include/*"
        ${SOURCE_GUI}
)


# PREPARE THE DEVICE CODE LIBRARY TO LINK WITH THE MAIN PROJECT

if(CMAKE_CUDA_COMPILER)
    message(STATUS "Setup static library with GPU device code")    

    set(CUDA_SOURCE
        parallel_bin.cu
        Util_cuda.cu
    )

    list(TRANSFORM CUDA_SOURCE PREPEND "src_cuda/")

    file(GLOB CUDA_SOURCES
        "include_cuda/*.cuh"
        "include/simuConsts.h"
        ${CUDA_SOURCE}
    )

    set(_CMAKE_CUDA_EXTRA_FLAGS "-forward-unknown-to-host-compiler -ccbin=\"${CMAKE_CXX_COMPILER}\"")

    add_library(GPUBin STATIC ${CUDA_SOURCES})
    set_target_properties(GPUBin PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON
                            CUDA_RESOLVE_DEVICE_SYMBOLS ON
                            POSITION_INDEPENDENT_CODE ON
                        )
    
    target_compile_definitions(GPUBin PRIVATE -DCUDA)
    # target_link_options(GPUBin PRIVATE /NODEFAULTLIB:LIBCMT.LIB)
endif()



# ADD THE RESOURCE FILE FOR THE EXE ICON ON WINDOWS

if(WIN32)
    set(app_WINRC ${CMAKE_SOURCE_DIR}/res/app_win32.rc)
    set_property(SOURCE app_WINRC APPEND PROPERTY
            OBJECT_DEPENDS ${CMAKE_SOURCE_DIR}/res/icons/simu_icon.ico
            )
endif()



# LIBRARY WRAPPER BUILD AND INSTALL SETTINGS

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
    add_library(DCSsimuWrapper STATIC ${SOURCES_GUI})
    
    if(CMAKE_CUDA_COMPILER)
        if(OpenMP_CXX_FOUND)
            message(STATUS "Configuring <LIB Wrapper> target for OpenMP and CUDA parallelization")
            target_compile_definitions(DCSsimuWrapper PRIVATE -DENTRYPOINT=dummy -DLIB_DEF -DQT_EXISTS -DOPENMP -DCUDA -DLIB_DATA_DIR="${CMAKE_INSTALL_PREFIX}/${INSTALL_DATA_DIR}")
        else()
            message(STATUS "Configuring <LIB Wrapper> target for CUDA parallelization")
            target_compile_definitions(DCSsimuWrapper PRIVATE -DENTRYPOINT=dummy -DLIB_DEF -DQT_EXISTS -DCUDA -DLIB_DATA_DIR="${CMAKE_INSTALL_PREFIX}/${INSTALL_DATA_DIR}")
        endif()
        set_target_properties(DCSsimuWrapper PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        
        # target_link_options(DCSsimuWrapper PRIVATE /NODEFAULTLIB:LIBCMT.LIB)
        target_link_libraries(DCSsimuWrapper PRIVATE GPUBin)
    else()
        if(OpenMP_CXX_FOUND)
            message(STATUS "Configuring <LIB Wrapper> target for OpenMP parallelization")
            target_compile_definitions(DCSsimuWrapper PRIVATE -DENTRYPOINT=dummy -DLIB_DEF -DQT_EXISTS -DOPENMP -DLIB_DATA_DIR="${CMAKE_INSTALL_PREFIX}/${INSTALL_DATA_DIR}")
        else()
            message(STATUS "Configuring <LIB Wrapper> target for NO parallelization")
            target_compile_definitions(DCSsimuWrapper PRIVATE -DENTRYPOINT=dummy -DLIB_DEF -DQT_EXISTS -DLIB_DATA_DIR="${CMAKE_INSTALL_PREFIX}/${INSTALL_DATA_DIR}")
        endif()
    endif()

    if(OpenMP_CXX_FOUND)
        target_link_libraries(DCSsimuWrapper PRIVATE
            Qt${QT_VERSION_MAJOR}::Widgets
            Qt${QT_VERSION_MAJOR}::Core
            Qt${QT_VERSION_MAJOR}::Gui
            Qt${QT_VERSION_MAJOR}::OpenGL
            Qt${QT_VERSION_MAJOR}::PrintSupport
            "${OPENGL_LIBRARIES}"
            OpenMP::OpenMP_CXX
        )
    else()
        target_link_libraries(DCSsimuWrapper PRIVATE
            Qt${QT_VERSION_MAJOR}::Widgets
            Qt${QT_VERSION_MAJOR}::Core
            Qt${QT_VERSION_MAJOR}::Gui
            Qt${QT_VERSION_MAJOR}::OpenGL
            Qt${QT_VERSION_MAJOR}::PrintSupport
            "${OPENGL_LIBRARIES}"
        )
    endif()

    install(TARGETS DCSsimuWrapper EXPORT DCSsimuWrapper DESTINATION ${INSTALL_LIB_DIR})
    install(FILES ${CMAKE_SOURCE_DIR}/${INSTALL_HEADERS} DESTINATION ${INSTALL_INCLUDE_DIR}/DCSsimu)
    install(FILES ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/DCSsimuWrapperConfig.cmake DESTINATION ${INSTALL_CMAKE_DIR})
    install(EXPORT DCSsimuWrapper DESTINATION ${INSTALL_CMAKE_DIR})
    
    if(CMAKE_CUDA_COMPILER)
        install(TARGETS GPUBin EXPORT DCSsimuWrapper)
    endif()

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

    add_custom_target(LIBWrapper ALL DEPENDS DCSsimuWrapper)
endif()




# EXE BUILD SETTINGS

if(INSTALL_GUI)
    # Keep the original target here, in case we want to build standalone with cmake 
    add_executable(DCSsimu ${SOURCES_GUI} ${app_WINRC})

    # Disable console if in release mode and with the interface installed
    set_target_properties(DCSsimu PROPERTIES WIN32_EXECUTABLE $<CONFIG:Release>)

    if(CMAKE_CUDA_COMPILER)
        if(OpenMP_CXX_FOUND)
            message(STATUS "Configuring <GUI application> target for OpenMP and CUDA parallelization")
            target_compile_definitions(DCSsimu PRIVATE -DENTRYPOINT=main -DQT_EXISTS -DOPENMP -DCUDA)
        else()
            message(STATUS "Configuring <GUI application> target for CUDA parallelization")
            target_compile_definitions(DCSsimu PRIVATE -DENTRYPOINT=main -DQT_EXISTS -DCUDA)
        endif()
        set_target_properties(DCSsimu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        
        target_link_options(DCSsimu PRIVATE /NODEFAULTLIB:LIBCMT.LIB)
        target_link_libraries(DCSsimu PRIVATE GPUBin)
    else()
        if(OpenMP_CXX_FOUND)
            message(STATUS "Configuring <GUI application> target for OpenMP parallelization")
            target_compile_definitions(DCSsimu PRIVATE -DENTRYPOINT=main -DQT_EXISTS -DOPENMP)
        else()
            message(STATUS "Configuring <GUI application> target for NO parallelization")
            target_compile_definitions(DCSsimu PRIVATE -DENTRYPOINT=main -DQT_EXISTS)
        endif()
    endif()

    if(OpenMP_CXX_FOUND)
        target_link_libraries(DCSsimu PRIVATE
            Qt${QT_VERSION_MAJOR}::Widgets
            Qt${QT_VERSION_MAJOR}::Core
            Qt${QT_VERSION_MAJOR}::Gui
            Qt${QT_VERSION_MAJOR}::OpenGL
            Qt${QT_VERSION_MAJOR}::PrintSupport
            "${OPENGL_LIBRARIES}"
            OpenMP::OpenMP_CXX
        )
    else()
        target_link_libraries(DCSsimu PRIVATE
            Qt${QT_VERSION_MAJOR}::Widgets
            Qt${QT_VERSION_MAJOR}::Core
            Qt${QT_VERSION_MAJOR}::Gui
            Qt${QT_VERSION_MAJOR}::OpenGL
            Qt${QT_VERSION_MAJOR}::PrintSupport
            "${OPENGL_LIBRARIES}"
        )
    endif()
endif()



# Keep the original target here, in case we want to build standalone with cmake 
add_executable(DCSsimu_cli ${SOURCES_CLI} ${app_WINRC})

if(CMAKE_CUDA_COMPILER)
    if(OpenMP_CXX_FOUND)
        message(STATUS "Configuring <console application> target for OpenMP and CUDA parallelization")
        target_compile_definitions(DCSsimu_cli PRIVATE -DENTRYPOINT=main -DOPENMP -DCUDA)
    else()
        message(STATUS "Configuring <console application> target for CUDA parallelization")
        target_compile_definitions(DCSsimu_cli PRIVATE -DENTRYPOINT=main -DCUDA)
    endif()
    set_target_properties(DCSsimu_cli PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    
    target_link_options(DCSsimu_cli PRIVATE /NODEFAULTLIB:LIBCMT.LIB)
    target_link_libraries(DCSsimu_cli PRIVATE GPUBin)
else()
    if(OpenMP_CXX_FOUND)
        message(STATUS "Configuring <console application> target for OpenMP parallelization")
        target_compile_definitions(DCSsimu_cli PRIVATE -DENTRYPOINT=main -DOPENMP)
    else()
        message(STATUS "Configuring <console application> target for NO parallelization")
        target_compile_definitions(DCSsimu_cli PRIVATE -DENTRYPOINT=main)
    endif()
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(DCSsimu_cli PRIVATE
        OpenMP::OpenMP_CXX
    )
endif()