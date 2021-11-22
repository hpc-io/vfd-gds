cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
# cmake_minimum_required(VERSION 2.8.12.2 FATAL_ERROR)

# Setup cmake policies.
foreach(policy
    CMP0012
    CMP0013
    CMP0014
    CMP0022 # CMake 2.8.12
    CMP0025 # CMake 3.0
    CMP0053 # CMake 3.1
    CMP0054 # CMake 3.1
    CMP0074 # CMake 3.12
    CMP0075 # CMake 3.12
    CMP0083 # CMake 3.14
    CMP0093 # CMake 3.15
  )
  if(POLICY ${policy})
    cmake_policy(SET ${policy} NEW)
  endif()
endforeach()

# Set a consistent MACOSX_RPATH default across all CMake versions.
# When CMake 2.8.12 is required, change this default to 1.
# When CMake 3.0.0 is required, remove this block (see CMP0042).
if(NOT DEFINED CMAKE_MACOSX_RPATH)
  set(CMAKE_MACOSX_RPATH 0)
endif()

# project(HDF5_VFD_GDS C)
project(HDF5_VFD_GDS LANGUAGES CUDA C CXX)


#------------------------------------------------------------------------------
# Setup install and output Directories
#------------------------------------------------------------------------------
if(NOT HDF5_VFD_GDS_INSTALL_BIN_DIR)
  set(HDF5_VFD_GDS_INSTALL_BIN_DIR ${CMAKE_INSTALL_PREFIX}/bin)
endif()
if(NOT HDF5_VFD_GDS_INSTALL_LIB_DIR)
  set(HDF5_VFD_GDS_INSTALL_LIB_DIR ${CMAKE_INSTALL_PREFIX}/lib)
endif()
if(NOT HDF5_VFD_GDS_INSTALL_INCLUDE_DIR)
  # Interface include will default to prefix/include
  set(HDF5_VFD_GDS_INSTALL_INTERFACE include)
  set(HDF5_VFD_GDS_INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include)
else()
  set(HDF5_VFD_GDS_INSTALL_INTERFACE ${HDF5_VFD_GDS_INSTALL_INCLUDE_DIR})
endif()
if(NOT HDF5_VFD_GDS_INSTALL_DATA_DIR)
  set(HDF5_VFD_GDS_INSTALL_DATA_DIR ${CMAKE_INSTALL_PREFIX}/share)
endif()

# Setting this ensures that "make install" will leave rpaths to external
# libraries intact on "make install". This ensures that one can install a
# version of the HDF5 GDS VFD on the build machine without any issues. If
# thisis  not desired, simply specify CMAKE_INSTALL_RPATH_USE_LINK_PATH when
# configuring the HDF5 GDS VFD and "make install" will strip all rpaths, which
# is default behavior.
if(NOT CMAKE_INSTALL_RPATH_USE_LINK_PATH)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

#------------------------------------------------------------------------------
# Set module path
#------------------------------------------------------------------------------
set(HDF5_VFD_GDS_CMAKE_MODULE_PATH "${HDF5_VFD_GDS_SOURCE_DIR}/CMake")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${HDF5_VFD_GDS_CMAKE_MODULE_PATH})

#------------------------------------------------------------------------------
# Locate CUDA
#------------------------------------------------------------------------------
find_package(Threads REQUIRED)
find_package(CUDA REQUIRED)

# Set Cufile installation directory to CUDA installation directory by default
set (HDF5_VFD_GDS_CUFILE_DIR ${CUDA_TOOLKIT_ROOT_DIR} CACHE PATH "Cufile installation directory for Nvidia GDS support")

# Try to locate cufile library
find_library (HDF5_VFD_GDS_CUFILE_LIB
  NAMES
    cufile
  HINTS
    "${HDF5_VFD_GDS_CUFILE_DIR}/lib"
    "${HDF5_VFD_GDS_CUFILE_DIR}/lib64"
  REQUIRED
)

set(HDF5_VFD_GDS_EXT_INCLUDE_DEPENDENCIES
  ${HDF5_VFD_GDS_EXT_INCLUDE_DEPENDENCIES}
  ${CUDA_INCLUDE_DIRS}
)
set(HDF5_VFD_GDS_EXT_LIB_DEPENDENCIES
  ${HDF5_VFD_GDS_EXT_LIB_DEPENDENCIES}
  ${CUDA_LIBRARIES}
  ${HDF5_VFD_GDS_CUFILE_LIB}
)

#------------------------------------------------------------------------------
# Locate HDF5
#------------------------------------------------------------------------------
find_package(HDF5 1.13.0 NO_MODULE NAMES hdf5 COMPONENTS C shared)
if(HDF5_FOUND)
  set(HDF5_C_SHARED_LIBRARY hdf5-shared)
  if(NOT TARGET ${HDF5_C_SHARED_LIBRARY})
      message(FATAL_ERROR "Could not find hdf5 shared target, please make "
      "sure that HDF5 has been compiled with shared libraries enabled.")
  endif()
  set(HDF5_VFD_GDS_EXT_PKG_DEPENDENCIES
    ${HDF5_VFD_GDS_EXT_PKG_DEPENDENCIES}
    ${HDF5_C_SHARED_LIBRARY}
  )
  set(HDF5_VFD_GDS_USE_SYSTEM_CMAKE_HDF5 1)
else()
  # Allow for HDF5 autotools builds
  find_package(HDF5 1.13.0 MODULE REQUIRED)
  set(HDF5_VFD_GDS_EXT_INCLUDE_DEPENDENCIES
    ${HDF5_VFD_GDS_EXT_INCLUDE_DEPENDENCIES}
    ${HDF5_INCLUDE_DIRS}
  )
  set(HDF5_VFD_GDS_EXT_LIB_DEPENDENCIES
    ${HDF5_VFD_GDS_EXT_LIB_DEPENDENCIES}
    ${HDF5_LIBRARIES}
  )
  set(HDF5_VFD_GDS_USE_SYSTEM_CMAKE_HDF5 0)
endif()

#------------------------------------------------------------------------------
# Locate MPI - currently just used for building VFD's tests
#------------------------------------------------------------------------------
if(HDF5_FOUND AND HDF5_IS_PARALLEL)
  find_package(MPI REQUIRED COMPONENTS C)
  set(HDF5_VFD_GDS_EXT_INCLUDE_DEPENDENCIES
    ${HDF5_VFD_GDS_EXT_INCLUDE_DEPENDENCIES}
    ${MPI_C_INCLUDE_DIRS}
  )
  set(HDF5_VFD_GDS_EXT_LIB_DEPENDENCIES
    ${HDF5_VFD_GDS_EXT_LIB_DEPENDENCIES}
    ${MPI_C_LIBRARIES}
  )
endif()

#------------------------------------------------------------------------------
# Version information
#------------------------------------------------------------------------------
include(${HDF5_VFD_GDS_CMAKE_MODULE_PATH}/Git/Git.cmake)
include(Hdf5GdsVfdDetermineVersion)
# Hardcoded version variables are read-in from a separate file. This makes it
# easier to have a script to update version numbers automatically.
file(STRINGS version.txt version_txt)
extract_version_components("${version_txt}" "${PROJECT_NAME}")
determine_version(${HDF5_VFD_GDS_SOURCE_DIR} ${GIT_EXECUTABLE} "${PROJECT_NAME}")
set(HDF5_VFD_GDS_PACKAGE "hdf5_vfd_gds")
set(HDF5_VFD_GDS_PACKAGE_NAME "HDF5_VFD_GDS")
set(HDF5_VFD_GDS_PACKAGE_DESCRIPTION "HDF5 Nvidia GPUDirect Storage VFD")
set(HDF5_VFD_GDS_PACKAGE_URL "https://github.com/hpc-io/vfd-gds")
set(HDF5_VFD_GDS_PACKAGE_VENDOR "The HDF Group")
message(STATUS "Configuring ${HDF5_VFD_GDS_PACKAGE} v${HDF5_VFD_GDS_VERSION_FULL}")

#------------------------------------------------------------------------------
# Setup CMake Environment
#------------------------------------------------------------------------------
if(APPLE AND NOT HDF5_VFD_GDS_EXTERNALLY_CONFIGURED)
  # We are doing a unix-style install i.e. everything will be installed in
  # CMAKE_INSTALL_PREFIX/bin and CMAKE_INSTALL_PREFIX/lib etc. as on other unix
  # platforms. We still need to setup CMAKE_INSTALL_NAME_DIR correctly so that
  # the binaries point to appropriate location for the libraries.

  # 1. Make CMAKE_INSTALL_PREFIX publicly accessible, if it was hidden in
  #    previous pass
  get_property(is_internal CACHE CMAKE_INSTALL_PREFIX PROPERTY TYPE)
  if(is_internal STREQUAL "INTERNAL")
    set(CMAKE_INSTALL_PREFIX ${CACHED_CMAKE_INSTALL_PREFIX} CACHE PATH "Install prefix" FORCE)
  else()
    set(CMAKE_INSTALL_PREFIX ${CACHED_CMAKE_INSTALL_PREFIX} CACHE PATH "Install prefix")
  endif()
  unset(MACOSX_APP_INSTALL_PREFIX CACHE)

  set(CMAKE_INSTALL_NAME_DIR "@rpath")
  mark_as_advanced(
    CMAKE_OSX_ARCHITECTURES
    CMAKE_OSX_DEPLOYMENT_TARGET
    CMAKE_OSX_SYSROOT
    )
else()
  if(WIN32)
    message("The HDF5 GDS VFD is not supported on this platform." FATAL_ERROR)
  endif()
endif()

#------------------------------------------------------------------------------
if(NOT HDF5_VFD_GDS_EXTERNALLY_CONFIGURED)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
      ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all Executables."
  )
  set(EXECUTABLE_OUTPUT_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
      ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all Libraries"
  )
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
      ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all static libraries."
  )
endif()

#------------------------------------------------------------------------------
# Disallow in-source build
#------------------------------------------------------------------------------
if("${HDF5_VFD_GDS_SOURCE_DIR}" STREQUAL "${HDF5_VFD_GDS_BINARY_DIR}")
  message(FATAL_ERROR
    "HDF5 GDS VFD requires an out of source Build. "
    "Please create a separate binary directory and run CMake there.")
endif()

#------------------------------------------------------------------------------
# Set a default build type if none was specified
#------------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

if(NOT CMAKE_C_FLAGS AND CMAKE_COMPILER_IS_GNUCC)
  message(STATUS "GCC detected, setting additional flags")
  set(CMAKE_C_FLAGS "-Wall -Wextra -Winline -Wcast-qual -std=gnu99 -Wshadow" CACHE STRING "Flags used by the compiler during all build types." FORCE)
endif()

#-----------------------------------------------------------------------------
# Targets built within this project are exported at Install time for use
# by other projects.
#-----------------------------------------------------------------------------
if(NOT HDF5_VFD_GDS_EXPORTED_TARGETS)
  set(HDF5_VFD_GDS_EXPORTED_TARGETS "${HDF5_VFD_GDS_PACKAGE}-targets")
endif()

#------------------------------------------------------------------------------
# Choose static or shared libraries.
#------------------------------------------------------------------------------
option(BUILD_SHARED_LIBS "Build with shared libraries." ON)
if(BUILD_SHARED_LIBS)
  set(H5FD_GDS_VFD_BUILD_SHARED_LIBS 1)
  set(HDF5_VFD_GDS_LIBTYPE SHARED)
else()
  message(FATAL_ERROR
    "HDF5 GDS VFD requires shared libraries.")
  set(H5FD_GDS_VFD_BUILD_SHARED_LIBS 0)
  set(HDF5_VFD_GDS_LIBTYPE STATIC)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#-------------------------------------------------------------------------------
function(hdf5_vfd_gds_set_lib_options libtarget libname libtype)
  if(${libtype} MATCHES "SHARED")
    set(LIB_RELEASE_NAME "${libname}")
    set(LIB_DEBUG_NAME "${libname}_debug")
    set(LIB_VERSION ${HDF5_VFD_GDS_VERSION}.${HDF5_VFD_GDS_VERSION_PATCH})
    set(API_VERSION ${HDF5_VFD_GDS_VERSION_MAJOR})
  else()
    # if the generator supports configuration types or if the CMAKE_BUILD_TYPE has a value
    if(CMAKE_CONFIGURATION_TYPES OR CMAKE_BUILD_TYPE)
      set(LIB_RELEASE_NAME "${libname}")
      set(LIB_DEBUG_NAME "${libname}_debug")
    else()
      set(LIB_RELEASE_NAME "lib${libname}")
      set(LIB_DEBUG_NAME "lib${libname}_debug")
    endif()
  endif()

  set_target_properties(${libtarget}
      PROPERTIES
      OUTPUT_NAME_DEBUG          ${LIB_DEBUG_NAME}
      OUTPUT_NAME_RELEASE        ${LIB_RELEASE_NAME}
      OUTPUT_NAME_MINSIZEREL     ${LIB_RELEASE_NAME}
      OUTPUT_NAME_RELWITHDEBINFO ${LIB_RELEASE_NAME}
      OUTPUT_NAME_ASAN           ${LIB_DEBUG_NAME}
      OUTPUT_NAME_UBSAN          ${LIB_DEBUG_NAME}
      VERSION                    ${LIB_VERSION}
      SOVERSION                  ${API_VERSION}
  )
endfunction()

# Avoid explicitly including system include paths
set(HDF5_VFD_GDS_SYSTEM_INCLUDE_PATH ${CMAKE_SYSTEM_INCLUDE_PATH}
  ${CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES})
  set(HDF5_VFD_GDS_SYSTEM_INCLUDE_PATH /usr/include ${HDF5_VFD_GDS_SYSTEM_INCLUDE_PATH})

#-----------------------------------------------------------------------------
# Coverage
#-----------------------------------------------------------------------------
if(NOT HDF5_VFD_GDS_EXTERNALLY_CONFIGURED)
  option(HDF5_VFD_GDS_ENABLE_COVERAGE "Enable coverage." OFF)
  if(HDF5_VFD_GDS_ENABLE_COVERAGE)
    set(COVERAGE_FLAGS "-fprofile-arcs -ftest-coverage" CACHE STRING
      "Flags to the coverage program that CTest uses to perform coverage inspection"
    )
    mark_as_advanced(COVERAGE_FLAGS)
  endif()

  macro(set_coverage_flags target)
    set_target_properties(${target}
      PROPERTIES
        COMPILE_FLAGS ${COVERAGE_FLAGS}
        LINK_FLAGS ${COVERAGE_FLAGS}
    )
  endmacro()
endif()

#-----------------------------------------------------------------------------
# Source
#-----------------------------------------------------------------------------
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)

#-----------------------------------------------------------------------------
# Build doxygen documentation.
#-----------------------------------------------------------------------------
option(BUILD_DOCUMENTATION "Build documentation." ON)
#if(BUILD_DOCUMENTATION)
#  add_subdirectory(Documentation/Doxygen)
#endif()

#-----------------------------------------------------------------------------
# Examples
#-----------------------------------------------------------------------------
option(BUILD_EXAMPLES "Build examples." ON)
if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

#-----------------------------------------------------------------------------
# Testing
#-----------------------------------------------------------------------------
option(BUILD_TESTING "Build testing." ON)
if(NOT HDF5_VFD_GDS_EXTERNALLY_CONFIGURED AND BUILD_TESTING)
  enable_testing()
  include(CTest)
  add_subdirectory(test)
endif()

#-----------------------------------------------------------------------------
# Configure the config.cmake file for the build directory
#-----------------------------------------------------------------------------
set(HDF5_VFD_GDS_CONFIG_INSTALLED FALSE)
configure_file(
  ${HDF5_VFD_GDS_SOURCE_DIR}/CMake/${HDF5_VFD_GDS_PACKAGE}-config.cmake.in
  ${HDF5_VFD_GDS_BINARY_DIR}/${HDF5_VFD_GDS_PACKAGE}-config.cmake @ONLY
)

#-----------------------------------------------------------------------------
# Configure the config.cmake file for the install directory
#-----------------------------------------------------------------------------
set(HDF5_VFD_GDS_CONFIG_INSTALLED TRUE)
configure_file(
  ${HDF5_VFD_GDS_SOURCE_DIR}/CMake/${HDF5_VFD_GDS_PACKAGE}-config.cmake.in
  ${HDF5_VFD_GDS_BINARY_DIR}/CMakeFiles/${HDF5_VFD_GDS_PACKAGE}-config.cmake @ONLY
)

install(
  FILES
    ${HDF5_VFD_GDS_BINARY_DIR}/CMakeFiles/${HDF5_VFD_GDS_PACKAGE}-config.cmake
  DESTINATION
    ${HDF5_VFD_GDS_INSTALL_DATA_DIR}/cmake/${HDF5_VFD_GDS_PACKAGE}
)

#-----------------------------------------------------------------------------
# Configure the config-version.cmake file for the install directory
#-----------------------------------------------------------------------------
configure_file(
  ${HDF5_VFD_GDS_SOURCE_DIR}/CMake/${HDF5_VFD_GDS_PACKAGE}-config-version.cmake.in
  ${HDF5_VFD_GDS_BINARY_DIR}/CMakeFiles/${HDF5_VFD_GDS_PACKAGE}-config-version.cmake @ONLY
)

install(
  FILES
    ${HDF5_VFD_GDS_BINARY_DIR}/CMakeFiles/${HDF5_VFD_GDS_PACKAGE}-config-version.cmake
  DESTINATION
    ${HDF5_VFD_GDS_INSTALL_DATA_DIR}/cmake/${HDF5_VFD_GDS_PACKAGE}
)

#-----------------------------------------------------------------------------
# For automake compatibility, also provide a pkgconfig file
#-----------------------------------------------------------------------------
configure_file(
  ${HDF5_VFD_GDS_SOURCE_DIR}/CMake/${HDF5_VFD_GDS_PACKAGE}.pc.in
  ${HDF5_VFD_GDS_BINARY_DIR}/CMakeFiles/${HDF5_VFD_GDS_PACKAGE}.pc @ONLY
)

install(
  FILES
    ${HDF5_VFD_GDS_BINARY_DIR}/CMakeFiles/${HDF5_VFD_GDS_PACKAGE}.pc
  DESTINATION
    ${HDF5_VFD_GDS_INSTALL_LIB_DIR}/pkgconfig
)

#-----------------------------------------------------------------------------
# CPack
#-----------------------------------------------------------------------------
if(NOT HDF5_VFD_GDS_EXTERNALLY_CONFIGURED)
  set(CPACK_PACKAGE_DESCRIPTION_FILE ${HDF5_VFD_GDS_SOURCE_DIR}/README.md)
  set(CPACK_RESOURCE_FILE_LICENSE ${HDF5_VFD_GDS_SOURCE_DIR}/COPYING)
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${HDF5_VFD_GDS_PACKAGE_DESCRIPTION})
  set(CPACK_PACKAGE_NAME ${HDF5_VFD_GDS_PACKAGE_NAME})
  set(CPACK_PACKAGE_VENDOR ${HDF5_VFD_GDS_PACKAGE_VENDOR})
  set(CPACK_PACKAGE_VERSION_MAJOR ${HDF5_VFD_GDS_VERSION_MAJOR})
  set(CPACK_PACKAGE_VERSION_MINOR ${HDF5_VFD_GDS_VERSION_MINOR})
  set(CPACK_PACKAGE_VERSION_PATCH ${HDF5_VFD_GDS_VERSION_PATCH})
  set(CPACK_GENERATOR "TBZ2")
  set(CPACK_SOURCE_PACKAGE_FILE_NAME ${HDF5_VFD_GDS_PACKAGE}-${HDF5_VFD_GDS_VERSION_FULL})
  set(CPACK_SOURCE_IGNORE_FILES ".git*;/GitSetup/;/.git/;.swp$;.#;/#;.*~")
  set(CPACK_SOURCE_STRIP_FILES "")
  include(CPack)
endif()
