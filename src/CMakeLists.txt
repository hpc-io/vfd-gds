#------------------------------------------------------------------------------
# Include source and build directories
#------------------------------------------------------------------------------
set(HDF5_VFD_GDS_BUILD_INCLUDE_DEPENDENCIES
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

#------------------------------------------------------------------------------
# Internal dependencies (exported libs)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# External dependencies
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Set sources
#------------------------------------------------------------------------------
set(HDF5_VFD_GDS_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/H5FDgds.c
)

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

# Clean up system include path first
foreach(item ${HDF5_VFD_GDS_SYSTEM_INCLUDE_PATH})
  if(HDF5_VFD_GDS_EXT_INCLUDE_DEPENDENCIES)
    list(REMOVE_ITEM HDF5_VFD_GDS_EXT_INCLUDE_DEPENDENCIES ${item})
  endif()
endforeach()

# HDF5 GDS VFD
add_library(hdf5_vfd_gds ${HDF5_VFD_GDS_SRCS})
target_include_directories(hdf5_vfd_gds
  PUBLIC  "$<BUILD_INTERFACE:${HDF5_VFD_GDS_BUILD_INCLUDE_DEPENDENCIES}>"
          $<INSTALL_INTERFACE:${HDF5_VFD_GDS_INSTALL_INCLUDE_INTERFACE}>
)
target_include_directories(hdf5_vfd_gds
  SYSTEM PUBLIC ${HDF5_VFD_GDS_EXT_INCLUDE_DEPENDENCIES}
)
target_link_libraries(hdf5_vfd_gds
  ${HDF5_VFD_GDS_EXPORTED_LIBS}
  ${HDF5_VFD_GDS_EXT_LIB_DEPENDENCIES}
  ${HDF5_VFD_GDS_EXT_PKG_DEPENDENCIES}
  )
hdf5_vfd_gds_set_lib_options(hdf5_vfd_gds "hdf5_vfd_gds" ${HDF5_VFD_GDS_LIBTYPE})
if(HDF5_VFD_GDS_ENABLE_COVERAGE)
  set_coverage_flags(hdf5_vfd_gds)
endif()

set(HDF5_VFD_GDS_EXPORTED_LIBS hdf5_vfd_gds ${HDF5_VFD_GDS_EXPORTED_LIBS})
set(HDF5_VFD_GDS_EXPORTED_LIBS ${HDF5_VFD_GDS_EXPORTED_LIBS} PARENT_SCOPE)

#-----------------------------------------------------------------------------
# Specify project header files to be installed
#-----------------------------------------------------------------------------
set(HDF5_VFD_GDS_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/H5FDgds.h
)

#-----------------------------------------------------------------------------
# Add file(s) to CMake Install
#-----------------------------------------------------------------------------
install(
  FILES
    ${HDF5_VFD_GDS_HEADERS}
  DESTINATION
    ${HDF5_VFD_GDS_INSTALL_INCLUDE_DIR}
  COMPONENT
    headers
)

#-----------------------------------------------------------------------------
# Add Target(s) to CMake Install
#-----------------------------------------------------------------------------
install(
  TARGETS
    hdf5_vfd_gds
  EXPORT
    ${HDF5_VFD_GDS_EXPORTED_TARGETS}
  LIBRARY DESTINATION ${HDF5_VFD_GDS_INSTALL_LIB_DIR}
  ARCHIVE DESTINATION ${HDF5_VFD_GDS_INSTALL_LIB_DIR}
  RUNTIME DESTINATION ${HDF5_VFD_GDS_INSTALL_BIN_DIR}
)

#-----------------------------------------------------------------------------
# Add Target(s) to CMake Install for import into other projects
#-----------------------------------------------------------------------------
install(
  EXPORT
    ${HDF5_VFD_GDS_EXPORTED_TARGETS}
  DESTINATION
    ${HDF5_VFD_GDS_INSTALL_DATA_DIR}/cmake/${HDF5_VFD_GDS_PACKAGE}
  FILE
    ${HDF5_VFD_GDS_EXPORTED_TARGETS}.cmake
)

#-----------------------------------------------------------------------------
# Export all exported targets to the build tree for use by parent project
#-----------------------------------------------------------------------------
if(NOT HDF5_VFD_GDS_EXTERNALLY_CONFIGURED)
  export(
    TARGETS
      ${HDF5_VFD_GDS_EXPORTED_LIBS}
    FILE
      ${HDF5_VFD_GDS_EXPORTED_TARGETS}.cmake
  )
# TODO There is a namespace issue with the way HDF5 currently defines targets
#  if(NOT HDF5_VFD_GDS_USE_SYSTEM_HDF5)
#    export(
#      TARGETS
#        ...
#      APPEND FILE
#        ${HDF5_VFD_GDS_EXPORTED_TARGETS}.cmake
#      NAMESPACE hdf5::
#    )
#  endif()
endif()

#------------------------------------------------------------------------------
# Set variables for parent scope
#------------------------------------------------------------------------------

# Pkg-config configuration
if(CMAKE_BUILD_TYPE)
  string(TOLOWER ${CMAKE_BUILD_TYPE} lower_cmake_build_type)
endif()

# HDF5 GDS VFD package dependencies
foreach(pkg_dep ${HDF5_VFD_GDS_EXT_PKG_DEPENDENCIES})
  set(HDF5_VFD_GDS_PKG_DEPENDENCIES "${HDF5_VFD_GDS_PKG_DEPENDENCIES} ${pkg_dep}")
endforeach()
set(HDF5_VFD_GDS_PKG_DEPENDENCIES ${HDF5_VFD_GDS_PKG_DEPENDENCIES} PARENT_SCOPE)

# HDF5 GDS VFD private library dependencies
foreach(exported_lib ${HDF5_VFD_GDS_EXPORTED_LIBS})
  if(lower_cmake_build_type MATCHES "debug")
    get_target_property(HDF5_VFD_GDS_LIBRARY ${exported_lib} OUTPUT_NAME_DEBUG)
  else()
    get_target_property(HDF5_VFD_GDS_LIBRARY ${exported_lib} OUTPUT_NAME_RELEASE)
  endif()
  set(HDF5_VFD_GDS_LIBRARIES "${HDF5_VFD_GDS_LIBRARIES} -l${HDF5_VFD_GDS_LIBRARY}")
endforeach()
set(HDF5_VFD_GDS_LIBRARIES ${HDF5_VFD_GDS_LIBRARIES} PARENT_SCOPE)

# HDF5 GDS VFD external library dependencies
# Need to generate -llib if not already passed
set(HDF5_VFD_GDS_EXT_LIB_DEPENDENCIES
  ${HDF5_VFD_GDS_EXT_LIB_DEPENDENCIES}
  ${HDF5_VFD_GDS_EXT_PKG_LIB_DEPENDENCIES}
  PARENT_SCOPE
)
foreach(lib_dep ${HDF5_VFD_GDS_EXT_LIB_DEPENDENCIES})
  # get library name
  get_filename_component(lib_name ${lib_dep} NAME_WE)
  if(lib_name MATCHES "^-l")
    # lib_name found is -lxxx
    set(HDF5_VFD_GDS_EXT_LIB_DEPENDENCIES_LIST ${HDF5_VFD_GDS_EXT_LIB_DEPENDENCIES_LIST} ${lib_name})
  else()
    # lib_name is /path/to/lib so get library path and name
    get_filename_component(lib_path ${lib_dep} PATH)
    string(REGEX REPLACE "^lib" "" lib_name ${lib_name})
    set(HDF5_VFD_GDS_EXT_LIB_DEPENDENCIES_LIST ${HDF5_VFD_GDS_EXT_LIB_DEPENDENCIES_LIST} -L${lib_path} -l${lib_name})
  endif()
endforeach()
if(HDF5_VFD_GDS_EXT_LIB_DEPENDENCIES_LIST)
  list(REMOVE_DUPLICATES HDF5_VFD_GDS_EXT_LIB_DEPENDENCIES_LIST)
endif()
foreach(lib_dep ${HDF5_VFD_GDS_EXT_LIB_DEPENDENCIES_LIST})
  set(HDF5_VFD_GDS_LIB_DEPENDENCIES "${HDF5_VFD_GDS_LIB_DEPENDENCIES} ${lib_dep}")
endforeach()
set(HDF5_VFD_GDS_LIB_DEPENDENCIES ${HDF5_VFD_GDS_LIB_DEPENDENCIES} PARENT_SCOPE)

# External include dependencies
set(HDF5_VFD_GDS_EXT_INCLUDE_DEPENDENCIES
  ${HDF5_VFD_GDS_EXT_INCLUDE_DEPENDENCIES}
  ${HDF5_VFD_GDS_EXT_PKG_INCLUDE_DEPENDENCIES}
  PARENT_SCOPE
)
if(HDF5_VFD_GDS_EXT_INCLUDE_DEPENDENCIES)
  list(REMOVE_DUPLICATES HDF5_VFD_GDS_EXT_INCLUDE_DEPENDENCIES)
endif()
foreach(inc_dep ${HDF5_VFD_GDS_EXT_INCLUDE_DEPENDENCIES})
  set(HDF5_VFD_GDS_INCLUDE_DEPENDENCIES "${HDF5_VFD_GDS_INCLUDE_DEPENDENCIES} -I${inc_dep}")
endforeach()
set(HDF5_VFD_GDS_INCLUDE_DEPENDENCIES ${HDF5_VFD_GDS_INCLUDE_DEPENDENCIES} PARENT_SCOPE)
