cmake_minimum_required(VERSION 2.8.12.2 FATAL_ERROR)

#------------------------------------------------------------------------------
# Include source and build directories
#------------------------------------------------------------------------------
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

#-----------------------------------------------------------------------------
# Define Sources
#-----------------------------------------------------------------------------
set(examples
  simple_dset_write
)

foreach (example ${examples})
  add_executable (h5gds_${example}
    ${CMAKE_CURRENT_SOURCE_DIR}/${example}.c
  )
  target_include_directories(h5gds_${example}
    PUBLIC ${CMAKE_SOURCE_DIR}/src
  )
  target_include_directories(h5gds_${example}
    PUBLIC  "$<BUILD_INTERFACE:${HDF5_VFD_GDS_BUILD_INCLUDE_DEPENDENCIES}>"
            $<INSTALL_INTERFACE:${HDF5_VFD_GDS_INSTALL_INCLUDE_INTERFACE}>
  )
  target_include_directories(h5gds_${example}
    SYSTEM PUBLIC ${HDF5_VFD_GDS_EXT_INCLUDE_DEPENDENCIES}
  )
  target_link_libraries(h5gds_${example}
      ${HDF5_VFD_GDS_EXPORTED_LIBS}
      ${HDF5_VFD_GDS_EXT_LIB_DEPENDENCIES}
      ${HDF5_VFD_GDS_EXT_PKG_DEPENDENCIES}
      CUDA::cudart
  )
endforeach()
