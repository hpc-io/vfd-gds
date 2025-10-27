#------------------------------------------------------------------------------ # Include source and build directories
#------------------------------------------------------------------------------
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

#------------------------------------------------------------------------------
# External dependencies
#------------------------------------------------------------------------------
find_package(MPI REQUIRED)
if(MPI_C_FOUND)
  set(HDF5_VFD_GDS_EXT_LIB_DEPENDENCIES
    ${HDF5_VFD_GDS_EXT_LIB_DEPENDENCIES}
    ${MPI_C_LIBRARIES}
  )
  include_directories(
    ${MPI_C_INCLUDE_DIRS}
  )
endif()

#-----------------------------------------------------------------------------
# Define Sources and tests
#-----------------------------------------------------------------------------
set(vfd_gds_tests
  gds_test
)

foreach(vfd_test ${vfd_gds_tests})
  add_executable(${vfd_test}
    ${CMAKE_CURRENT_SOURCE_DIR}/${vfd_test}.c
  )
  target_include_directories(${vfd_test}
    PUBLIC  "$<BUILD_INTERFACE:${HDF5_VFD_GDS_BUILD_INCLUDE_DEPENDENCIES}>"
            $<INSTALL_INTERFACE:${HDF5_VFD_GDS_INSTALL_INCLUDE_INTERFACE}>
  )
  target_include_directories(${vfd_test}
    SYSTEM PUBLIC ${HDF5_VFD_GDS_EXT_INCLUDE_DEPENDENCIES}
  )
  target_link_libraries(${vfd_test}
      ${HDF5_VFD_GDS_EXPORTED_LIBS}
      ${HDF5_VFD_GDS_EXT_LIB_DEPENDENCIES}
      ${HDF5_VFD_GDS_EXT_PKG_DEPENDENCIES}
      CUDA::cudart
  )
endforeach()

#-----------------------------------------------------------------------------
# Tests
#-----------------------------------------------------------------------------
foreach(vfd_test ${vfd_gds_tests})
  add_test(NAME "${vfd_test}"
    COMMAND $<TARGET_FILE:${vfd_test}>
  )
  set_tests_properties(${vfd_test} PROPERTIES
    ENVIRONMENT "HDF5_PLUGIN_PATH=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
  )
endforeach()
