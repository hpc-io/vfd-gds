/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Board of Trustees of the University of Illinois.         *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of HDF5.  The full HDF5 copyright notice, including     *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the root of the source code       *
 * distribution tree, or in https://support.hdfgroup.org/ftp/HDF5/releases.  *
 * If you do not have access to either file, you may request a copy from     *
 * help@hdfgroup.org.                                                        *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/*
 * Programmer:  John Ravi <jjravi@lbl.gov>
 *              Wednesday, July  1, 2020
 *
 * Purpose:	The public header file for the CUDA GPUDirect Storage driver.
 */
#ifndef H5FDgds_H
#define H5FDgds_H

#ifdef H5_GDS_SUPPORT
#       define H5FD_GDS	    (H5FD_gds_init())
#else
#       define H5FD_GDS	        (-1)
#endif /* H5_GDS_SUPPORT */

#define check_cudadrivercall(fn) \
  { \
    CUresult res = fn; \
    if (res != CUDA_SUCCESS) { \
      const char *str = nullptr; \
      cuGetErrorName(res, &str); \
      fprintf(stderr, "cuda driver api call failed %d, %d : %s\n", fn, \
        __LINE__, str); \
      fprintf(stderr, "EXITING program!!!\n"); \
      exit(1); \
    } \
  }

#define check_cudaruntimecall(fn) \
  { \
    cudaError_t res = fn; \
    if (res != cudaSuccess) { \
      const char *str = cudaGetErrorName(res); \
      fprintf(stderr, "cuda runtime api call failed %d, %d : %s\n", fn, \
        __LINE__, str); \
      fprintf(stderr, "EXITING program!!!\n"); \
      exit(1); \
    } \
  }

#ifdef H5_GDS_SUPPORT
#ifdef __cplusplus
extern "C" {
#endif

  /* Default values for memory boundary, file block size, and maximal copy buffer size.
   * Application can set these values through the function H5Pset_fapl_gds. */
#define MBOUNDARY_DEF		4096
#define FBSIZE_DEF		4096
#define CBSIZE_DEF		16*1024*1024

  H5_DLL hid_t H5FD_gds_init(void);
  H5_DLL herr_t H5Pset_fapl_gds(hid_t fapl_id, size_t alignment, size_t block_size,
    size_t cbuf_size);
  H5_DLL herr_t H5Pget_fapl_gds(hid_t fapl_id, size_t *boundary/*out*/,
    size_t *block_size/*out*/, size_t *cbuf_size/*out*/);

#ifdef __cplusplus
}
#endif

#endif /* H5_GDS_SUPPORT */

#endif

