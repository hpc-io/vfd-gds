/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright by The HDF Group.                                               *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of the HDF5 GDS Virtual File Driver. The full copyright *
 * notice, including terms governing use, modification, and redistribution,  *
 * is contained in the COPYING file, which can be found at the root of the   *
 * source code distribution tree.                                            *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/*
 * Programmer:  John Ravi <jjravi@lbl.gov>
 *              Wednesday, July  1, 2020
 *
 * Purpose:	The public header file for the CUDA GPUDirect Storage driver.
 */
#ifndef H5FDgds_H
#define H5FDgds_H

#include <hdf5.h>

#define H5FD_GDS_NAME  "gds"
#define H5FD_GDS_VALUE ((H5FD_class_value_t)(512))

#ifdef __cplusplus
extern "C" {
#endif

/* Default values for memory boundary, file block size, and maximal copy buffer size.
 * Application can set these values through the function H5Pset_fapl_gds. */
#define H5FD_GDS_MBOUNDARY_DEF 4096
#define H5FD_GDS_FBSIZE_DEF    4096
#define H5FD_GDS_CBSIZE_DEF    (16 * 1024 * 1024)

herr_t H5Pset_fapl_gds(hid_t fapl_id, size_t alignment, size_t block_size, size_t cbuf_size);
herr_t H5Pget_fapl_gds(hid_t fapl_id, size_t *boundary /*out*/, size_t *block_size /*out*/,
                       size_t *cbuf_size /*out*/);

#ifdef __cplusplus
}
#endif

#endif
