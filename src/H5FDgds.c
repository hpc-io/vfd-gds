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
 * Purpose: Interfaces with the CUDA GPUDirect Storage API
 *          Based on the Direct I/O file driver which forces the data to be written to
 *          the file directly without being copied into system kernel
 *          buffer.  The main system support this feature is Linux.
 */

#include <fcntl.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>
#include <sys/file.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufile.h>

#include <pthread.h>

#include "hdf5.h"

/* HDF5 header for dynamic plugin loading */
#include "H5PLextern.h"

#include "H5FDgds.h"     /* cuda gds file driver     */
#include "H5FDgds_err.h" /* error handling           */

#define H5FD_GDS (H5FD_gds_init())

/* HDF5 doesn't currently have a driver init callback. Use
 * macro to initialize driver if loaded as a plugin.
 */
#define H5FD_GDS_INIT          \
do {                           \
    if (H5FD_GDS_g < 0)        \
        H5FD_GDS_g = H5FD_GDS; \
} while(0)

/* #define ADVISE_OS_DISABLE_READ_CACHE */

#ifdef ADVISE_OS_DISABLE_READ_CACHE
#include <fcntl.h>
#endif /* ADVISE_OS_DISABLE_READ_CACHE */

typedef struct thread_data_t {
    union {
        void *      rd_devPtr; /* read device address */
        const void *wr_devPtr; /* write device address */
    };
    int            fd;
    CUfileHandle_t cfr_handle;    /* cuFile Handle */
    off_t          offset;        /* File offset */
    off_t          devPtr_offset; /* device address offset */
    size_t         block_size;    /* I/O chunk size */
    size_t         size;          /* Read/Write size */
} thread_data_t;

static bool cu_file_driver_opened = false;

/* static bool reg_once = false; */

/* The driver identification number, initialized at runtime */
static hid_t H5FD_GDS_g = H5I_INVALID_HID;

/* Identifiers for HDF5's error API */
hid_t H5FDgds_err_stack_g = H5I_INVALID_HID;
hid_t H5FDgds_err_class_g = H5I_INVALID_HID;

/* Whether to ignore file locks when disabled (env var value) */
static htri_t ignore_disabled_file_locks_s = FAIL;

/* File operations */
#define OP_UNKNOWN 0
#define OP_READ    1
#define OP_WRITE   2

/* POSIX I/O mode used as the third parameter to open/_open
 * when creating a new file (O_CREAT is set).
 */
#if defined(H5_HAVE_WIN32_API)
#define H5FD_GDS_POSIX_CREATE_MODE_RW (_S_IREAD | _S_IWRITE)
#else
#define H5FD_GDS_POSIX_CREATE_MODE_RW 0666
#endif

/* Driver-specific file access properties */
typedef struct H5FD_gds_fapl_t {
    size_t  mboundary;  /* Memory boundary for alignment    */
    size_t  fbsize;     /* File system block size      */
    size_t  cbsize;     /* Maximal buffer size for copying user data  */
    hbool_t must_align; /* Decides if data alignment is required        */
} H5FD_gds_fapl_t;

/*
 * The description of a file belonging to this driver. The `eoa' and `eof'
 * determine the amount of hdf5 address space in use and the high-water mark
 * of the file (the current size of the underlying Unix file). The `pos'
 * value is used to eliminate file position updates when they would be a
 * no-op. Unfortunately we've found systems that use separate file position
 * indicators for reading and writing so the lseek can only be eliminated if
 * the current operation is the same as the previous operation.  When opening
 * a file the `eof' will be set to the current file size, `eoa' will be set
 * to zero, `pos' will be set to H5F_ADDR_UNDEF (as it is when an error
 * occurs), and `op' will be set to H5F_OP_UNKNOWN.
 */
typedef struct H5FD_gds_t {
    H5FD_t          pub; /*public stuff, must be first  */
    int             fd;  /*the unix file      */
    haddr_t         eoa; /*end of allocated region  */
    haddr_t         eof; /*end of file; current file size*/
    haddr_t         pos; /*current file I/O position  */
    int             op;  /*last operation    */
    H5FD_gds_fapl_t fa;  /*file access properties  */
    hbool_t         ignore_disabled_file_locks;

    CUfileHandle_t cf_handle;      /* cufile handle */
    int            num_io_threads; /* number of io threads for cufile */
    size_t         io_block_size;  /* io block size or cufile */
    pthread_t *    threads;
    thread_data_t *td;

#ifndef H5_HAVE_WIN32_API
    /*
     * On most systems the combination of device and i-node number uniquely
     * identify a file.
     */
    dev_t device; /*file device number    */
    ino_t inode;  /*file i-node number    */
#else
    /*
     * On H5_HAVE_WIN32_API the low-order word of a unique identifier associated with the
     * file and the volume serial number uniquely identify a file. This number
     * (which, both? -rpm) may change when the system is restarted or when the
     * file is opened. After a process opens a file, the identifier is
     * constant until the file is closed. An application can use this
     * identifier and the volume serial number to determine whether two
     * handles refer to the same file.
     */
    DWORD fileindexlo;
    DWORD fileindexhi;
#endif

} H5FD_gds_t;

/* multiple threads for one io request */
static void *
read_thread_fn(void *data)
{
    ssize_t        ret;
    thread_data_t *td = (thread_data_t *)data;

    /*
     * fprintf(stderr, "read thread -- ptr: %p, size: %lu, foffset: %ld, doffset: %ld\n",
     * td->rd_devPtr, td->size, td->offset, td->devPtr_offset);
     */

    while (td->size > 0) {
        if (td->size > td->block_size) {
            ret = cuFileRead(td->cfr_handle, td->rd_devPtr, td->block_size, td->offset, td->devPtr_offset);
            td->offset += td->block_size;
            td->devPtr_offset += td->block_size;
            td->size -= td->block_size;
        }
        else {
            ret      = cuFileRead(td->cfr_handle, td->rd_devPtr, td->size, td->offset, td->devPtr_offset);
            td->size = 0;
        }
        assert(ret > 0);
    }

    /*
     * fprintf(stderr, "read success thread -- ptr: %p, size: %lu, foffset: %ld, doffset: %ld\n",
     * td->rd_devPtr, td->size, td->offset, td->devPtr_offset);
     */

    return NULL;
}

static void *
write_thread_fn(void *data)
{
    ssize_t        ret;
    thread_data_t *td = (thread_data_t *)data;

    /*
     * fprintf(stderr, "wrt thread -- ptr: %p, size: %lu, foffset: %ld, doffset: %ld\n",
     * td->wr_devPtr, td->size, td->offset, td->devPtr_offset);
     */

    while (td->size > 0) {
        if (td->size > td->block_size) {
            ret = cuFileWrite(td->cfr_handle, td->wr_devPtr, td->block_size, td->offset, td->devPtr_offset);
            td->offset += td->block_size;
            td->devPtr_offset += td->block_size;
            td->size -= td->block_size;
        }
        else {
            ret      = cuFileWrite(td->cfr_handle, td->wr_devPtr, td->size, td->offset, td->devPtr_offset);
            td->size = 0;
        }
        assert(ret > 0);
    }

    /*
     * printf("wrt success thread -- ptr: %p, size: %lu, foffset: %ld, doffset: %ld\n",
     * td->wr_devPtr, td->size, td->offset, td->devPtr_offset);
     */

    return NULL;
}

/* end multiple threads for one io request */

/*
 * These macros check for overflow of various quantities.  These macros
 * assume that off_t is signed and haddr_t and size_t are unsigned.
 *
 * ADDR_OVERFLOW:  Checks whether a file address of type `haddr_t'
 *      is too large to be represented by the second argument
 *      of the file seek function.
 *
 * SIZE_OVERFLOW:  Checks whether a buffer size of type `hsize_t' is too
 *      large to be represented by the `size_t' type.
 *
 * REGION_OVERFLOW:  Checks whether an address and size pair describe data
 *      which can be addressed entirely by the second
 *      argument of the file seek function.
 */
#define MAXADDR          (((haddr_t)1 << (8 * sizeof(off_t) - 1)) - 1)
#define ADDR_OVERFLOW(A) (HADDR_UNDEF == (A) || ((A) & ~(haddr_t)MAXADDR))
#define SIZE_OVERFLOW(Z) ((Z) & ~(hsize_t)MAXADDR)
#define REGION_OVERFLOW(A, Z)                                                                                \
    (ADDR_OVERFLOW(A) || SIZE_OVERFLOW(Z) || HADDR_UNDEF == (A) + (Z) || (off_t)((A) + (Z)) < (off_t)(A))


#define check_cudadrivercall(fn)                                                                             \
    {                                                                                                        \
        CUresult res = fn;                                                                                   \
        if (res != CUDA_SUCCESS) {                                                                           \
            const char *str = nullptr;                                                                       \
            cuGetErrorName(res, &str);                                                                       \
            fprintf(stderr, "cuda driver api call failed %d, %d : %s\n", fn, __LINE__, str);                 \
            fprintf(stderr, "EXITING program!!!\n");                                                         \
            exit(1);                                                                                         \
        }                                                                                                    \
    }

#define check_cudaruntimecall(fn)                                                                            \
    {                                                                                                        \
        cudaError_t res = fn;                                                                                \
        if (res != cudaSuccess) {                                                                            \
            const char *str = cudaGetErrorName(res);                                                         \
            fprintf(stderr, "cuda runtime api call failed %d, %d : %s\n", fn, __LINE__, str);                \
            fprintf(stderr, "EXITING program!!!\n");                                                         \
            exit(1);                                                                                         \
        }                                                                                                    \
    }

/* Prototypes */
static hid_t   H5FD_gds_init(void);
static herr_t  H5FD__gds_term(void);
static herr_t  H5FD__gds_populate_config(size_t boundary, size_t block_size, size_t cbuf_size,
                                         H5FD_gds_fapl_t *fa_out);
static void *  H5FD__gds_fapl_get(H5FD_t *file);
static void *  H5FD__gds_fapl_copy(const void *_old_fa);
static H5FD_t *H5FD__gds_open(const char *name, unsigned flags, hid_t fapl_id, haddr_t maxaddr);
static herr_t  H5FD__gds_close(H5FD_t *_file);
static int     H5FD__gds_cmp(const H5FD_t *_f1, const H5FD_t *_f2);
static herr_t  H5FD__gds_query(const H5FD_t *_f1, unsigned long *flags);
static haddr_t H5FD__gds_get_eoa(const H5FD_t *_file, H5FD_mem_t type);
static herr_t  H5FD__gds_set_eoa(H5FD_t *_file, H5FD_mem_t type, haddr_t addr);
static haddr_t H5FD__gds_get_eof(const H5FD_t *_file, H5FD_mem_t type);
static herr_t  H5FD__gds_get_handle(H5FD_t *_file, hid_t fapl, void **file_handle);
static herr_t  H5FD__gds_read(H5FD_t *_file, H5FD_mem_t type, hid_t fapl_id, haddr_t addr, size_t size,
                              void *buf);
static herr_t  H5FD__gds_write(H5FD_t *_file, H5FD_mem_t type, hid_t fapl_id, haddr_t addr, size_t size,
                               const void *buf);
static herr_t  H5FD__gds_flush(H5FD_t *_file, hid_t dxpl_id, hbool_t closing);
static herr_t  H5FD__gds_truncate(H5FD_t *_file, hid_t dxpl_id, hbool_t closing);
static herr_t  H5FD__gds_lock(H5FD_t *_file, hbool_t rw);
static herr_t  H5FD__gds_unlock(H5FD_t *_file);
static herr_t  H5FD__gds_delete(const char *filename, hid_t fapl_id);
static herr_t  H5FD__gds_ctl(H5FD_t *_file, uint64_t op_code, uint64_t flags, const void *input,
                              void **output);

static const H5FD_class_t H5FD_gds_g = {
    H5FD_GDS_VALUE,          /* value                */
    H5FD_GDS_NAME,           /* name                 */
    MAXADDR,                 /* maxaddr              */
    H5F_CLOSE_WEAK,          /* fc_degree            */
    H5FD__gds_term,          /* terminate            */
    NULL,                    /* sb_size              */
    NULL,                    /* sb_encode            */
    NULL,                    /* sb_decode            */
    sizeof(H5FD_gds_fapl_t), /* fapl_size            */
    H5FD__gds_fapl_get,      /* fapl_get             */
    H5FD__gds_fapl_copy,     /* fapl_copy            */
    NULL,                    /* fapl_free            */
    0,                       /* dxpl_size            */
    NULL,                    /* dxpl_copy            */
    NULL,                    /* dxpl_free            */
    H5FD__gds_open,          /* open                 */
    H5FD__gds_close,         /* close                */
    H5FD__gds_cmp,           /* cmp                  */
    H5FD__gds_query,         /* query                */
    NULL,                    /* get_type_map         */
    NULL,                    /* alloc                */
    NULL,                    /* free                 */
    H5FD__gds_get_eoa,       /* get_eoa              */
    H5FD__gds_set_eoa,       /* set_eoa              */
    H5FD__gds_get_eof,       /* get_eof              */
    H5FD__gds_get_handle,    /* get_handle           */
    H5FD__gds_read,          /* read                 */
    H5FD__gds_write,         /* write                */
    H5FD__gds_flush,         /* flush                */
    H5FD__gds_truncate,      /* truncate             */
    H5FD__gds_lock,          /* lock                 */
    H5FD__gds_unlock,        /* unlock               */
    H5FD__gds_delete,        /* delete               */
    H5FD__gds_ctl,           /* ctl                  */
    H5FD_FLMAP_DICHOTOMY     /* fl_map               */
};

/*-------------------------------------------------------------------------
 * Function:    H5FD_gds_init
 *
 * Purpose:     Initialize this driver by registering the driver with the
 *              library.
 *
 * Return:      Success:    The driver ID for the gds driver
 *              Failure:    H5I_INVALID_HID
 *
 * Programmer:  John J Ravi
 *              Tuesday, 06 October 2020
 *
 *-------------------------------------------------------------------------
 */
static hid_t
H5FD_gds_init(void)
{
    CUfileError_t status;
    char *        lock_env_var = NULL; /* Environment variable pointer */
    hid_t         ret_value = H5I_INVALID_HID; /* Return value */

    /* Initialize error reporting */
    if ((H5FDgds_err_stack_g = H5Ecreate_stack()) < 0)
        H5FD_GDS_GOTO_ERROR(H5E_VFL, H5E_CANTINIT, H5I_INVALID_HID, "can't create HDF5 error stack");
    if ((H5FDgds_err_class_g = H5Eregister_class(H5FD_GDS_ERR_CLS_NAME, H5FD_GDS_ERR_LIB_NAME, H5FD_GDS_ERR_VER)) < 0)
        H5FD_GDS_GOTO_ERROR(H5E_VFL, H5E_CANTINIT, H5I_INVALID_HID, "can't register error class with HDF5 error API");

    /* Check the use disabled file locks environment variable */
    lock_env_var = getenv("HDF5_USE_FILE_LOCKING");
    if (lock_env_var && !strcmp(lock_env_var, "BEST_EFFORT"))
        ignore_disabled_file_locks_s = TRUE; /* Override: Ignore disabled locks */
    else if (lock_env_var && (!strcmp(lock_env_var, "TRUE") || !strcmp(lock_env_var, "1")))
        ignore_disabled_file_locks_s = FALSE; /* Override: Don't ignore disabled locks */
    else
        ignore_disabled_file_locks_s = FAIL; /* Environment variable not set, or not set correctly */

    if (!cu_file_driver_opened) {
        status = cuFileDriverOpen();

        if (status.err == CU_FILE_SUCCESS) {
            cu_file_driver_opened = true;
        }
        else {
            H5FD_GDS_GOTO_ERROR(H5E_INTERNAL, H5E_SYSTEM, H5I_INVALID_HID, "unable to open cufile driver");
            /* TODO: get the error string once the cufile c api is ready */
            /*
             * fprintf(stderr, "cufile driver open error: %s\n",
             * cuFileGetErrorString(status));
             */
        }
    }

    if (H5I_VFL != H5Iget_type(H5FD_GDS_g))
        H5FD_GDS_g = H5FDregister(&H5FD_gds_g);

    /* Set return value */
    ret_value = H5FD_GDS_g;

done:
    H5FD_GDS_FUNC_LEAVE;
} /* end H5FD_gds_init() */

/*---------------------------------------------------------------------------
 * Function:  H5FD__gds_term
 *
 * Purpose:  Shut down the VFD
 *
 * Returns:     Non-negative on success or negative on failure
 *
 * Programmer:  John J Ravi
 *              Tuesday, 06 October 2020
 *
 *---------------------------------------------------------------------------
 */
static herr_t
H5FD__gds_term(void)
{
    herr_t ret_value = SUCCEED; /* Return value */

    if (cu_file_driver_opened) {
        /* CUfileError_t status; */

        /* FIXME: cuFileDriveClose is throwing errors with h5py and cupy */
        /*
         * status = cuFileDriverClose();
         * if (status.err == CU_FILE_SUCCESS) {
         *   cu_file_driver_opened = false;
         * }
         * else {
         *   H5FD_GDS_GOTO_ERROR(H5E_INTERNAL, H5E_SYSTEM, NULL, "unable to close cufile driver");
         *   // TODO: get the error string once the cufile c api is ready
         *   // fprintf(stderr, "cufile driver close failed: %s\n",
         *   //   cuFileGetErrorString(status));
         * }
         */
    }

    /* Unregister from HDF5 error API */
    if (H5FDgds_err_class_g >= 0) {
        if (H5Eunregister_class(H5FDgds_err_class_g) < 0)
            H5FD_GDS_GOTO_ERROR(H5E_VFL, H5E_CLOSEERROR, FAIL, "can't unregister error class from HDF5 error API");

        /* Print the current error stack before destroying it */
        PRINT_ERROR_STACK;

        /* Destroy the error stack */
        if (H5Eclose_stack(H5FDgds_err_stack_g) < 0) {
            H5FD_GDS_GOTO_ERROR(H5E_VFL, H5E_CLOSEERROR, FAIL, "can't close HDF5 error stack");
            PRINT_ERROR_STACK;
        } /* end if */

        H5FDgds_err_stack_g = H5I_INVALID_HID;
        H5FDgds_err_class_g = H5I_INVALID_HID;
    } /* end if */

    /* Reset VFL ID */
    H5FD_GDS_g = H5I_INVALID_HID;

done:
    H5FD_GDS_FUNC_LEAVE_API;
} /* end H5FD__gds_term() */

/*-------------------------------------------------------------------------
 * Function:  H5Pset_fapl_gds
 *
 * Purpose:  Modify the file access property list to use the H5FD_GDS
 *    driver defined in this source file.  There are no driver
 *    specific properties.
 *
 * Return:  Non-negative on success/Negative on failure
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5Pset_fapl_gds(hid_t fapl_id, size_t boundary, size_t block_size, size_t cbuf_size)
{
    H5FD_gds_fapl_t fa;
    herr_t          ret_value;

    if (H5I_GENPROP_LST != H5Iget_type(fapl_id) || TRUE != H5Pisa_class(fapl_id, H5P_FILE_ACCESS))
        H5FD_GDS_GOTO_ERROR(H5E_ARGS, H5E_BADTYPE, FAIL, "not a file access property list");

    if (H5FD__gds_populate_config(boundary, block_size, cbuf_size, &fa) < 0)
        H5FD_GDS_GOTO_ERROR(H5E_VFL, H5E_CANTSET, FAIL, "can't initialize driver configuration info");

    ret_value = H5Pset_driver(fapl_id, H5FD_GDS, &fa);

done:
    H5FD_GDS_FUNC_LEAVE_API;
}

/*-------------------------------------------------------------------------
 * Function:  H5Pget_fapl_gds
 *
 * Purpose:  Returns information about the gds file access property
 *    list though the function arguments.
 *
 * Return:  Success:  Non-negative
 *
 *    Failure:  Negative
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5Pget_fapl_gds(hid_t fapl_id, size_t *boundary /*out*/, size_t *block_size /*out*/,
                size_t *cbuf_size /*out*/)
{
    const H5FD_gds_fapl_t *fa;
    H5FD_gds_fapl_t        default_fa;
    herr_t                 ret_value = SUCCEED; /* Return value */

    if (H5I_GENPROP_LST != H5Iget_type(fapl_id) || TRUE != H5Pisa_class(fapl_id, H5P_FILE_ACCESS))
        H5FD_GDS_GOTO_ERROR(H5E_ARGS, H5E_BADTYPE, FAIL, "not a file access property list");
    if (H5FD_GDS != H5Pget_driver(fapl_id))
        H5FD_GDS_GOTO_ERROR(H5E_PLIST, H5E_BADVALUE, FAIL, "incorrect VFL driver");
    H5E_BEGIN_TRY
    {
        fa = H5Pget_driver_info(fapl_id);
    }
    H5E_END_TRY;
    if (!fa || (H5P_FILE_ACCESS_DEFAULT == fapl_id)) {
        if (H5FD__gds_populate_config(0, 0, 0, &default_fa) < 0)
            H5FD_GDS_GOTO_ERROR(H5E_VFL, H5E_CANTSET, FAIL, "can't initialize driver configuration info");

        fa = &default_fa;
    }

    if (boundary)
        *boundary = fa->mboundary;
    if (block_size)
        *block_size = fa->fbsize;
    if (cbuf_size)
        *cbuf_size = fa->cbsize;

done:
    H5FD_GDS_FUNC_LEAVE_API;
} /* end H5Pget_fapl_gds() */

/*-------------------------------------------------------------------------
 * Function:    H5FD__gds_populate_config
 *
 * Purpose:    Populates a H5FD_gds_fapl_t structure with the provided
 *             values, supplying defaults where values are not provided.
 *
 * Return:    Non-negative on success/Negative on failure
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5FD__gds_populate_config(size_t boundary, size_t block_size, size_t cbuf_size, H5FD_gds_fapl_t *fa_out)
{
    herr_t ret_value = SUCCEED;

    assert(fa_out);

    memset(fa_out, 0, sizeof(H5FD_gds_fapl_t));

    if (boundary != 0)
        fa_out->mboundary = boundary;
    else
        fa_out->mboundary = MBOUNDARY_DEF;

    if (block_size != 0)
        fa_out->fbsize = block_size;
    else
        fa_out->fbsize = FBSIZE_DEF;

    if (cbuf_size != 0)
        fa_out->cbsize = cbuf_size;
    else
        fa_out->cbsize = CBSIZE_DEF;

    /* Set the default to be true for data alignment */
    fa_out->must_align = TRUE;

    /* Copy buffer size must be a multiple of file block size */
    if (fa_out->cbsize % fa_out->fbsize != 0)
        H5FD_GDS_GOTO_ERROR(H5E_ARGS, H5E_BADVALUE, FAIL, "copy buffer size must be a multiple of block size");

done:
    H5FD_GDS_FUNC_LEAVE;
} /* end H5FD__gds_populate_config() */

/*-------------------------------------------------------------------------
 * Function:  H5FD__gds_fapl_get
 *
 * Purpose:  Returns a file access property list which indicates how the
 *    specified file is being accessed. The return list could be
 *    used to access another file the same way.
 *
 * Return:  Success:  Ptr to new file access property list with all
 *        members copied from the file struct.
 *
 *    Failure:  NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5FD__gds_fapl_get(H5FD_t *_file)
{
    H5FD_gds_t *file = (H5FD_gds_t *)_file;
    void *      ret_value; /* Return value */

    /* Set return value */
    ret_value = H5FD__gds_fapl_copy(&(file->fa));

    H5FD_GDS_FUNC_LEAVE_API;
} /* end H5FD__gds_fapl_get() */

/*-------------------------------------------------------------------------
 * Function:  H5FD__gds_fapl_copy
 *
 * Purpose:  Copies the gds-specific file access properties.
 *
 * Return:  Success:  Ptr to a new property list
 *
 *    Failure:  NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5FD__gds_fapl_copy(const void *_old_fa)
{
    const H5FD_gds_fapl_t *old_fa = (const H5FD_gds_fapl_t *)_old_fa;
    H5FD_gds_fapl_t *      new_fa = calloc(1, sizeof(H5FD_gds_fapl_t));
    void *                 ret_value = NULL;

    assert(new_fa);

    /* Copy the general information */
    memcpy(new_fa, old_fa, sizeof(H5FD_gds_fapl_t));

    ret_value = new_fa;

    H5FD_GDS_FUNC_LEAVE_API;
} /* end H5FD__gds_fapl_copy() */

/*-------------------------------------------------------------------------
 * Function:  H5FD__gds_open
 *
 * Purpose:  Create and/or opens a Unix file for direct I/O as an HDF5 file.
 *
 * Return:  Success:  A pointer to a new file data structure. The
 *        public fields will be initialized by the
 *        caller, which is always H5FD_open().
 *
 *    Failure:  NULL
 *
 * Programmer:  John J Ravi
 *              Tuesday, 06 October 2020
 *
 *-------------------------------------------------------------------------
 */
static H5FD_t *
H5FD__gds_open(const char *name, unsigned flags, hid_t fapl_id, haddr_t maxaddr)
{
    CUfileError_t status;
    CUfileDescr_t cf_descr;
    char *        num_io_threads_var;
    char *        io_block_size_var;

    int              o_flags;
    int              fd   = (-1);
    H5FD_gds_t *     file = NULL;
    H5FD_gds_fapl_t *fa;
    H5FD_gds_fapl_t  default_fa;
#ifdef H5_HAVE_WIN32_API
    HFILE                              filehandle;
    struct _BY_HANDLE_FILE_INFORMATION fileinfo;
#endif
    struct stat     sb;
    void *          buf1, *buf2;
    H5FD_t *        ret_value = NULL;

    H5FD_GDS_INIT;

    /* Sanity check on file offsets */
    assert(sizeof(off_t) >= sizeof(size_t));

    /* Check arguments */
    if (!name || !*name)
        H5FD_GDS_GOTO_ERROR(H5E_ARGS, H5E_BADVALUE, NULL, "invalid file name");
    if (0 == maxaddr || HADDR_UNDEF == maxaddr)
        H5FD_GDS_GOTO_ERROR(H5E_ARGS, H5E_BADRANGE, NULL, "bogus maxaddr");
    if (ADDR_OVERFLOW(maxaddr))
        H5FD_GDS_GOTO_ERROR(H5E_ARGS, H5E_OVERFLOW, NULL, "bogus maxaddr");

    /* Build the open flags */
    o_flags = (H5F_ACC_RDWR & flags) ? O_RDWR : O_RDONLY;
    if (H5F_ACC_TRUNC & flags)
        o_flags |= O_TRUNC;
    if (H5F_ACC_CREAT & flags)
        o_flags |= O_CREAT;
    if (H5F_ACC_EXCL & flags)
        o_flags |= O_EXCL;

    /* Flag for GPUDirect Storage I/O */
    o_flags |= O_DIRECT;

    /* Open the file */
    if ((fd = open(name, o_flags, H5FD_GDS_POSIX_CREATE_MODE_RW)) < 0)
        H5FD_GDS_SYS_GOTO_ERROR(H5E_FILE, H5E_CANTOPENFILE, NULL, "unable to open file");

    if (fstat(fd, &sb) < 0)
        H5FD_GDS_SYS_GOTO_ERROR(H5E_FILE, H5E_BADFILE, NULL, "unable to fstat file");

#ifdef ADVISE_OS_DISABLE_READ_CACHE
    if (posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM) != 0) {
        perror("posix_fadvise");
        exit(EXIT_FAILURE);
    }

    if (posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) != 0) {
        perror("posix_fadvise");
        exit(EXIT_FAILURE);
    }

    if (posix_fadvise(fd, 0, 0, POSIX_FADV_NOREUSE) != 0) {
        perror("posix_fadvise");
        exit(EXIT_FAILURE);
    }
#endif /* ADVISE_OS_DISABLE_READ_CACHE */

    /* Create the new file struct */
    if (NULL == (file = calloc(1, sizeof(H5FD_gds_t))))
        H5FD_GDS_GOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, NULL, "unable to allocate file struct");

    /* Get the driver specific information */
    H5E_BEGIN_TRY
    {
        fa = H5Pget_driver_info(fapl_id);
    }
    H5E_END_TRY;
    if (!fa || (H5P_FILE_ACCESS_DEFAULT == fapl_id)) {
        if (H5FD__gds_populate_config(0, 0, 0, &default_fa) < 0)
            H5FD_GDS_GOTO_ERROR(H5E_VFL, H5E_CANTSET, NULL, "can't initialize driver configuration info");
        fa = &default_fa;
    }

    memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fd;
    cf_descr.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    status             = cuFileHandleRegister(&file->cf_handle, &cf_descr);
    if (status.err != CU_FILE_SUCCESS) {
        H5FD_GDS_GOTO_ERROR(H5E_INTERNAL, H5E_SYSTEM, NULL, "unable to register file with cufile driver");
    }

    /* DEFAULT io worker params */
    /* TODO: error checking */
    file->num_io_threads = 1;
    file->io_block_size  = 8 * 1024 * 1024;
    num_io_threads_var   = getenv("H5_GDS_VFD_IO_THREADS");
    io_block_size_var    = getenv("H5_GDS_VFD_IO_BLOCK_SIZE");

    if (num_io_threads_var) {
        file->num_io_threads = atoi(num_io_threads_var);
    }

    if (io_block_size_var) {
        file->io_block_size = atoi(io_block_size_var);
    }

    /* TODO: error checking for num_io_threads */
    /* FIXME: move to set fapl */
    /*
     * H5Pget( fapl_id, "H5_GDS_VFD_IO_THREADS", &file->num_io_threads );
     * H5Pget( fapl_id, "H5_GDS_VFD_IO_BLOCK_SIZE", &file->io_block_size );
     */

    /* IOThreads */
    /* TODO: POSSIBLE MEMORY LEAK! figure out how to deal with the the double open H5Fint does */
    file->td      = (thread_data_t *)malloc((unsigned)file->num_io_threads * sizeof(thread_data_t));
    file->threads = (pthread_t *)malloc((unsigned)file->num_io_threads * sizeof(pthread_t));

    file->fd = fd;

    /* FIXME: Possible overflow! */
    file->eof = (haddr_t)sb.st_size;

    file->pos = HADDR_UNDEF;
    file->op  = OP_UNKNOWN;
#ifdef H5_HAVE_WIN32_API
    filehandle = _get_osfhandle(fd);
    (void)GetFileInformationByHandle((HANDLE)filehandle, &fileinfo);
    file->fileindexhi = fileinfo.nFileIndexHigh;
    file->fileindexlo = fileinfo.nFileIndexLow;
#else
    file->device = sb.st_dev;
    file->inode  = sb.st_ino;
#endif /*H5_HAVE_WIN32_API*/
    file->fa.mboundary = fa->mboundary;
    file->fa.fbsize    = fa->fbsize;
    file->fa.cbsize    = fa->cbsize;

    /* Check the file locking flags in the fapl */
    if (ignore_disabled_file_locks_s != FAIL)
        /* The environment variable was set, so use that preferentially */
        file->ignore_disabled_file_locks = ignore_disabled_file_locks_s;
    else {
        hbool_t unused;

        /* Use the value in the property list */
        if (H5Pget_file_locking(fapl_id, &unused, &file->ignore_disabled_file_locks) < 0)
            H5FD_GDS_GOTO_ERROR(H5E_VFL, H5E_CANTGET, NULL, "can't get ignore disabled file locks property");
    }

    /* Try to decide if data alignment is required.  The reason to check it here
     * is to handle correctly the case that the file is in a different file system
     * than the one where the program is running.
     */
    /* NOTE: Use malloc and free here to ensure compatibility with
     *       posix_memalign.
     */
    buf1 = malloc(sizeof(int));
    if (posix_memalign(&buf2, file->fa.mboundary, file->fa.fbsize) != 0)
        H5FD_GDS_GOTO_ERROR(H5E_RESOURCE, H5E_CANTALLOC, NULL, "posix_memalign failed");

    if (o_flags & O_CREAT) {
        if (write(file->fd, buf1, sizeof(int)) < 0) {
            if (write(file->fd, buf2, file->fa.fbsize) < 0)
                H5FD_GDS_GOTO_ERROR(H5E_FILE, H5E_WRITEERROR, NULL,
                            "file system may not support GPUDirect Storage I/O");
            else
                file->fa.must_align = TRUE;
        }
        else {
            file->fa.must_align = FALSE;
            if (-1 == ftruncate(file->fd, (off_t)0))
                H5FD_GDS_SYS_GOTO_ERROR(H5E_IO, H5E_SEEKERROR, NULL, "unable to extend file properly");
        }
    }
    else {
        if (read(file->fd, buf1, sizeof(int)) < 0) {
            if (read(file->fd, buf2, file->fa.fbsize) < 0)
                H5FD_GDS_GOTO_ERROR(H5E_FILE, H5E_READERROR, NULL,
                            "file system may not support GPUDirect Storage I/O");
            else
                file->fa.must_align = TRUE;
        }
        else {
            if (o_flags & O_RDWR) {
                if (lseek(file->fd, (off_t)0, SEEK_SET) < 0)
                    H5FD_GDS_SYS_GOTO_ERROR(H5E_IO, H5E_SEEKERROR, NULL, "unable to seek to proper position");
                if (write(file->fd, buf1, sizeof(int)) < 0)
                    file->fa.must_align = TRUE;
                else
                    file->fa.must_align = FALSE;
            }
            else
                file->fa.must_align = FALSE;
        }
    }

    if (buf1)
        free(buf1);
    if (buf2)
        free(buf2);

    /* Set return value */
    ret_value = (H5FD_t *)file;

done:
    if (ret_value == NULL) {
        if (fd >= 0)
            close(fd);
    } /* end if */

    H5FD_GDS_FUNC_LEAVE_API;
}

/*-------------------------------------------------------------------------
 * Function:  H5FD__gds_close
 *
 * Purpose:  Closes the file.
 *
 * Return:  Success:  0
 *
 *    Failure:  -1, file not closed.
 *
 * Programmer:  John J Ravi
 *              Tuesday, 06 October 2020
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5FD__gds_close(H5FD_t *_file)
{
    H5FD_gds_t *file      = (H5FD_gds_t *)_file;
    herr_t      ret_value = SUCCEED; /* Return value */

    /* close file handle */
    cuFileHandleDeregister(file->cf_handle);

    if (close(file->fd) < 0)
        H5FD_GDS_SYS_GOTO_ERROR(H5E_IO, H5E_CANTCLOSEFILE, FAIL, "unable to close file");

    if (file->td)
        free(file->td);

    if (file->threads)
        free(file->threads);

    free(file);

done:
    H5FD_GDS_FUNC_LEAVE_API;
}

/*-------------------------------------------------------------------------
 * Function:  H5FD__gds_cmp
 *
 * Purpose:  Compares two files belonging to this driver using an
 *    arbitrary (but consistent) ordering.
 *
 * Return:  Success:  A value like strcmp()
 *
 *    Failure:  never fails (arguments were checked by the
 *        caller).
 *
 *-------------------------------------------------------------------------
 */
static int
H5FD__gds_cmp(const H5FD_t *_f1, const H5FD_t *_f2)
{
    const H5FD_gds_t *f1        = (const H5FD_gds_t *)_f1;
    const H5FD_gds_t *f2        = (const H5FD_gds_t *)_f2;
    int               ret_value = 0;

#ifdef H5_HAVE_WIN32_API
    if (f1->fileindexhi < f2->fileindexhi)
        H5FD_GDS_GOTO_DONE(-1);
    if (f1->fileindexhi > f2->fileindexhi)
        H5FD_GDS_GOTO_DONE(1);

    if (f1->fileindexlo < f2->fileindexlo)
        H5FD_GDS_GOTO_DONE(-1);
    if (f1->fileindexlo > f2->fileindexlo)
        H5FD_GDS_GOTO_DONE(1);

#else
#ifdef H5_DEV_T_IS_SCALAR
    if (f1->device < f2->device)
        H5FD_GDS_GOTO_DONE(-1);
    if (f1->device > f2->device)
        H5FD_GDS_GOTO_DONE(1);
#else  /* H5_DEV_T_IS_SCALAR */
    /* If dev_t isn't a scalar value on this system, just use memcmp to
     * determine if the values are the same or not.  The actual return value
     * shouldn't really matter...
     */
    if (memcmp(&(f1->device), &(f2->device), sizeof(dev_t)) < 0)
        H5FD_GDS_GOTO_DONE(-1);
    if (memcmp(&(f1->device), &(f2->device), sizeof(dev_t)) > 0)
        H5FD_GDS_GOTO_DONE(1);
#endif /* H5_DEV_T_IS_SCALAR */

    if (f1->inode < f2->inode)
        H5FD_GDS_GOTO_DONE(-1);
    if (f1->inode > f2->inode)
        H5FD_GDS_GOTO_DONE(1);

#endif

done:
    H5FD_GDS_FUNC_LEAVE_API;
}

/*-------------------------------------------------------------------------
 * Function:  H5FD__gds_query
 *
 * Purpose:  Set the flags that this VFL driver is capable of supporting.
 *              (listed in H5FDpublic.h)
 *
 * Return:  Success:  non-negative
 *
 *    Failure:  negative
 *
 * Programmer:  John J Ravi
 *              Tuesday, 06 October 2020
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5FD__gds_query(const H5FD_t *_f, unsigned long *flags /* out */)
{
    herr_t ret_value = SUCCEED;

    /* Silence compiler */
    (void)_f;

    /* Set the VFL feature flags that this driver supports */
    if (flags) {
        *flags = 0;
        *flags |= H5FD_FEAT_AGGREGATE_METADATA;  /* OK to aggregate metadata allocations  */
        *flags |= H5FD_FEAT_ACCUMULATE_METADATA; /* OK to accumulate metadata for faster writes */
        *flags |= H5FD_FEAT_AGGREGATE_SMALLDATA; /* OK to aggregate "small" raw data allocations */
        *flags |=
            H5FD_FEAT_SUPPORTS_SWMR_IO; /* VFD supports the single-writer/multiple-readers (SWMR) pattern   */
        *flags |= H5FD_FEAT_DEFAULT_VFD_COMPATIBLE; /* VFD creates a file which can be opened with the default
                                                       VFD      */
        *flags |= H5FD_FEAT_MEMMANAGE; /* VFD uses CUDA memory management routines */
    }

    H5FD_GDS_FUNC_LEAVE_API;
}

/*-------------------------------------------------------------------------
 * Function:  H5FD__gds_get_eoa
 *
 * Purpose:  Gets the end-of-address marker for the file. The EOA marker
 *    is the first address past the last byte allocated in the
 *    format address space.
 *
 * Return:  Success:  The end-of-address marker.
 *
 *    Failure:  HADDR_UNDEF
 *
 *-------------------------------------------------------------------------
 */
static haddr_t
H5FD__gds_get_eoa(const H5FD_t *_file, H5FD_mem_t type)
{
    const H5FD_gds_t *file = (const H5FD_gds_t *)_file;
    haddr_t           ret_value = HADDR_UNDEF;

    assert(file);

    /* Silence compiler */
    (void)type;

    ret_value = file->eoa;

    H5FD_GDS_FUNC_LEAVE_API;
}

/*-------------------------------------------------------------------------
 * Function:  H5FD__gds_set_eoa
 *
 * Purpose:  Set the end-of-address marker for the file. This function is
 *    called shortly after an existing HDF5 file is opened in order
 *    to tell the driver where the end of the HDF5 data is located.
 *
 * Return:  Success:  0
 *
 *    Failure:  -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5FD__gds_set_eoa(H5FD_t *_file, H5FD_mem_t type, haddr_t addr)
{
    H5FD_gds_t *file = (H5FD_gds_t *)_file;
    herr_t      ret_value = SUCCEED;

    /* Silence compiler */
    (void)type;

    file->eoa = addr;

    H5FD_GDS_FUNC_LEAVE_API;
}

/*-------------------------------------------------------------------------
 * Function:  H5FD__gds_get_eof
 *
 * Purpose:  Returns the end-of-file marker, which is the greater of
 *    either the Unix end-of-file or the HDF5 end-of-address
 *    markers.
 *
 * Return:  Success:  End of file address, the first address past
 *        the end of the "file", either the Unix file
 *        or the HDF5 file.
 *
 *    Failure:  HADDR_UNDEF
 *
 *-------------------------------------------------------------------------
 */
static haddr_t
H5FD__gds_get_eof(const H5FD_t *_file, H5FD_mem_t type)
{
    const H5FD_gds_t *file = (const H5FD_gds_t *)_file;
    haddr_t           ret_value = HADDR_UNDEF;

    assert(file);

    /* Silence compiler */
    (void)type;

    ret_value = file->eof;

    H5FD_GDS_FUNC_LEAVE_API;
}

/*-------------------------------------------------------------------------
 * Function:       H5FD_gds_get_handle
 *
 * Purpose:        Returns the file handle of gds file driver.
 *
 * Returns:        Non-negative if succeed or negative if fails.
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5FD__gds_get_handle(H5FD_t *_file, hid_t fapl, void **file_handle)
{
    H5FD_gds_t *file      = (H5FD_gds_t *)_file;
    herr_t      ret_value = SUCCEED;

    /* Silence compiler */
    (void)fapl;

    if (!file_handle)
        H5FD_GDS_GOTO_ERROR(H5E_ARGS, H5E_BADVALUE, FAIL, "file handle not valid");
    *file_handle = &(file->fd);

done:
    H5FD_GDS_FUNC_LEAVE_API;
}

bool is_device_pointer(const void *ptr);
bool
is_device_pointer(const void *ptr)
{
    struct cudaPointerAttributes attributes;
    cudaPointerGetAttributes(&attributes, ptr);
    return (attributes.devicePointer != NULL);
}

/*-------------------------------------------------------------------------
 * Function:  H5FD__gds_read
 *
 * Purpose:
 *    GPU buf:
 *    interface with NVIDIA GPUDirect Storage
 *
 *    CPU buf:
 *    Reads SIZE bytes of data from FILE beginning at address ADDR
 *    into buffer BUF according to data transfer properties in
 *    DXPL_ID.
 *
 * Return:  Success:  Zero. Result is stored in caller-supplied
 *        buffer BUF.
 *
 *    Failure:  -1, Contents of buffer BUF are undefined.
 *
 * Programmer:  John J Ravi
 *              Tuesday, 06 October 2020
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5FD__gds_read(H5FD_t *_file, H5FD_mem_t type, hid_t dxpl_id, haddr_t addr,
               size_t size, void *buf /*out*/)
{
    H5FD_gds_t *file = (H5FD_gds_t *)_file;
    ssize_t     nbytes;
    hbool_t     _must_align = TRUE;
    herr_t      ret_value   = SUCCEED; /* Return value */
    size_t      alloc_size;
    void *      copy_buf = NULL, *p2;
    size_t      _boundary;
    size_t      _fbsize;
    size_t      _cbsize;
    haddr_t     read_size;        /* Size to read into copy buffer */
    size_t      copy_size = size; /* Size remaining to read when using copy buffer */
    size_t      copy_offset;      /* Offset into copy buffer of the requested data */

    ssize_t       ret        = -1;
    int           io_threads = file->num_io_threads;
    int           block_size = file->io_block_size;

    off_t offset = (off_t)addr;
    ssize_t io_chunk;
    ssize_t io_chunk_rem;

    assert(file && file->pub.cls);
    assert(buf);

    /* Silence compiler */
    (void)type;
    (void)dxpl_id;

    /* Check for overflow conditions */
    if (HADDR_UNDEF == addr)
        H5FD_GDS_GOTO_ERROR(H5E_ARGS, H5E_BADVALUE, FAIL, "addr undefined");
    if (REGION_OVERFLOW(addr, size))
        H5FD_GDS_GOTO_ERROR(H5E_ARGS, H5E_OVERFLOW, FAIL, "addr overflow");

    if (is_device_pointer(buf)) {
        /* CUfileError_t status; */

        /* TODO: register device memory only once */
        /*
         * if (!reg_once) {
         *   status = cuFileBufRegister(buf, size, 0);
         *   if (status.err != CU_FILE_SUCCESS) {
         *     H5FD_GDS_GOTO_ERROR(H5E_INTERNAL, H5E_SYSTEM, NULL, "cufile buffer register failed");
         *   }
         *   reg_once = true;
         * }
         */

        if (io_threads > 0) {
            assert(size != 0);

            /* make each thread access at least a 4K page */
            if ((1 + (size - 1) / 4096) < (unsigned)io_threads) {
                io_threads = (int)(1 + ((size - 1) / 4096));
            }

            /*
             * printf("\tH5Pset_gds_read using io_threads: %d\n", io_threads);
             * printf("\tH5Pset_gds_read using io_block_size: %d\n", block_size);
             */

            io_chunk     = (unsigned)size / (unsigned)io_threads;
            io_chunk_rem = (unsigned)size % (unsigned)io_threads;

            for (int ii = 0; ii < io_threads; ii++) {
                file->td[ii].rd_devPtr  = buf;
                file->td[ii].cfr_handle = file->cf_handle;

                file->td[ii].offset        = (off_t)(offset + ii * io_chunk);
                file->td[ii].devPtr_offset = (off_t)ii * io_chunk;
                file->td[ii].size          = (size_t)io_chunk;
                file->td[ii].block_size    = block_size;

                if (ii == io_threads - 1) {
                    file->td[ii].size = (size_t)(io_chunk + io_chunk_rem);
                }
            }

            for (int ii = 0; ii < io_threads; ii++) {
                pthread_create(&file->threads[ii], NULL, &read_thread_fn, &file->td[ii]);
            }

            for (int ii = 0; ii < io_threads; ii++) {
                pthread_join(file->threads[ii], NULL);
            }
        }
        else {
            ret = cuFileRead(file->cf_handle, buf, size, offset, 0);
            assert(ret > 0);
        }

        /* TODO: deregister device memory only once */
        /*
         * status = cuFileBufDeregister(buf);
         * if (status.err != CU_FILE_SUCCESS) {
         *   H5FD_GDS_GOTO_ERROR(H5E_INTERNAL, H5E_SYSTEM, NULL, "cufile buffer deregister failed");
         * }
         */
    }
    else {
        /* If the system doesn't require data to be aligned, read the data in
         * the same way as sec2 driver.
         */
        _must_align = file->fa.must_align;

        /* Get the memory boundary for alignment, file system block size, and maximal
         * copy buffer size.
         */
        _boundary = file->fa.mboundary;
        _fbsize   = file->fa.fbsize;
        _cbsize   = file->fa.cbsize;

        /* if the data is aligned or the system doesn't require data to be aligned,
         * read it directly from the file.  If not, read a bigger
         * and aligned data first, then copy the data into memory buffer.
         */
        if (!_must_align ||
            ((addr % _fbsize == 0) && (size % _fbsize == 0) && ((size_t)buf % _boundary == 0))) {
            /* Seek to the correct location */
            if ((addr != file->pos || OP_READ != file->op) && lseek(file->fd, (off_t)addr, SEEK_SET) < 0)
                H5FD_GDS_SYS_GOTO_ERROR(H5E_IO, H5E_SEEKERROR, FAIL, "unable to seek to proper position");
            /* Read the aligned data in file first, being careful of interrupted
             * system calls and partial results. */
            while (size > 0) {
                do {
                    nbytes = read(file->fd, buf, size);
                } while (-1 == nbytes && EINTR == errno);
                if (-1 == nbytes) /* error */
                    H5FD_GDS_SYS_GOTO_ERROR(H5E_IO, H5E_READERROR, FAIL, "file read failed");
                if (0 == nbytes) {
                    /* end of file but not end of format address space */
                    memset(buf, 0, size);
                    break;
                }
                assert(nbytes >= 0);
                assert((size_t)nbytes <= size);

                /* FIXME */
                /* H5_CHECK_OVERFLOW(nbytes, ssize_t, size_t); */

                size -= (size_t)nbytes;

                /* FIXME */
                /* H5_CHECK_OVERFLOW(nbytes, ssize_t, haddr_t); */

                addr += (haddr_t)nbytes;
                buf = (char *)buf + nbytes;
            }
        }
        else {
            /* Calculate where we will begin copying from the copy buffer */
            copy_offset = (size_t)(addr % _fbsize);

            /* allocate memory needed for the GPUDirect Storage IO option up to the maximal
             * copy buffer size. Make a bigger buffer for aligned I/O if size is
             * smaller than maximal copy buffer. */
            alloc_size = ((copy_offset + size - 1) / _fbsize + 1) * _fbsize;
            if (alloc_size > _cbsize)
                alloc_size = _cbsize;
            assert(!(alloc_size % _fbsize));
            if (posix_memalign(&copy_buf, _boundary, alloc_size) != 0)
                H5FD_GDS_GOTO_ERROR(H5E_RESOURCE, H5E_CANTALLOC, FAIL, "posix_memalign failed");

            /* look for the aligned position for reading the data */
            assert(!(((addr / _fbsize) * _fbsize) % _fbsize));
            if (lseek(file->fd, (off_t)((addr / _fbsize) * _fbsize), SEEK_SET) < 0)
                H5FD_GDS_SYS_GOTO_ERROR(H5E_IO, H5E_SEEKERROR, FAIL, "unable to seek to proper position");

            /*
             * Read the aligned data in file into aligned buffer first, then copy the data
             * into the final buffer.  If the data size is bigger than maximal copy buffer
             * size, do the reading by segment (the outer while loop).  If not, do one step
             * reading.
             */
            do {
                /* Read the aligned data in file first.  Not able to handle interrupted
                 * system calls and partial results like sec2 driver does because the
                 * data may no longer be aligned. It's especially true when the data in
                 * file is smaller than ALLOC_SIZE. */
                memset(copy_buf, 0, alloc_size);

                /* Calculate how much data we have to read in this iteration
                 * (including unused parts of blocks) */
                if ((copy_size + copy_offset) < alloc_size)
                    read_size = ((copy_size + copy_offset - 1) / _fbsize + 1) * _fbsize;
                else
                    read_size = alloc_size;

                assert(!(read_size % _fbsize));
                do {
                    nbytes = read(file->fd, copy_buf, read_size);
                } while (-1 == nbytes && EINTR == errno);

                if (-1 == nbytes) /* error */
                    H5FD_GDS_SYS_GOTO_ERROR(H5E_IO, H5E_READERROR, FAIL, "file read failed");

                /* Copy the needed data from the copy buffer to the output
                 * buffer, and update copy_size.  If the copy buffer does not
                 * contain the rest of the data, just copy what's in the copy
                 * buffer and also update read_addr and copy_offset to read the
                 * next section of data. */
                p2 = (unsigned char *)copy_buf + copy_offset;
                if ((copy_size + copy_offset) <= alloc_size) {
                    memcpy(buf, p2, copy_size);
                    buf       = (unsigned char *)buf + copy_size;
                    copy_size = 0;
                } /* end if */
                else {
                    memcpy(buf, p2, alloc_size - copy_offset);
                    buf = (unsigned char *)buf + alloc_size - copy_offset;
                    copy_size -= alloc_size - copy_offset;
                    copy_offset = 0;
                } /* end else */
            } while (copy_size > 0);

            /*Final step: update address*/
            addr = (haddr_t)(((addr + size - 1) / _fbsize + 1) * _fbsize);

            if (copy_buf) {
                /* Free with free since it came from posix_memalign */
                free(copy_buf);
                copy_buf = NULL;
            } /* end if */
        }

        /* Update current position */
        file->pos = addr;
        file->op  = OP_READ;
    }

done:
    if (ret_value < 0) {
        /* Free with free since it came from posix_memalign */
        if (copy_buf)
            free(copy_buf);

        /* Reset last file I/O information */
        file->pos = HADDR_UNDEF;
        file->op  = OP_UNKNOWN;
    } /* end if */

    H5FD_GDS_FUNC_LEAVE_API;
}

/*-------------------------------------------------------------------------
 * Function:  H5FD__gds_write
 *
 * Purpose:
 *    GPU buf:
 *    interface with NVIDIA GPUDirect Storage
 *
 *    CPU buf:
 *    Writes SIZE bytes of data to FILE beginning at address ADDR
 *    from buffer BUF according to data transfer properties in
 *    DXPL_ID.
 *
 * Return:  Success:  Zero
 *
 *    Failure:  -1
 *
 * Programmer:  John J Ravi
 *              Tuesday, 06 October 2020
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5FD__gds_write(H5FD_t *_file, H5FD_mem_t type, hid_t dxpl_id, haddr_t addr,
                size_t size, const void *buf)
{
    H5FD_gds_t *file = (H5FD_gds_t *)_file;
    ssize_t     nbytes;
    hbool_t     _must_align = TRUE;
    herr_t      ret_value   = SUCCEED; /* Return value */
    size_t      alloc_size;
    void *      copy_buf = NULL, *p1;
    const void *p3;
    size_t      _boundary;
    size_t      _fbsize;
    size_t      _cbsize;
    haddr_t     write_addr;       /* Address to write copy buffer */
    haddr_t     write_size;       /* Size to write from copy buffer */
    haddr_t     read_size;        /* Size to read into copy buffer */
    size_t      copy_size = size; /* Size remaining to write when using copy buffer */
    size_t      copy_offset;      /* Offset into copy buffer of the data to write */

    ssize_t       ret        = -1;
    int           io_threads = file->num_io_threads;
    int           block_size = file->io_block_size;

    ssize_t io_chunk;
    ssize_t io_chunk_rem;

    off_t offset = (off_t)addr;

    assert(file && file->pub.cls);
    assert(buf);

    /* Silence compiler */
    (void)type;
    (void)dxpl_id;

    /* Check for overflow conditions */
    if (HADDR_UNDEF == addr)
        H5FD_GDS_GOTO_ERROR(H5E_ARGS, H5E_BADVALUE, FAIL, "addr undefined");
    if (REGION_OVERFLOW(addr, size))
        H5FD_GDS_GOTO_ERROR(H5E_ARGS, H5E_OVERFLOW, FAIL, "addr overflow");

    if (is_device_pointer(buf)) {
        /* CUfileError_t status; */

        /* TODO: register device memory only once */
        /*
         * if (!reg_once) {
         *   status = cuFileBufRegister(buf, size, 0);
         *   if (status.err != CU_FILE_SUCCESS) {
         *     H5FD_GDS_GOTO_ERROR(H5E_INTERNAL, H5E_SYSTEM, NULL, "cufile buffer register failed");
         *   }
         *   reg_once = true;
         * }
         */

        if (io_threads > 0) {
            assert(size != 0);

            /* make each thread access at least a 4K page */
            if ((1 + (size - 1) / 4096) < (unsigned)io_threads) {
                io_threads = (int)(1 + ((size - 1) / 4096));
            }

            /*
             * printf("\tH5Pset_gds_write using io_threads: %d\n", io_threads);
             * printf("\tH5Pset_gds_write using io_block_size: %d\n", block_size);
             */

            io_chunk     = (unsigned)size / (unsigned)io_threads;
            io_chunk_rem = (unsigned)size % (unsigned)io_threads;

            for (int ii = 0; ii < io_threads; ii++) {
                file->td[ii].wr_devPtr  = buf;
                file->td[ii].cfr_handle = file->cf_handle;

                file->td[ii].offset        = (off_t)(offset + ii * io_chunk);
                file->td[ii].devPtr_offset = (off_t)ii * io_chunk;
                file->td[ii].size          = (size_t)io_chunk;
                file->td[ii].block_size    = block_size;

                if (ii == io_threads - 1) {
                    file->td[ii].size = (size_t)(io_chunk + io_chunk_rem);
                }
            }

            for (int ii = 0; ii < io_threads; ii++) {
                pthread_create(&file->threads[ii], NULL, &write_thread_fn, &file->td[ii]);
            }

            for (int ii = 0; ii < io_threads; ii++) {
                pthread_join(file->threads[ii], NULL);
            }
        }
        else {
            /* FIXME: max xfer size, need to batch transfers */
            ret = cuFileWrite(file->cf_handle, buf, size, offset, 0);
            assert(ret > 0);
        }

        /* TODO: deregister device memory only once */
        /*
         * status = cuFileBufDeregister(buf);
         * if (status.err != CU_FILE_SUCCESS) {
         *   H5FD_GDS_GOTO_ERROR(H5E_INTERNAL, H5E_SYSTEM, NULL, "cufile buffer deregister failed");
         * }
         */
    }
    else {
        /* If the system doesn't require data to be aligned, read the data in
         * the same way as sec2 driver.
         */
        _must_align = file->fa.must_align;

        /* Get the memory boundary for alignment, file system block size, and maximal
         * copy buffer size.
         */
        _boundary = file->fa.mboundary;
        _fbsize   = file->fa.fbsize;
        _cbsize   = file->fa.cbsize;

        /* if the data is aligned or the system doesn't require data to be aligned,
         * write it directly to the file.  If not, read a bigger and aligned data
         * first, update buffer with user data, then write the data out.
         */
        if (!_must_align ||
            ((addr % _fbsize == 0) && (size % _fbsize == 0) && ((size_t)buf % _boundary == 0))) {
            /* Seek to the correct location */
            if ((addr != file->pos || OP_WRITE != file->op) && lseek(file->fd, (off_t)addr, SEEK_SET) < 0)
                H5FD_GDS_SYS_GOTO_ERROR(H5E_IO, H5E_SEEKERROR, FAIL, "unable to seek to proper position");

            while (size > 0) {
                do {
                    nbytes = write(file->fd, buf, size);
                } while (-1 == nbytes && EINTR == errno);
                if (-1 == nbytes) /* error */
                    H5FD_GDS_SYS_GOTO_ERROR(H5E_IO, H5E_WRITEERROR, FAIL, "file write failed");
                assert(nbytes > 0);
                assert((size_t)nbytes <= size);

                /* FIXME */
                /* H5_CHECK_OVERFLOW(nbytes, ssize_t, size_t); */

                size -= (size_t)nbytes;

                /* FIXME */
                /* H5_CHECK_OVERFLOW(nbytes, ssize_t, haddr_t); */

                addr += (haddr_t)nbytes;
                buf = (const char *)buf + nbytes;
            }
        }
        else {
            /* Calculate where we will begin reading from (on disk) and where we
             * will begin copying from the copy buffer */
            write_addr  = (addr / _fbsize) * _fbsize;
            copy_offset = (size_t)(addr % _fbsize);

            /* allocate memory needed for the GPUDirect Storage IO option up to the maximal
             * copy buffer size. Make a bigger buffer for aligned I/O if size is
             * smaller than maximal copy buffer.
             */
            alloc_size = ((copy_offset + size - 1) / _fbsize + 1) * _fbsize;
            if (alloc_size > _cbsize)
                alloc_size = _cbsize;
            assert(!(alloc_size % _fbsize));

            if (posix_memalign(&copy_buf, _boundary, alloc_size) != 0)
                H5FD_GDS_GOTO_ERROR(H5E_RESOURCE, H5E_CANTALLOC, FAIL, "posix_memalign failed");

            /* look for the right position for reading or writing the data */
            if (lseek(file->fd, (off_t)write_addr, SEEK_SET) < 0)
                H5FD_GDS_SYS_GOTO_ERROR(H5E_IO, H5E_SEEKERROR, FAIL, "unable to seek to proper position");

            p3 = buf;
            do {
                /* Calculate how much data we have to write in this iteration
                 * (including unused parts of blocks) */
                if ((copy_size + copy_offset) < alloc_size)
                    write_size = ((copy_size + copy_offset - 1) / _fbsize + 1) * _fbsize;
                else
                    write_size = alloc_size;

                /*
                 * Read the aligned data first if the aligned region doesn't fall
                 * entirely in the range to be written.  Not able to handle interrupted
                 * system calls and partial results like sec2 driver does because the
                 * data may no longer be aligned. It's especially true when the data in
                 * file is smaller than ALLOC_SIZE.  Only read the entire section if
                 * both ends are misaligned, otherwise only read the block on the
                 * misaligned end.
                 */
                memset(copy_buf, 0, _fbsize);

                if (copy_offset > 0) {
                    if ((write_addr + write_size) > (addr + size)) {
                        assert((write_addr + write_size) - (addr + size) < _fbsize);
                        read_size = write_size;
                        p1        = copy_buf;
                    } /* end if */
                    else {
                        read_size = _fbsize;
                        p1        = copy_buf;
                    } /* end else */
                }     /* end if */
                else if ((write_addr + write_size) > (addr + size)) {
                    assert((write_addr + write_size) - (addr + size) < _fbsize);
                    read_size = _fbsize;
                    p1        = (unsigned char *)copy_buf + write_size - _fbsize;

                    /* Seek to the last block, for reading */
                    assert(!((write_addr + write_size - _fbsize) % _fbsize));
                    if (lseek(file->fd, (off_t)(write_addr + write_size - _fbsize), SEEK_SET) < 0)
                        H5FD_GDS_SYS_GOTO_ERROR(H5E_IO, H5E_SEEKERROR, FAIL, "unable to seek to proper position");
                } /* end if */
                else
                    p1 = NULL;

                if (p1) {
                    assert(!(read_size % _fbsize));
                    do {
                        nbytes = read(file->fd, p1, read_size);
                    } while (-1 == nbytes && EINTR == errno);

                    if (-1 == nbytes) /* error */
                        H5FD_GDS_SYS_GOTO_ERROR(H5E_IO, H5E_READERROR, FAIL, "file read failed");
                } /* end if */

                /* look for the right position and append or copy the data to be written to
                 * the aligned buffer.
                 * Consider all possible situations here: file address is not aligned on
                 * file block size; the end of data address is not aligned; the end of data
                 * address is aligned; data size is smaller or bigger than maximal copy size.
                 */
                p1 = (unsigned char *)copy_buf + copy_offset;
                if ((copy_size + copy_offset) <= alloc_size) {
                    memcpy(p1, p3, copy_size);
                    copy_size = 0;
                } /* end if */
                else {
                    memcpy(p1, p3, alloc_size - copy_offset);
                    p3 = (const unsigned char *)p3 + (alloc_size - copy_offset);
                    copy_size -= alloc_size - copy_offset;
                    copy_offset = 0;
                } /* end else */

                /*look for the aligned position for writing the data*/
                assert(!(write_addr % _fbsize));
                if (lseek(file->fd, (off_t)write_addr, SEEK_SET) < 0)
                    H5FD_GDS_SYS_GOTO_ERROR(H5E_IO, H5E_SEEKERROR, FAIL, "unable to seek to proper position");

                /*
                 * Write the data. It doesn't truncate the extra data introduced by
                 * alignment because that step is done in H5FD_gds_flush.
                 */
                assert(!(write_size % _fbsize));
                do {
                    nbytes = write(file->fd, copy_buf, write_size);
                } while (-1 == nbytes && EINTR == errno);

                if (-1 == nbytes) /* error */
                    H5FD_GDS_SYS_GOTO_ERROR(H5E_IO, H5E_WRITEERROR, FAIL, "file write failed");

                /* update the write address */
                write_addr += write_size;
            } while (copy_size > 0);

            /*Update the address and size*/
            addr = write_addr;
            buf  = (const char *)buf + size;

            if (copy_buf) {
                /* Free with free since it came from posix_memalign */
                free(copy_buf);
                copy_buf = NULL;
            } /* end if */
        }

        /* Update current position and eof */
        file->pos = addr;
        file->op  = OP_WRITE;
        if (file->pos > file->eof)
            file->eof = file->pos;
    }

done:
    if (ret_value < 0) {
        /* Free with free since it came from posix_memalign */
        if (copy_buf)
            free(copy_buf);

        /* Reset last file I/O information */
        file->pos = HADDR_UNDEF;
        file->op  = OP_UNKNOWN;
    } /* end if */

    H5FD_GDS_FUNC_LEAVE_API;
}

/*-------------------------------------------------------------------------
 * Function:  H5FD__gds_flush
 *
 * Purpose:  Flush makes use of fsync to flush data to persistent storage.
 *    O_DIRECT will disable the OS cache, but fsync maybe necessary on
 *    certain file system to get data to persistant storage.
 *
 * Return:  Success:  Zero
 *
 *    Failure:  -1
 *
 * Programmer:  John J Ravi
 *              Saturday, 3 October 2020
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5FD__gds_flush(H5FD_t *_file, hid_t dxpl_id, hbool_t closing)
{
    H5FD_gds_t *file      = (H5FD_gds_t *)_file; /* VFD file struct */
    herr_t      ret_value = SUCCEED;             /* Return value */

    assert(file);

    /* Silence compiler */
    (void)dxpl_id;
    (void)closing;

    if (fsync(file->fd) < 0) {
        H5FD_GDS_SYS_GOTO_ERROR(H5E_VFL, H5E_CANTFLUSH, FAIL, "unable perform fsync on file descriptor");
    }

done:
    H5FD_GDS_FUNC_LEAVE_API;
} /* end H5FD__gds_flush() */

/*-------------------------------------------------------------------------
 * Function:  H5FD__gds_truncate
 *
 * Purpose:  Makes sure that the true file size is the same (or larger)
 *    than the end-of-address.
 *
 * Return:  Success:  Non-negative
 *
 *    Failure:  Negative
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5FD__gds_truncate(H5FD_t *_file, hid_t dxpl_id, hbool_t closing)
{
    H5FD_gds_t *file      = (H5FD_gds_t *)_file;
    herr_t      ret_value = SUCCEED; /* Return value */

    assert(file);

    /* Silence compiler */
    (void)dxpl_id;
    (void)closing;

    /* Extend the file to make sure it's large enough */
    if (file->eoa != file->eof) {
#ifdef H5_HAVE_WIN32_API
        HFILE         filehandle; /* Windows file handle */
        LARGE_INTEGER li;         /* 64-bit integer for SetFilePointer() call */

        /* Map the posix file handle to a Windows file handle */
        filehandle = _get_osfhandle(file->fd);

        /* Translate 64-bit integers into form Windows wants */
        /* [This algorithm is from the Windows documentation for SetFilePointer()] */
        li.QuadPart = (LONGLONG)file->eoa;
        (void)SetFilePointer((HANDLE)filehandle, li.LowPart, &li.HighPart, FILE_BEGIN);
        if (SetEndOfFile((HANDLE)filehandle) == 0)
            H5FD_GDS_GOTO_ERROR(H5E_IO, H5E_SEEKERROR, FAIL, "unable to extend file properly");
#else  /* H5_HAVE_WIN32_API */
        if (-1 == ftruncate(file->fd, (off_t)file->eoa))
            H5FD_GDS_SYS_GOTO_ERROR(H5E_IO, H5E_SEEKERROR, FAIL, "unable to extend file properly");
#endif /* H5_HAVE_WIN32_API */

        /* Update the eof value */
        file->eof = file->eoa;

        /* Reset last file I/O information */
        file->pos = HADDR_UNDEF;
        file->op  = OP_UNKNOWN;
    }
    else if (file->fa.must_align) {
        /*Even though eof is equal to eoa, file is still truncated because GPUDirect Storage I/O
         *write introduces some extra data for alignment.
         */
        if (-1 == ftruncate(file->fd, (off_t)file->eof))
            H5FD_GDS_SYS_GOTO_ERROR(H5E_IO, H5E_SEEKERROR, FAIL, "unable to extend file properly");
    }

done:
    H5FD_GDS_FUNC_LEAVE_API;
} /* end H5FD__gds_truncate() */

/*-------------------------------------------------------------------------
 * Function:    H5FD__gds_lock
 *
 * Purpose:     To place an advisory lock on a file.
 *		The lock type to apply depends on the parameter "rw":
 *			TRUE--opens for write: an exclusive lock
 *			FALSE--opens for read: a shared lock
 *
 * Return:      SUCCEED/FAIL
 *
 * Programmer:  Vailin Choi; May 2013
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5FD__gds_lock(H5FD_t *_file, hbool_t rw)
{
    H5FD_gds_t *file = (H5FD_gds_t *)_file; /* VFD file struct      */
    int         lock_flags;                 /* file locking flags   */
    herr_t      ret_value = SUCCEED;        /* Return value         */

    assert(file);

    /* Set exclusive or shared lock based on rw status */
    lock_flags = rw ? LOCK_EX : LOCK_SH;

    /* Place a non-blocking lock on the file */
    if (flock(file->fd, lock_flags | LOCK_NB) < 0) {
        if (file->ignore_disabled_file_locks && ENOSYS == errno) {
            /* When errno is set to ENOSYS, the file system does not support
             * locking, so ignore it.
             */
            errno = 0;
        }
        else
            H5FD_GDS_SYS_GOTO_ERROR(H5E_VFL, H5E_CANTLOCKFILE, FAIL, "unable to lock file");
    }

done:
    H5FD_GDS_FUNC_LEAVE_API;
} /* end H5FD__gds_lock() */

/*-------------------------------------------------------------------------
 * Function:    H5FD__gds_unlock
 *
 * Purpose:     To remove the existing lock on the file
 *
 * Return:      SUCCEED/FAIL
 *
 * Programmer:  Vailin Choi; May 2013
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5FD__gds_unlock(H5FD_t *_file)
{
    H5FD_gds_t *file      = (H5FD_gds_t *)_file; /* VFD file struct */
    herr_t      ret_value = SUCCEED;             /* Return value */

    assert(file);

    if (flock(file->fd, LOCK_UN) < 0) {
        if (file->ignore_disabled_file_locks && ENOSYS == errno) {
            /* When errno is set to ENOSYS, the file system does not support
             * locking, so ignore it.
             */
            errno = 0;
        }
        else
            H5FD_GDS_SYS_GOTO_ERROR(H5E_VFL, H5E_CANTUNLOCKFILE, FAIL, "unable to unlock file");
    }

done:
    H5FD_GDS_FUNC_LEAVE_API;
} /* end H5FD__gds_unlock() */

/*-------------------------------------------------------------------------
 * Function:    H5FD__gds_delete
 *
 * Purpose:     Delete a file
 *
 * Return:      SUCCEED/FAIL
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5FD__gds_delete(const char *filename, hid_t fapl_id)
{
    herr_t ret_value = SUCCEED; /* Return value */

    H5FD_GDS_INIT;

    assert(filename);

    /* Silence compiler */
    (void)fapl_id;

    if (remove(filename) < 0)
        H5FD_GDS_SYS_GOTO_ERROR(H5E_VFL, H5E_CANTDELETEFILE, FAIL, "unable to delete file");

done:
    H5FD_GDS_FUNC_LEAVE_API;
} /* end H5FD__gds_delete() */

/*-------------------------------------------------------------------------
 * Function:    H5FD__gds_ctl
 *
 * Purpose:     Perform an optional "ctl" operation
 *
 * Return:      SUCCEED/FAIL
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5FD__gds_ctl(H5FD_t *_file, uint64_t op_code, uint64_t flags, const void *input,
              void **output)
{
    H5FD_gds_t *file      = (H5FD_gds_t *)_file; /* VFD file struct */
    herr_t      ret_value = SUCCEED;             /* Return value */

    assert(file);

    /* Silence compiler */
    (void)file;
    (void)output;

    switch (op_code) {
        /* Driver-level memory copy */
        case H5FD_CTL__MEM_COPY:
        {
            const H5FD_ctl_memcpy_args_t *copy_args = (const H5FD_ctl_memcpy_args_t *)input;
            enum cudaMemcpyKind cpyKind;
            hbool_t src_on_device = FALSE;
            hbool_t dst_on_device = FALSE;
            const void *src;
            void *dst;

            if (!copy_args)
                H5FD_GDS_GOTO_ERROR(H5E_ARGS, H5E_BADVALUE, FAIL, "invalid arguments to ctl operation");

            /* Add offsets to source and destination buffers */
            src = ((const unsigned char *)copy_args->srcbuf) + copy_args->src_off;
            dst = ((unsigned char *)copy_args->dstbuf) + copy_args->dst_off;

            /* Determine type of memory copy to perform */
            src_on_device = is_device_pointer(copy_args->srcbuf);
            dst_on_device = is_device_pointer(copy_args->dstbuf);

            if (src_on_device && dst_on_device)
                cpyKind = cudaMemcpyDeviceToDevice;
            else if (src_on_device && !dst_on_device)
                cpyKind = cudaMemcpyDeviceToHost;
            else if (!src_on_device && dst_on_device)
                cpyKind = cudaMemcpyHostToDevice;
            else
                cpyKind = cudaMemcpyHostToHost;

            check_cudaruntimecall(cudaMemcpy(dst, src, copy_args->len, cpyKind))

            break;
        }

        /* Unknown op code */
        default:
            if (flags & H5FD_CTL__FAIL_IF_UNKNOWN_FLAG)
                H5FD_GDS_GOTO_ERROR(H5E_VFL, H5E_FCNTL, FAIL, "unknown op_code and fail if unknown flag is set");
            break;
    }

done:
    H5FD_GDS_FUNC_LEAVE_API;
} /* end H5FD__gds_ctl() */

/*
 * Stub routines for dynamic plugin loading
 */

H5PL_type_t
H5PLget_plugin_type(void) {
    return H5PL_TYPE_VFD;
}

const void*
H5PLget_plugin_info(void) {
    return &H5FD_gds_g;
}
