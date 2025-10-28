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

#ifndef _GNU_SOURCE
#define _GNU_SOURCE /* For O_DIRECT flag */
#endif

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
    hbool_t         ignore_disabled_file_locks;

    CUfileHandle_t cf_handle;      /* cufile handle */

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
    H5FD_CLASS_VERSION,      /* struct version       */
    H5FD_GDS_VALUE,          /* value                */
    H5FD_GDS_NAME,           /* name                 */
    MAXADDR,                 /* maxaddr              */
    H5F_CLOSE_WEAK,          /* fc_degree            */
    H5FD__gds_term,          /* terminate            */
    NULL,                    /* sb_size              */
    NULL,                    /* sb_encode            */
    NULL,                    /* sb_decode            */
    0,                       /* fapl_size            */
    NULL,                    /* fapl_get             */
    NULL,                    /* fapl_copy            */
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
    NULL,                    /* read_vector          */
    NULL,                    /* write_vector         */
    NULL,                    /* read_selection       */
    NULL,                    /* write_selection      */
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
    herr_t          ret_value;

    /* Silence compiler */
    (void)boundary;
    (void)block_size;
    (void)cbuf_size;

    if (H5I_GENPROP_LST != H5Iget_type(fapl_id) || TRUE != H5Pisa_class(fapl_id, H5P_FILE_ACCESS))
        H5FD_GDS_GOTO_ERROR(H5E_ARGS, H5E_BADTYPE, FAIL, "not a file access property list");

    ret_value = H5Pset_driver(fapl_id, H5FD_GDS, NULL);

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
    herr_t                 ret_value = SUCCEED; /* Return value */

    if (H5I_GENPROP_LST != H5Iget_type(fapl_id) || TRUE != H5Pisa_class(fapl_id, H5P_FILE_ACCESS))
        H5FD_GDS_GOTO_ERROR(H5E_ARGS, H5E_BADTYPE, FAIL, "not a file access property list");
    if (H5FD_GDS != H5Pget_driver(fapl_id))
        H5FD_GDS_GOTO_ERROR(H5E_PLIST, H5E_BADVALUE, FAIL, "incorrect VFL driver");

    if (boundary)
        *boundary = 0;
    if (block_size)
        *block_size = 0;
    if (cbuf_size)
        *cbuf_size = 0;

done:
    H5FD_GDS_FUNC_LEAVE_API;
} /* end H5Pget_fapl_gds() */

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

    int              o_flags;
    int              fd   = (-1);
    H5FD_gds_t *     file = NULL;
#ifdef H5_HAVE_WIN32_API
    HFILE                              filehandle;
    struct _BY_HANDLE_FILE_INFORMATION fileinfo;
#endif
    struct stat     sb;
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

    memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fd;
    cf_descr.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    status             = cuFileHandleRegister(&file->cf_handle, &cf_descr);
    if (status.err != CU_FILE_SUCCESS) {
        H5FD_GDS_GOTO_ERROR(H5E_INTERNAL, H5E_SYSTEM, NULL, "unable to register file with cufile driver");
    }

    file->fd = fd;
    /* FIXME: Possible overflow! */
    file->eof = (haddr_t)sb.st_size;

#ifdef H5_HAVE_WIN32_API
    filehandle = _get_osfhandle(fd);
    (void)GetFileInformationByHandle((HANDLE)filehandle, &fileinfo);
    file->fileindexhi = fileinfo.nFileIndexHigh;
    file->fileindexlo = fileinfo.nFileIndexLow;
#else
    file->device = sb.st_dev;
    file->inode  = sb.st_ino;
#endif /*H5_HAVE_WIN32_API*/

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

    /* Set return value */
    ret_value = (H5FD_t *)file;
fprintf(stderr, "%s:%u - Successfully opened file w/GDS VFD\n", __func__, __LINE__);

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
    off_t offset = (off_t)addr;
    herr_t      ret_value   = SUCCEED; /* Return value */

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

    /* Pass to cuFile */
    if (cuFileRead(file->cf_handle, buf, size, offset, 0) < 0)
	H5FD_GDS_GOTO_ERROR(H5E_ARGS, H5E_OVERFLOW, FAIL,
		"file read failed: file descriptor = %d, "
		"buf = %p, total read size = %zu, offset = %llu",
		file->fd, buf, size, (unsigned long long)offset);

done:
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
    off_t offset = (off_t)addr;
    herr_t      ret_value   = SUCCEED; /* Return value */

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

    /* Pass to cuFile */
    if (cuFileWrite(file->cf_handle, buf, size, offset, 0) < 0)
	H5FD_GDS_GOTO_ERROR(H5E_ARGS, H5E_OVERFLOW, FAIL,
		"file write failed: file descriptor = %d, "
		"buf = %p, total write size = %zu, offset = %llu",
		file->fd, buf, size, (unsigned long long)offset);

done:
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
        case H5FD_CTL_MEM_COPY:
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
            if (flags & H5FD_CTL_FAIL_IF_UNKNOWN_FLAG)
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
