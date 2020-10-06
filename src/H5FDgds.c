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
 * Purpose: Interfaces with the CUDA GPUDirect Storage API
 *          Based on the Direct I/O file driver which forces the data to be written to
 *          the file directly without being copied into system kernel
 *          buffer.  The main system support this feature is Linux.
 */

#include "H5FDdrvr_module.h" /* This source code file is part of the H5FD driver module */


#include "H5private.h"      /* Generic Functions        */
#include "H5Eprivate.h"     /* Error handling           */
#include "H5Fprivate.h"     /* File access              */
#include "H5FDprivate.h"    /* File drivers             */
#include "H5FDgds.h"        /* cuda gds file driver     */
#include "H5FLprivate.h"    /* Free Lists               */
#include "H5Iprivate.h"     /* IDs                      */
#include "H5MMprivate.h"    /* Memory management        */
#include "H5Pprivate.h"     /* Property lists           */

#ifdef H5_GDS_SUPPORT

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufile.h>

#include <pthread.h>

#include <time.h>
#endif

#ifdef H5_GDS_SUPPORT
typedef struct thread_data_t {
  union {
    void *rd_devPtr;            /* read device address */
    const void *wr_devPtr;      /* write device address */
  };
  int fd;
  CUfileHandle_t cfr_handle; /* cuFile Handle */
  off_t offset;              /* File offset */
  off_t devPtr_offset;       /* device address offset */
  size_t block_size;         /* I/O chunk size */
  size_t size;               /* Read/Write size */
} thread_data_t;

static bool cu_file_driver_opened = false;
#endif

/* The driver identification number, initialized at runtime */
static hid_t H5FD_GDS_g = 0;

/* Whether to ignore file locks when disabled (env var value) */
static htri_t ignore_disabled_file_locks_s = FAIL;

/* File operations */
#define OP_UNKNOWN  0
#define OP_READ    1
#define OP_WRITE  2

/* Driver-specific file access properties */
typedef struct H5FD_gds_fapl_t {
    size_t  mboundary;  /* Memory boundary for alignment    */
    size_t  fbsize;    /* File system block size      */
    size_t  cbsize;    /* Maximal buffer size for copying user data  */
    hbool_t     must_align;     /* Decides if data alignment is required        */
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
    H5FD_t  pub;      /*public stuff, must be first  */
    int    fd;      /*the unix file      */
    haddr_t  eoa;      /*end of allocated region  */
    haddr_t  eof;      /*end of file; current file size*/
    haddr_t  pos;      /*current file I/O position  */
    int    op;      /*last operation    */
    H5FD_gds_fapl_t  fa;    /*file access properties  */
    hbool_t         ignore_disabled_file_locks;

#ifdef H5_GDS_SUPPORT
    CUfileHandle_t  cf_handle; /* cufile handle */
    int             num_io_threads; /* number of io threads for cufile */
    size_t          io_block_size; /* io block size or cufile */
    pthread_t       *threads;
    thread_data_t   *td;
#endif

#ifndef H5_HAVE_WIN32_API
    /*
     * On most systems the combination of device and i-node number uniquely
     * identify a file.
     */
    dev_t  device;      /*file device number    */
    ino_t  inode;      /*file i-node number    */
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
static void *read_thread_fn(void *data) {
  ssize_t ret;
  thread_data_t *td = (thread_data_t *)data;

  // fprintf(stderr, "read thread -- ptr: %p, size: %lu, foffset: %ld, doffset: %ld\n",
    // td->rd_devPtr, td->size, td->offset, td->devPtr_offset);

  while( td->size > 0 ) {
    if(td->size > td->block_size) {
      ret = cuFileRead(td->cfr_handle, td->rd_devPtr, td->block_size, td->offset, td->devPtr_offset);
      td->offset += td->block_size;
      td->devPtr_offset += td->block_size;
      td->size -= td->block_size;
    }
    else {
      ret = cuFileRead(td->cfr_handle, td->rd_devPtr, td->size, td->offset, td->devPtr_offset);
      td->size = 0;
    }
    assert(ret > 0);
  }

  // fprintf(stderr, "read success thread -- ptr: %p, size: %lu, foffset: %ld, doffset: %ld\n",
    // td->rd_devPtr, td->size, td->offset, td->devPtr_offset);

  return NULL;
}

static void *write_thread_fn(void *data) {
  ssize_t ret;
  thread_data_t *td = (thread_data_t *)data;

  // fprintf(stderr, "wrt thread -- ptr: %p, size: %lu, foffset: %ld, doffset: %ld\n",
  // td->wr_devPtr, td->size, td->offset, td->devPtr_offset);

  while( td->size > 0 ) {
    if(td->size > td->block_size) {
      ret = cuFileWrite(td->cfr_handle, td->wr_devPtr, td->block_size, td->offset, td->devPtr_offset);
      td->offset += td->block_size;
      td->devPtr_offset += td->block_size;
      td->size -= td->block_size;
    }
    else {
      ret = cuFileWrite(td->cfr_handle, td->wr_devPtr, td->size, td->offset, td->devPtr_offset);
      td->size = 0;
    }
    assert(ret > 0);
  }

  // printf("wrt success thread -- ptr: %p, size: %lu, foffset: %ld, doffset: %ld\n",
    // td->wr_devPtr, td->size, td->offset, td->devPtr_offset);

  return NULL;
}

/* end multiple threads for one io request */

/*
 * These macros check for overflow of various quantities.  These macros
 * assume that HDoff_t is signed and haddr_t and size_t are unsigned.
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
#define MAXADDR (((haddr_t)1 << (8 * sizeof(HDoff_t) - 1)) - 1)
#define ADDR_OVERFLOW(A)  (HADDR_UNDEF == (A) || ((A) & ~(haddr_t)MAXADDR))
#define SIZE_OVERFLOW(Z)  ((Z) & ~(hsize_t)MAXADDR)
#define REGION_OVERFLOW(A,Z)  (ADDR_OVERFLOW(A) || SIZE_OVERFLOW(Z) ||  \
                                 HADDR_UNDEF == (A) + (Z) ||            \
                                 (HDoff_t)((A) + (Z)) < (HDoff_t)(A))

/* Prototypes */
static herr_t H5FD__gds_term(void);
static void *H5FD__gds_fapl_get(H5FD_t *file);
static void *H5FD__gds_fapl_copy(const void *_old_fa);
static H5FD_t *H5FD__gds_open(const char *name, unsigned flags, hid_t fapl_id,
            haddr_t maxaddr);
static herr_t H5FD__gds_close(H5FD_t *_file);
static int H5FD__gds_cmp(const H5FD_t *_f1, const H5FD_t *_f2);
static herr_t H5FD__gds_query(const H5FD_t *_f1, unsigned long *flags);
static haddr_t H5FD__gds_get_eoa(const H5FD_t *_file, H5FD_mem_t type);
static herr_t H5FD__gds_set_eoa(H5FD_t *_file, H5FD_mem_t type, haddr_t addr);
static haddr_t H5FD__gds_get_eof(const H5FD_t *_file, H5FD_mem_t type);
static herr_t  H5FD__gds_get_handle(H5FD_t *_file, hid_t fapl, void** file_handle);
static herr_t H5FD__gds_read(H5FD_t *_file, H5FD_mem_t type, hid_t fapl_id, haddr_t addr,
           size_t size, void *buf);
static herr_t H5FD__gds_write(H5FD_t *_file, H5FD_mem_t type, hid_t fapl_id, haddr_t addr,
            size_t size, const void *buf);
static herr_t H5FD__gds_flush(H5FD_t *_file, hid_t dxpl_id, hbool_t closing);
static herr_t H5FD__gds_truncate(H5FD_t *_file, hid_t dxpl_id, hbool_t closing);
static herr_t H5FD__gds_lock(H5FD_t *_file, hbool_t rw);
static herr_t H5FD__gds_unlock(H5FD_t *_file);


static const H5FD_class_t H5FD_gds_g = {
    "gds",                   /* name                 */
    MAXADDR,                    /* maxaddr              */
    H5F_CLOSE_WEAK,             /* fc_degree            */
    H5FD__gds_term,          /* terminate            */
    NULL,                       /* sb_size              */
    NULL,                       /* sb_encode            */
    NULL,                       /* sb_decode            */
    sizeof(H5FD_gds_fapl_t), /* fapl_size            */
    H5FD__gds_fapl_get,      /* fapl_get             */
    H5FD__gds_fapl_copy,     /* fapl_copy            */
    NULL,                       /* fapl_free            */
    0,                          /* dxpl_size            */
    NULL,                       /* dxpl_copy            */
    NULL,                       /* dxpl_free            */
    H5FD__gds_open,          /* open                 */
    H5FD__gds_close,         /* close                */
    H5FD__gds_cmp,           /* cmp                  */
    H5FD__gds_query,         /* query                */
    NULL,                       /* get_type_map         */
    NULL,                       /* alloc                */
    NULL,                       /* free                 */
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
    H5FD_FLMAP_DICHOTOMY       	/* fl_map               */
};

/* Declare a free list to manage the H5FD_gds_t struct */
H5FL_DEFINE_STATIC(H5FD_gds_t);


/*--------------------------------------------------------------------------
NAME
   H5FD__init_package -- Initialize interface-specific information
USAGE
    herr_t H5FD__init_package()
RETURNS
    Non-negative on success/Negative on failure
DESCRIPTION
    Initializes any interface-specific data or routines.  (Just calls
    H5FD_gds_init currently).

--------------------------------------------------------------------------*/
static herr_t
H5FD__init_package(void)
{
    char    *lock_env_var   = NULL;     /* Environment variable pointer */
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_STATIC

    /* Check the use disabled file locks environment variable */
    lock_env_var = HDgetenv("HDF5_USE_FILE_LOCKING");
    if(lock_env_var && !HDstrcmp(lock_env_var, "BEST_EFFORT"))
        ignore_disabled_file_locks_s = TRUE;    /* Override: Ignore disabled locks */
    else if(lock_env_var && (!HDstrcmp(lock_env_var, "TRUE") || !HDstrcmp(lock_env_var, "1")))
        ignore_disabled_file_locks_s = FALSE;   /* Override: Don't ignore disabled locks */
    else
        ignore_disabled_file_locks_s = FAIL;    /* Environment variable not set, or not set correctly */

    if(H5FD_gds_init() < 0)
        HGOTO_ERROR(H5E_VFL, H5E_CANTINIT, FAIL, "unable to initialize gds VFD")

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* H5FD__init_package() */

/*      ns timer      */
static struct timespec gettime_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC_RAW, &t);
  return t;
}

static struct timespec timediff(struct timespec start, struct timespec stop) {
  struct timespec t;
  t.tv_sec = (stop.tv_sec - start.tv_sec);
  t.tv_nsec = (stop.tv_nsec - start.tv_nsec);
  return t;
}

static void timeprint(const char *msg, struct timespec t) {
  // printf("%s %ld us\n", msg, (t.tv_sec) * 1000000 + (t.tv_nsec) / 1000);
}
//////////////////////////////////////////////////////////////////////////


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
hid_t
H5FD_gds_init(void)
{

#ifdef H5_GDS_SUPPORT
    CUfileError_t status;
    CUfileDescr_t cf_descr;
#endif

    hid_t ret_value = H5I_INVALID_HID;        /* Return value */

    FUNC_ENTER_NOAPI(H5I_INVALID_HID)

#ifdef H5_GDS_SUPPORT
    if(!cu_file_driver_opened) {
      status = cuFileDriverOpen();

      if (status.err == CU_FILE_SUCCESS) {
        cu_file_driver_opened = true;
      }
      else {
        HGOTO_ERROR(H5E_INTERNAL, H5E_SYSTEM, NULL, "unable to open cufile driver");
        // TODO: get the error string once the cufile c api is ready
        //fprintf(stderr, "cufile driver open error: %s\n",
        //  cuFileGetErrorString(status));
      }
    }
#endif

    if (H5I_VFL != H5I_get_type(H5FD_GDS_g))
        H5FD_GDS_g = H5FD_register(&H5FD_gds_g,sizeof(H5FD_class_t),FALSE);

    /* Set return value */
    ret_value = H5FD_GDS_g;

done:
    FUNC_LEAVE_NOAPI(ret_value)
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

#ifdef H5_GDS_SUPPORT
    CUfileError_t status;
    herr_t        ret_value=SUCCEED;       /* Return value */
    FUNC_ENTER_STATIC
#else
    FUNC_ENTER_STATIC_NOERR
#endif

#ifdef H5_GDS_SUPPORT
    if(cu_file_driver_opened) {
      status = cuFileDriverClose();
      if (status.err == CU_FILE_SUCCESS) {
        cu_file_driver_opened = false;
      }
      else {
        HGOTO_ERROR(H5E_INTERNAL, H5E_SYSTEM, NULL, "unable to close cufile driver")
        // TODO: get the error string once the cufile c api is ready
        // fprintf(stderr, "cufile driver close failed: %s\n",
        //   cuFileGetErrorString(status));
      }
    }
#endif

    /* Reset VFL ID */
    H5FD_GDS_g = 0;

#ifdef H5_GDS_SUPPORT
done:
    FUNC_LEAVE_NOAPI(ret_value)
#else
    FUNC_LEAVE_NOAPI(SUCCEED)
#endif

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
 * Programmer:  Raymond Lu
 *    Wednesday, 20 September 2006
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5Pset_fapl_gds(hid_t fapl_id, size_t boundary, size_t block_size, size_t cbuf_size)
{
    H5P_genplist_t      *plist;      /* Property list pointer */
    H5FD_gds_fapl_t  fa;
    herr_t     ret_value;

    FUNC_ENTER_API(FAIL)
    H5TRACE4("e", "izzz", fapl_id, boundary, block_size, cbuf_size);

    if(NULL == (plist = H5P_object_verify(fapl_id,H5P_FILE_ACCESS)))
        HGOTO_ERROR(H5E_ARGS, H5E_BADTYPE, FAIL, "not a file access property list")

    HDmemset(&fa, 0, sizeof(H5FD_gds_fapl_t));
    if(boundary != 0)
        fa.mboundary = boundary;
    else
        fa.mboundary = MBOUNDARY_DEF;
    if(block_size != 0)
        fa.fbsize = block_size;
    else
        fa.fbsize = FBSIZE_DEF;
    if(cbuf_size != 0)
        fa.cbsize = cbuf_size;
    else
        fa.cbsize = CBSIZE_DEF;

    /* Set the default to be true for data alignment */
    fa.must_align = TRUE;

    /* Copy buffer size must be a multiple of file block size */
    if(fa.cbsize % fa.fbsize != 0)
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, FAIL, "copy buffer size must be a multiple of block size")

    ret_value = H5P_set_driver(plist, H5FD_GDS, &fa);

done:
    FUNC_LEAVE_API(ret_value)
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
 * Programmer:  Raymond Lu
 *              Wednesday, October 18, 2006
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5Pget_fapl_gds(hid_t fapl_id, size_t *boundary/*out*/, size_t *block_size/*out*/,
    size_t *cbuf_size/*out*/)
{
    H5P_genplist_t *plist;      /* Property list pointer */
    const H5FD_gds_fapl_t  *fa;
    herr_t      ret_value = SUCCEED;       /* Return value */

    FUNC_ENTER_API(FAIL)
    H5TRACE4("e", "ixxx", fapl_id, boundary, block_size, cbuf_size);

    if(NULL == (plist = H5P_object_verify(fapl_id,H5P_FILE_ACCESS)))
        HGOTO_ERROR(H5E_ARGS, H5E_BADTYPE, FAIL, "not a file access list")
    if(H5FD_GDS != H5P_peek_driver(plist))
        HGOTO_ERROR(H5E_PLIST, H5E_BADVALUE, FAIL, "incorrect VFL driver")
    if(NULL == (fa = H5P_peek_driver_info(plist)))
        HGOTO_ERROR(H5E_PLIST, H5E_BADVALUE, FAIL, "bad VFL driver info")
    if(boundary)
        *boundary = fa->mboundary;
    if(block_size)
        *block_size = fa->fbsize;
    if (cbuf_size)
        *cbuf_size = fa->cbsize;

done:
    FUNC_LEAVE_API(ret_value)
} /* end H5Pget_fapl_gds() */


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
 * Programmer:  Raymond Lu
 *              Wednesday, 18 October 2006
 *
 *-------------------------------------------------------------------------
 */
static void *
H5FD__gds_fapl_get(H5FD_t *_file)
{
    H5FD_gds_t  *file = (H5FD_gds_t*)_file;
    void *ret_value;    /* Return value */

    FUNC_ENTER_STATIC

    /* Set return value */
    ret_value= H5FD__gds_fapl_copy(&(file->fa));

done:
    FUNC_LEAVE_NOAPI(ret_value)
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
 * Programmer:  Raymond Lu
 *              Wednesday, 18 October 2006
 *
 *-------------------------------------------------------------------------
 */
static void *
H5FD__gds_fapl_copy(const void *_old_fa)
{
    const H5FD_gds_fapl_t *old_fa = (const H5FD_gds_fapl_t*)_old_fa;
    H5FD_gds_fapl_t *new_fa = H5MM_calloc(sizeof(H5FD_gds_fapl_t));

    FUNC_ENTER_STATIC_NOERR

    HDassert(new_fa);

    /* Copy the general information */
    H5MM_memcpy(new_fa, old_fa, sizeof(H5FD_gds_fapl_t));

    FUNC_LEAVE_NOAPI(new_fa)
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

#ifdef H5_GDS_SUPPORT
    CUfileError_t status;
    CUfileDescr_t cf_descr;
#endif

    int      o_flags;
    int      fd=(-1);
    H5FD_gds_t  *file=NULL;
    H5FD_gds_fapl_t  *fa;
#ifdef H5_HAVE_WIN32_API
    HFILE     filehandle;
    struct _BY_HANDLE_FILE_INFORMATION fileinfo;
#endif
    h5_stat_t    sb;
    H5P_genplist_t   *plist;      /* Property list */
    void                 *buf1, *buf2;
    H5FD_t    *ret_value = NULL;

    FUNC_ENTER_STATIC

    /* Sanity check on file offsets */
    HDassert(sizeof(HDoff_t)>=sizeof(size_t));

    /* Check arguments */
    if (!name || !*name)
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, NULL, "invalid file name")
    if (0==maxaddr || HADDR_UNDEF==maxaddr)
        HGOTO_ERROR(H5E_ARGS, H5E_BADRANGE, NULL, "bogus maxaddr")
    if (ADDR_OVERFLOW(maxaddr))
        HGOTO_ERROR(H5E_ARGS, H5E_OVERFLOW, NULL, "bogus maxaddr")

    /* Build the open flags */
    o_flags = (H5F_ACC_RDWR & flags) ? O_RDWR : O_RDONLY;
    if (H5F_ACC_TRUNC & flags) o_flags |= O_TRUNC;
    if (H5F_ACC_CREAT & flags) o_flags |= O_CREAT;
    if (H5F_ACC_EXCL & flags) o_flags |= O_EXCL;

    /* Flag for GPUDirect Storage I/O */
    o_flags |= O_DIRECT;

    /* Open the file */
    if ((fd = HDopen(name, o_flags, H5_POSIX_CREATE_MODE_RW))<0)
        HSYS_GOTO_ERROR(H5E_FILE, H5E_CANTOPENFILE, NULL, "unable to open file")

    if (HDfstat(fd, &sb)<0)
        HSYS_GOTO_ERROR(H5E_FILE, H5E_BADFILE, NULL, "unable to fstat file")

    /* Create the new file struct */
    if (NULL==(file=H5FL_CALLOC(H5FD_gds_t)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, NULL, "unable to allocate file struct")

    /* Get the driver specific information */
    if(NULL == (plist = H5P_object_verify(fapl_id,H5P_FILE_ACCESS)))
        HGOTO_ERROR(H5E_ARGS, H5E_BADTYPE, NULL, "not a file access property list")
    if(NULL == (fa = (H5FD_gds_fapl_t *)H5P_peek_driver_info(plist)))
        HGOTO_ERROR(H5E_PLIST, H5E_BADVALUE, NULL, "bad VFL driver info")

#ifdef H5_GDS_SUPPORT
    memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    status = cuFileHandleRegister(&file->cf_handle, &cf_descr);
    if (status.err != CU_FILE_SUCCESS) {
      HGOTO_ERROR(H5E_INTERNAL, H5E_SYSTEM, NULL, "unable to register file with cufile driver");
    }

    // TODO: error checking for num_io_threads
    // FIXME: move to set fapl
    H5Pget( fapl_id, "H5_GDS_VFD_IO_THREADS", &file->num_io_threads );
    H5Pget( fapl_id, "H5_GDS_VFD_IO_BLOCK_SIZE", &file->io_block_size );

    /* IOThreads */
    // TODO: POSSIBLE MEMORY LEAK! figure out how to deal with the the double open H5Fint does
    file->td = (thread_data_t *)HDmalloc((unsigned)file->num_io_threads*sizeof(thread_data_t));
    file->threads = (pthread_t *)HDmalloc((unsigned)file->num_io_threads*sizeof(pthread_t));
#endif
    // TODO: add error print for #elseif 

    file->fd = fd;
    H5_CHECKED_ASSIGN(file->eof, haddr_t, sb.st_size, h5_stat_size_t);
    file->pos = HADDR_UNDEF;
    file->op = OP_UNKNOWN;
#ifdef H5_HAVE_WIN32_API
    filehandle = _get_osfhandle(fd);
    (void)GetFileInformationByHandle((HANDLE)filehandle, &fileinfo);
    file->fileindexhi = fileinfo.nFileIndexHigh;
    file->fileindexlo = fileinfo.nFileIndexLow;
#else
    file->device = sb.st_dev;
    file->inode = sb.st_ino;
#endif /*H5_HAVE_WIN32_API*/
    file->fa.mboundary = fa->mboundary;
    file->fa.fbsize = fa->fbsize;
    file->fa.cbsize = fa->cbsize;

    /* Check the file locking flags in the fapl */
    if(ignore_disabled_file_locks_s != FAIL)
        /* The environment variable was set, so use that preferentially */
        file->ignore_disabled_file_locks = ignore_disabled_file_locks_s;
    else {
        /* Use the value in the property list */
        if(H5P_get(plist, H5F_ACS_IGNORE_DISABLED_FILE_LOCKS_NAME, &file->ignore_disabled_file_locks) < 0)
            HGOTO_ERROR(H5E_VFL, H5E_CANTGET, NULL, "can't get ignore disabled file locks property")
    }

    /* Try to decide if data alignment is required.  The reason to check it here
     * is to handle correctly the case that the file is in a different file system
     * than the one where the program is running.
     */
    /* NOTE: Use HDmalloc and HDfree here to ensure compatibility with
     *       HDposix_memalign.
     */
    buf1 = HDmalloc(sizeof(int));
    if(HDposix_memalign(&buf2, file->fa.mboundary, file->fa.fbsize) != 0)
        HGOTO_ERROR(H5E_RESOURCE, H5E_CANTALLOC, NULL, "HDposix_memalign failed")

    if(o_flags & O_CREAT) {
        if(HDwrite(file->fd, buf1, sizeof(int))<0) {
            if(HDwrite(file->fd, buf2, file->fa.fbsize)<0)
                HGOTO_ERROR(H5E_FILE, H5E_WRITEERROR, NULL, "file system may not support GPUDirect Storage I/O")
            else
                file->fa.must_align = TRUE;
        } else {
            file->fa.must_align = FALSE;
            HDftruncate(file->fd, (HDoff_t)0);
        }
    } else {
        if(HDread(file->fd, buf1, sizeof(int))<0) {
            if(HDread(file->fd, buf2, file->fa.fbsize)<0)
                HGOTO_ERROR(H5E_FILE, H5E_READERROR, NULL, "file system may not support GPUDirect Storage I/O")
            else
                file->fa.must_align = TRUE;
        } else {
            if(o_flags & O_RDWR) {
                if(HDlseek(file->fd, (HDoff_t)0, SEEK_SET) < 0)
                    HSYS_GOTO_ERROR(H5E_IO, H5E_SEEKERROR, NULL, "unable to seek to proper position")
                if(HDwrite(file->fd, buf1, sizeof(int))<0)
                    file->fa.must_align = TRUE;
                else
                    file->fa.must_align = FALSE;
            } else
                file->fa.must_align = FALSE;
        }
    }

    if(buf1)
        HDfree(buf1);
    if(buf2)
        HDfree(buf2);

    /* Set return value */
    ret_value=(H5FD_t*)file;

done:
    if(ret_value==NULL) {
        if(fd>=0)
            HDclose(fd);
    } /* end if */

    FUNC_LEAVE_NOAPI(ret_value)
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

    H5FD_gds_t  *file = (H5FD_gds_t*)_file;
    herr_t        ret_value=SUCCEED;       /* Return value */

    FUNC_ENTER_STATIC

#ifdef H5_GDS_SUPPORT
    // close file handle
    cuFileHandleDeregister(file->cf_handle);
#endif

    if (HDclose(file->fd)<0)
        HSYS_GOTO_ERROR(H5E_IO, H5E_CANTCLOSEFILE, FAIL, "unable to close file")

#ifdef H5_GDS_SUPPORT
    if(file->td)
      HDfree(file->td);

    if(file->threads)
      HDfree(file->threads);
#endif

    H5FL_FREE(H5FD_gds_t,file);

done:
    FUNC_LEAVE_NOAPI(ret_value)
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
 * Programmer:  Raymond Lu
 *              Thursday, 21 September 2006
 *
 *-------------------------------------------------------------------------
 */
static int
H5FD__gds_cmp(const H5FD_t *_f1, const H5FD_t *_f2)
{
    const H5FD_gds_t  *f1 = (const H5FD_gds_t*)_f1;
    const H5FD_gds_t  *f2 = (const H5FD_gds_t*)_f2;
    int ret_value=0;

    FUNC_ENTER_STATIC_NOERR

#ifdef H5_HAVE_WIN32_API
    if (f1->fileindexhi < f2->fileindexhi) HGOTO_DONE(-1)
    if (f1->fileindexhi > f2->fileindexhi) HGOTO_DONE(1)

    if (f1->fileindexlo < f2->fileindexlo) HGOTO_DONE(-1)
    if (f1->fileindexlo > f2->fileindexlo) HGOTO_DONE(1)

#else
#ifdef H5_DEV_T_IS_SCALAR
    if (f1->device < f2->device) HGOTO_DONE(-1)
    if (f1->device > f2->device) HGOTO_DONE(1)
#else /* H5_DEV_T_IS_SCALAR */
    /* If dev_t isn't a scalar value on this system, just use memcmp to
     * determine if the values are the same or not.  The actual return value
     * shouldn't really matter...
     */
    if(HDmemcmp(&(f1->device),&(f2->device),sizeof(dev_t))<0) HGOTO_DONE(-1)
    if(HDmemcmp(&(f1->device),&(f2->device),sizeof(dev_t))>0) HGOTO_DONE(1)
#endif /* H5_DEV_T_IS_SCALAR */

    if (f1->inode < f2->inode) HGOTO_DONE(-1)
    if (f1->inode > f2->inode) HGOTO_DONE(1)

#endif

done:
    FUNC_LEAVE_NOAPI(ret_value)
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
H5FD__gds_query(const H5FD_t H5_ATTR_UNUSED * _f, unsigned long *flags /* out */)
{
    FUNC_ENTER_STATIC_NOERR

    /* Set the VFL feature flags that this driver supports */
    if(flags) {
        *flags = 0;
        *flags |= H5FD_FEAT_AGGREGATE_METADATA;     /* OK to aggregate metadata allocations                             */
        *flags |= H5FD_FEAT_ACCUMULATE_METADATA;    /* OK to accumulate metadata for faster writes                      */
        *flags |= H5FD_FEAT_AGGREGATE_SMALLDATA;    /* OK to aggregate "small" raw data allocations                     */
        *flags |= H5FD_FEAT_SUPPORTS_SWMR_IO;       /* VFD supports the single-writer/multiple-readers (SWMR) pattern   */
        *flags |= H5FD_FEAT_DEFAULT_VFD_COMPATIBLE; /* VFD creates a file which can be opened with the default VFD      */
    }

    FUNC_LEAVE_NOAPI(SUCCEED)
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
 * Programmer:  Raymond Lu
 *              Wednesday, 20 September 2006
 *
 *-------------------------------------------------------------------------
 */
static haddr_t
H5FD__gds_get_eoa(const H5FD_t *_file, H5FD_mem_t H5_ATTR_UNUSED type)
{
    const H5FD_gds_t  *file = (const H5FD_gds_t*)_file;

    FUNC_ENTER_STATIC_NOERR

    FUNC_LEAVE_NOAPI(file->eoa)
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
 * Programmer:  Raymond Lu
 *              Wednesday, 20 September 2006
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5FD__gds_set_eoa(H5FD_t *_file, H5FD_mem_t H5_ATTR_UNUSED type, haddr_t addr)
{
    H5FD_gds_t  *file = (H5FD_gds_t*)_file;

    FUNC_ENTER_STATIC_NOERR

    file->eoa = addr;

    FUNC_LEAVE_NOAPI(SUCCEED)
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
 * Programmer:  Raymond Lu
 *              Wednesday, 20 September 2006
 *
 *-------------------------------------------------------------------------
 */
static haddr_t
H5FD__gds_get_eof(const H5FD_t *_file, H5FD_mem_t H5_ATTR_UNUSED type)
{
    const H5FD_gds_t  *file = (const H5FD_gds_t*)_file;

    FUNC_ENTER_STATIC

    FUNC_LEAVE_NOAPI(file->eof)
}


/*-------------------------------------------------------------------------
 * Function:       H5FD_gds_get_handle
 *
 * Purpose:        Returns the file handle of gds file driver.
 *
 * Returns:        Non-negative if succeed or negative if fails.
 *
 * Programmer:     Raymond Lu
 *                 21 September 2006
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5FD__gds_get_handle(H5FD_t *_file, hid_t H5_ATTR_UNUSED fapl, void** file_handle)
{
    H5FD_gds_t       *file = (H5FD_gds_t *)_file;
    herr_t              ret_value = SUCCEED;

    FUNC_ENTER_STATIC

    if(!file_handle)
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, FAIL, "file handle not valid")
    *file_handle = &(file->fd);

done:
    FUNC_LEAVE_NOAPI(ret_value)
}

bool is_device_pointer (const void *ptr);
bool is_device_pointer (const void *ptr) 
{
  struct cudaPointerAttributes attributes;
  cudaPointerGetAttributes (&attributes, ptr);
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
H5FD__gds_read(H5FD_t *_file, H5FD_mem_t H5_ATTR_UNUSED type, hid_t H5_ATTR_UNUSED dxpl_id, haddr_t addr,
         size_t size, void *buf/*out*/)
{
    H5FD_gds_t  *file = (H5FD_gds_t*)_file;
    ssize_t    nbytes;
    hbool_t    _must_align = TRUE;
    herr_t        ret_value=SUCCEED;       /* Return value */
    size_t    alloc_size;
    void    *copy_buf = NULL, *p2;
    size_t    _boundary;
    size_t    _fbsize;
    size_t    _cbsize;
    haddr_t    read_size;              /* Size to read into copy buffer */
    size_t    copy_size = size;       /* Size remaining to read when using copy buffer */
    size_t              copy_offset;            /* Offset into copy buffer of the requested data */

#ifdef H5_GDS_SUPPORT
    CUfileError_t status;
    ssize_t ret = -1;
    struct timespec start_time, stop_time;

    int io_threads = file->num_io_threads;
    int block_size = file->io_block_size;

    HDoff_t         offset      = (HDoff_t)addr;
    ssize_t io_chunk;
    ssize_t io_chunk_rem;
#endif

    FUNC_ENTER_STATIC

    HDassert(file && file->pub.cls);
    HDassert(buf);

    /* Check for overflow conditions */
    if (HADDR_UNDEF==addr)
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, FAIL, "addr undefined")
    if (REGION_OVERFLOW(addr, size))
        HGOTO_ERROR(H5E_ARGS, H5E_OVERFLOW, FAIL, "addr overflow")

#ifdef H5_GDS_SUPPORT
    if(is_device_pointer(buf)) 
    {
      // TODO: register device memory only once
      status = cuFileBufRegister(buf, size, 0);
      if (status.err != CU_FILE_SUCCESS) {
        HGOTO_ERROR(H5E_INTERNAL, H5E_SYSTEM, NULL, "cufile buffer register failed");
      }

      if( io_threads > 0 ) {
        assert(size != 0);

        // make each thread access at least a 4K page
        if( (1 + (size-1)/4096) < (unsigned)io_threads ) {
          io_threads = (int) (1 + ((size-1)/4096));
        }

        // printf("\tH5Pset_gds_read using io_threads: %d\n", io_threads);
        // printf("\tH5Pset_gds_read using io_block_size: %d\n", block_size);

        io_chunk = (unsigned)size / (unsigned)io_threads;
        io_chunk_rem = (unsigned)size % (unsigned)io_threads;

        for (int ii = 0; ii < io_threads; ii++) {
          file->td[ii].rd_devPtr = buf;
          file->td[ii].cfr_handle = file->cf_handle;

          file->td[ii].offset = (off_t)(offset + ii*io_chunk);
          file->td[ii].devPtr_offset = (off_t)ii*io_chunk;
          file->td[ii].size = (size_t)io_chunk;
          file->td[ii].block_size = block_size;

          if(ii == io_threads-1) {
            file->td[ii].size = (size_t)(io_chunk + io_chunk_rem);
          }
        }

        start_time = gettime_ms();
        for (int ii = 0; ii < io_threads; ii++) {
          pthread_create(&file->threads[ii], NULL, &read_thread_fn, &file->td[ii]);
        }

        for (int ii = 0; ii < io_threads; ii++) {
          pthread_join(file->threads[ii], NULL);
        }
        stop_time = gettime_ms();
        timeprint( "pthread_time:", timediff(start_time, stop_time) );
      }
      else {
        start_time = gettime_ms();
        ret = cuFileRead(file->cf_handle, buf, size, offset, 0);
        stop_time = gettime_ms();
        assert(ret > 0);

        timeprint( "cuFileRead:", timediff(start_time, stop_time) );
      }

      // TODO: deregister device memory only once
      status = cuFileBufDeregister(buf);
      if (status.err != CU_FILE_SUCCESS) {
        HGOTO_ERROR(H5E_INTERNAL, H5E_SYSTEM, NULL, "cufile buffer deregister failed");
      }
    }
    else {
#endif
      /* If the system doesn't require data to be aligned, read the data in
       * the same way as sec2 driver.
       */
      _must_align = file->fa.must_align;

      /* Get the memory boundary for alignment, file system block size, and maximal
       * copy buffer size.
       */
      _boundary = file->fa.mboundary;
      _fbsize = file->fa.fbsize;
      _cbsize = file->fa.cbsize;

      /* if the data is aligned or the system doesn't require data to be aligned,
       * read it directly from the file.  If not, read a bigger
       * and aligned data first, then copy the data into memory buffer.
       */
      if(!_must_align || ((addr%_fbsize==0) && (size%_fbsize==0) && ((size_t)buf%_boundary==0))) {
        /* Seek to the correct location */
        if ((addr!=file->pos || OP_READ!=file->op) &&
          HDlseek(file->fd, (HDoff_t)addr, SEEK_SET)<0)
      HSYS_GOTO_ERROR(H5E_IO, H5E_SEEKERROR, FAIL, "unable to seek to proper position")
         /* Read the aligned data in file first, being careful of interrupted
         * system calls and partial results. */
        while (size>0) {
      do {
          nbytes = HDread(file->fd, buf, size);
      } while (-1==nbytes && EINTR==errno);
      if (-1==nbytes) /* error */
          HSYS_GOTO_ERROR(H5E_IO, H5E_READERROR, FAIL, "file read failed")
      if (0==nbytes) {
          /* end of file but not end of format address space */
          HDmemset(buf, 0, size);
          break;
      }
      HDassert(nbytes>=0);
      HDassert((size_t)nbytes<=size);
      H5_CHECK_OVERFLOW(nbytes,ssize_t,size_t);
      size -= (size_t)nbytes;
      H5_CHECK_OVERFLOW(nbytes,ssize_t,haddr_t);
      addr += (haddr_t)nbytes;
      buf = (char*)buf + nbytes;
        }
      } else {
              /* Calculate where we will begin copying from the copy buffer */
              copy_offset = (size_t)(addr % _fbsize);

        /* allocate memory needed for the GPUDirect Storage IO option up to the maximal
         * copy buffer size. Make a bigger buffer for aligned I/O if size is
         * smaller than maximal copy buffer. */
        alloc_size = ((copy_offset + size - 1) / _fbsize + 1) * _fbsize;
        if(alloc_size > _cbsize)
          alloc_size = _cbsize;
        HDassert(!(alloc_size % _fbsize));
        if (HDposix_memalign(&copy_buf, _boundary, alloc_size) != 0)
      HGOTO_ERROR(H5E_RESOURCE, H5E_CANTALLOC, FAIL, "HDposix_memalign failed")

              /* look for the aligned position for reading the data */
              HDassert(!(((addr / _fbsize) * _fbsize) % _fbsize));
              if(HDlseek(file->fd, (HDoff_t)((addr / _fbsize) * _fbsize),
                      SEEK_SET) < 0)
                  HSYS_GOTO_ERROR(H5E_IO, H5E_SEEKERROR, FAIL, "unable to seek to proper position")

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
          HDmemset(copy_buf, 0, alloc_size);

                  /* Calculate how much data we have to read in this iteration
                   * (including unused parts of blocks) */
                  if((copy_size + copy_offset) < alloc_size)
                      read_size = ((copy_size + copy_offset - 1) / _fbsize + 1)
                              * _fbsize;
                  else
                      read_size = alloc_size;

                  HDassert(!(read_size % _fbsize));
          do {
          nbytes = HDread(file->fd, copy_buf, read_size);
          } while(-1==nbytes && EINTR==errno);

          if (-1==nbytes) /* error */
          HSYS_GOTO_ERROR(H5E_IO, H5E_READERROR, FAIL, "file read failed")

          /* Copy the needed data from the copy buffer to the output
           * buffer, and update copy_size.  If the copy buffer does not
                   * contain the rest of the data, just copy what's in the copy
                   * buffer and also update read_addr and copy_offset to read the
                   * next section of data. */
          p2 = (unsigned char*)copy_buf + copy_offset;
          if((copy_size + copy_offset) <= alloc_size) {
              H5MM_memcpy(buf, p2, copy_size);
              buf = (unsigned char *)buf + copy_size;
              copy_size = 0;
                  } /* end if */
                  else {
                      H5MM_memcpy(buf, p2, alloc_size - copy_offset);
                      buf = (unsigned char*)buf + alloc_size - copy_offset;
                      copy_size -= alloc_size - copy_offset;
                      copy_offset = 0;
                  } /* end else */
        } while (copy_size > 0);

        /*Final step: update address*/
        addr = (haddr_t)(((addr + size - 1) / _fbsize + 1) * _fbsize);

        if(copy_buf) {
                  /* Free with HDfree since it came from posix_memalign */
                  HDfree(copy_buf);
                  copy_buf = NULL;
              } /* end if */
      }

      /* Update current position */
      file->pos = addr;
      file->op = OP_READ;
#ifdef H5_GDS_SUPPORT
    }
#endif

done:
    if(ret_value<0) {
        /* Free with HDfree since it came from posix_memalign */
        if(copy_buf)
            HDfree(copy_buf);

        /* Reset last file I/O information */
        file->pos = HADDR_UNDEF;
        file->op = OP_UNKNOWN;
    } /* end if */

    FUNC_LEAVE_NOAPI(ret_value)
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
H5FD__gds_write(H5FD_t *_file, H5FD_mem_t H5_ATTR_UNUSED type, hid_t H5_ATTR_UNUSED dxpl_id, haddr_t addr,
    size_t size, const void *buf)
{
    H5FD_gds_t  *file = (H5FD_gds_t*)_file;
    ssize_t    nbytes;
    hbool_t    _must_align = TRUE;
    herr_t        ret_value=SUCCEED;       /* Return value */
    size_t    alloc_size;
    void    *copy_buf = NULL, *p1;
    const void    *p3;
    size_t    _boundary;
    size_t    _fbsize;
    size_t    _cbsize;
    haddr_t             write_addr;             /* Address to write copy buffer */
    haddr_t             write_size;             /* Size to write from copy buffer */
    haddr_t             read_size;              /* Size to read into copy buffer */
    size_t              copy_size = size;       /* Size remaining to write when using copy buffer */
    size_t              copy_offset;            /* Offset into copy buffer of the data to write */

#ifdef H5_GDS_SUPPORT
    CUfileError_t status;
    ssize_t ret = -1;
    struct timespec start_time, stop_time;

    int io_threads = file->num_io_threads;
    int block_size = file->io_block_size;

    ssize_t io_chunk;
    ssize_t io_chunk_rem;

    HDoff_t         offset      = (HDoff_t)addr;

    printf("HERE: %s:%d\n", __FILE__, __LINE__);
#endif

    FUNC_ENTER_STATIC

    HDassert(file && file->pub.cls);
    HDassert(buf);

    /* Check for overflow conditions */
    if (HADDR_UNDEF==addr)
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, FAIL, "addr undefined")
    if (REGION_OVERFLOW(addr, size))
        HGOTO_ERROR(H5E_ARGS, H5E_OVERFLOW, FAIL, "addr overflow")

#ifdef H5_GDS_SUPPORT
    if(is_device_pointer(buf)) 
    {
      printf("\tH5Pset_gds_write using io_threads: %d\n", io_threads);
      printf("\tH5Pset_gds_write using io_block_size: %d\n", block_size);
      fflush(stdout);

      // TODO: registers device memory only once
      status = cuFileBufRegister(buf, size, 0);
      if (status.err != CU_FILE_SUCCESS) {
        HGOTO_ERROR(H5E_INTERNAL, H5E_SYSTEM, NULL, "cufile buffer register failed");
      }

      if( io_threads > 0 ) {
        assert(size != 0);

        // make each thread access at least a 4K page
        if( (1 + (size-1)/4096) < (unsigned)io_threads ) {
          io_threads = (int) (1 + ((size-1)/4096));
        }

        // printf("\tH5Pset_gds_write using io_threads: %d\n", io_threads);
        // printf("\tH5Pset_gds_write using io_block_size: %d\n", block_size);

        io_chunk = (unsigned)size / (unsigned)io_threads;
        io_chunk_rem = (unsigned)size % (unsigned)io_threads;

        for (int ii = 0; ii < io_threads; ii++) {
          file->td[ii].wr_devPtr = buf;
          file->td[ii].cfr_handle = file->cf_handle;

          file->td[ii].offset = (off_t)(offset + ii*io_chunk);
          file->td[ii].devPtr_offset = (off_t)ii*io_chunk;
          file->td[ii].size = (size_t)io_chunk;
          file->td[ii].block_size = block_size;

          if(ii == io_threads-1) {
            file->td[ii].size = (size_t)(io_chunk + io_chunk_rem);
          }
        }

        start_time = gettime_ms();
        for (int ii = 0; ii < io_threads; ii++) {
          pthread_create(&file->threads[ii], NULL, &write_thread_fn, &file->td[ii]);
        }

        for (int ii = 0; ii < io_threads; ii++) {
          pthread_join(file->threads[ii], NULL);
        }
        stop_time = gettime_ms();
        timeprint( "pthread_time:", timediff(start_time, stop_time) );
      }
      else {
        start_time = gettime_ms();
        ret = cuFileWrite(file->cf_handle, buf, size, offset, 0);
        stop_time = gettime_ms();
        assert(ret > 0);

        timeprint( "cuFileWrite:", timediff(start_time, stop_time) );
      }

      // TODO: deregister device memory only once
      status = cuFileBufDeregister(buf);
      if (status.err != CU_FILE_SUCCESS) {
        HGOTO_ERROR(H5E_INTERNAL, H5E_SYSTEM, NULL, "cufile buffer deregister failed");
      }
    }
    else {
#endif
      /* If the system doesn't require data to be aligned, read the data in
       * the same way as sec2 driver.
       */
      _must_align = file->fa.must_align;

      /* Get the memory boundary for alignment, file system block size, and maximal
       * copy buffer size.
       */
      _boundary = file->fa.mboundary;
      _fbsize = file->fa.fbsize;
      _cbsize = file->fa.cbsize;

      /* if the data is aligned or the system doesn't require data to be aligned,
       * write it directly to the file.  If not, read a bigger and aligned data
       * first, update buffer with user data, then write the data out.
       */
      if(!_must_align || ((addr%_fbsize==0) && (size%_fbsize==0) && ((size_t)buf%_boundary==0))) {
        /* Seek to the correct location */
        if ((addr!=file->pos || OP_WRITE!=file->op) &&
          HDlseek(file->fd, (HDoff_t)addr, SEEK_SET)<0)
      HSYS_GOTO_ERROR(H5E_IO, H5E_SEEKERROR, FAIL, "unable to seek to proper position")

        while (size>0) {
      do {
          nbytes = HDwrite(file->fd, buf, size);
      } while (-1==nbytes && EINTR==errno);
      if (-1==nbytes) /* error */
          HSYS_GOTO_ERROR(H5E_IO, H5E_WRITEERROR, FAIL, "file write failed")
      HDassert(nbytes>0);
      HDassert((size_t)nbytes<=size);
      H5_CHECK_OVERFLOW(nbytes,ssize_t,size_t);
      size -= (size_t)nbytes;
      H5_CHECK_OVERFLOW(nbytes,ssize_t,haddr_t);
      addr += (haddr_t)nbytes;
      buf = (const char*)buf + nbytes;
        }
      } else {
              /* Calculate where we will begin reading from (on disk) and where we
               * will begin copying from the copy buffer */
              write_addr = (addr / _fbsize) * _fbsize;
              copy_offset = (size_t)(addr % _fbsize);

        /* allocate memory needed for the GPUDirect Storage IO option up to the maximal
         * copy buffer size. Make a bigger buffer for aligned I/O if size is
         * smaller than maximal copy buffer.
         */
        alloc_size = ((copy_offset + size - 1) / _fbsize + 1) * _fbsize;
              if(alloc_size > _cbsize)
                  alloc_size = _cbsize;
              HDassert(!(alloc_size % _fbsize));

        if (HDposix_memalign(&copy_buf, _boundary, alloc_size) != 0)
      HGOTO_ERROR(H5E_RESOURCE, H5E_CANTALLOC, FAIL, "HDposix_memalign failed")

              /* look for the right position for reading or writing the data */
              if(HDlseek(file->fd, (HDoff_t)write_addr, SEEK_SET) < 0)
                  HSYS_GOTO_ERROR(H5E_IO, H5E_SEEKERROR, FAIL, "unable to seek to proper position")

        p3 = buf;
        do {
                  /* Calculate how much data we have to write in this iteration
                   * (including unused parts of blocks) */
                  if((copy_size + copy_offset) < alloc_size)
                      write_size = ((copy_size + copy_offset - 1) / _fbsize + 1)
                              * _fbsize;
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
          HDmemset(copy_buf, 0, _fbsize);

                  if(copy_offset > 0) {
                      if((write_addr + write_size) > (addr + size)) {
                          HDassert((write_addr + write_size) - (addr + size) < _fbsize);
                          read_size = write_size;
                          p1 = copy_buf;
                      } /* end if */
                      else {
                          read_size = _fbsize;
                          p1 = copy_buf;
                      } /* end else */
                  } /* end if */
                  else if((write_addr + write_size) > (addr + size)) {
                      HDassert((write_addr + write_size) - (addr + size) < _fbsize);
                      read_size = _fbsize;
                      p1 = (unsigned char *)copy_buf + write_size - _fbsize;

                      /* Seek to the last block, for reading */
                      HDassert(!((write_addr + write_size - _fbsize) % _fbsize));
                      if(HDlseek(file->fd,
                              (HDoff_t)(write_addr + write_size - _fbsize),
                              SEEK_SET) < 0)
                          HSYS_GOTO_ERROR(H5E_IO, H5E_SEEKERROR, FAIL, "unable to seek to proper position")
                  } /* end if */
                  else
                      p1 = NULL;

                  if(p1) {
                      HDassert(!(read_size % _fbsize));
                      do {
                          nbytes = HDread(file->fd, p1, read_size);
                      } while (-1==nbytes && EINTR==errno);

                      if (-1==nbytes) /* error */
                          HSYS_GOTO_ERROR(H5E_IO, H5E_READERROR, FAIL, "file read failed")
                  } /* end if */

          /* look for the right position and append or copy the data to be written to
            * the aligned buffer.
                 * Consider all possible situations here: file address is not aligned on
                 * file block size; the end of data address is not aligned; the end of data
                 * address is aligned; data size is smaller or bigger than maximal copy size.
           */
          p1 = (unsigned char *)copy_buf + copy_offset;
          if((copy_size + copy_offset) <= alloc_size) {
                      H5MM_memcpy(p1, p3, copy_size);
                      copy_size = 0;
                  } /* end if */
                  else {
                      H5MM_memcpy(p1, p3, alloc_size - copy_offset);
                      p3 = (const unsigned char *)p3 + (alloc_size - copy_offset);
                      copy_size -= alloc_size - copy_offset;
                      copy_offset = 0;
                  } /* end else */

          /*look for the aligned position for writing the data*/
          HDassert(!(write_addr % _fbsize));
          if(HDlseek(file->fd, (HDoff_t)write_addr, SEEK_SET) < 0)
          HSYS_GOTO_ERROR(H5E_IO, H5E_SEEKERROR, FAIL, "unable to seek to proper position")

          /*
            * Write the data. It doesn't truncate the extra data introduced by
       * alignment because that step is done in H5FD_gds_flush.
           */
          HDassert(!(write_size % _fbsize));
      do {
          nbytes = HDwrite(file->fd, copy_buf, write_size);
      } while (-1==nbytes && EINTR==errno);

      if (-1==nbytes) /* error */
          HSYS_GOTO_ERROR(H5E_IO, H5E_WRITEERROR, FAIL, "file write failed")

                /* update the write address */
                write_addr += write_size;
    } while (copy_size > 0);

    /*Update the address and size*/
    addr = write_addr;
    buf = (const char*)buf + size;

    if(copy_buf) {
      /* Free with HDfree since it came from posix_memalign */
        HDfree(copy_buf);
        copy_buf = NULL;
          } /* end if */
      }

      /* Update current position and eof */
      file->pos = addr;
      file->op = OP_WRITE;
      if (file->pos>file->eof)
          file->eof = file->pos;
#ifdef H5_GDS_SUPPORT
    }
#endif

done:
    if(ret_value<0) {
        /* Free with HDfree since it came from posix_memalign */
        if(copy_buf)
            HDfree(copy_buf);

        /* Reset last file I/O information */
        file->pos = HADDR_UNDEF;
        file->op = OP_UNKNOWN;
    } /* end if */

    FUNC_LEAVE_NOAPI(ret_value)
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
    H5FD_gds_t  *file = (H5FD_gds_t*)_file;	/* VFD file struct */
    herr_t ret_value = SUCCEED;                 	/* Return value */

    FUNC_ENTER_STATIC

    HDassert(file);
    if(fsync(file->fd) < 0) {
        HSYS_GOTO_ERROR(H5E_VFL, H5E_CANTFLUSH, FAIL, "unable perform fsync on file descriptor")
    }

done:
    FUNC_LEAVE_NOAPI(ret_value)
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
 * Programmer:  Raymond Lu
 *              Thursday, 21 September 2006
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5FD__gds_truncate(H5FD_t *_file, hid_t H5_ATTR_UNUSED dxpl_id, hbool_t H5_ATTR_UNUSED closing)
{
    H5FD_gds_t  *file = (H5FD_gds_t*)_file;
    herr_t        ret_value = SUCCEED;       /* Return value */

    FUNC_ENTER_STATIC

    HDassert(file);

    /* Extend the file to make sure it's large enough */
    if (file->eoa!=file->eof) {
#ifdef H5_HAVE_WIN32_API
        HFILE filehandle;   /* Windows file handle */
        LARGE_INTEGER li;   /* 64-bit integer for SetFilePointer() call */

        /* Map the posix file handle to a Windows file handle */
        filehandle = _get_osfhandle(file->fd);

        /* Translate 64-bit integers into form Windows wants */
        /* [This algorithm is from the Windows documentation for SetFilePointer()] */
        li.QuadPart = (LONGLONG)file->eoa;
        (void)SetFilePointer((HANDLE)filehandle,li.LowPart,&li.HighPart,FILE_BEGIN);
        if(SetEndOfFile((HANDLE)filehandle)==0)
            HGOTO_ERROR(H5E_IO, H5E_SEEKERROR, FAIL, "unable to extend file properly")
#else /* H5_HAVE_WIN32_API */
        if (-1==HDftruncate(file->fd, (HDoff_t)file->eoa))
            HSYS_GOTO_ERROR(H5E_IO, H5E_SEEKERROR, FAIL, "unable to extend file properly")
#endif /* H5_HAVE_WIN32_API */

        /* Update the eof value */
        file->eof = file->eoa;

        /* Reset last file I/O information */
        file->pos = HADDR_UNDEF;
        file->op = OP_UNKNOWN;
    }
    else if (file->fa.must_align){
  /*Even though eof is equal to eoa, file is still truncated because GPUDirect Storage I/O
   *write introduces some extra data for alignment.
   */
        if (-1==HDftruncate(file->fd, (HDoff_t)file->eof))
            HSYS_GOTO_ERROR(H5E_IO, H5E_SEEKERROR, FAIL, "unable to extend file properly")
    }

done:
    FUNC_LEAVE_NOAPI(ret_value)
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
    H5FD_gds_t *file = (H5FD_gds_t*)_file;    /* VFD file struct      */
    int lock_flags;                                 /* file locking flags   */
    herr_t ret_value = SUCCEED;                     /* Return value         */

    FUNC_ENTER_STATIC

    HDassert(file);

    /* Set exclusive or shared lock based on rw status */
    lock_flags = rw ? LOCK_EX : LOCK_SH;

    /* Place a non-blocking lock on the file */
    if(HDflock(file->fd, lock_flags | LOCK_NB) < 0) {
        if(file->ignore_disabled_file_locks && ENOSYS == errno) {
            /* When errno is set to ENOSYS, the file system does not support
             * locking, so ignore it.
             */
            errno = 0;
        }
        else
            HSYS_GOTO_ERROR(H5E_VFL, H5E_CANTLOCKFILE, FAIL, "unable to lock file")
    }

done:
    FUNC_LEAVE_NOAPI(ret_value)
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
    H5FD_gds_t  *file = (H5FD_gds_t*)_file;	/* VFD file struct */
    herr_t ret_value = SUCCEED;                 	/* Return value */

    FUNC_ENTER_STATIC

    HDassert(file);

    if(HDflock(file->fd, LOCK_UN) < 0) {
        if(file->ignore_disabled_file_locks && ENOSYS == errno) {
            /* When errno is set to ENOSYS, the file system does not support
             * locking, so ignore it.
             */
            errno = 0;
        }
        else
            HSYS_GOTO_ERROR(H5E_VFL, H5E_CANTUNLOCKFILE, FAIL, "unable to unlock file")
    }

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5FD__gds_unlock() */

