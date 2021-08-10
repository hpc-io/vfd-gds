#include <stdlib.h>
#include <assert.h>

#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "hdf5.h"

#define FILENAME    "gds_vfd_test.h5"
#define MAINPROCESS (!mpi_rank) /* define process 0 as main process */

/* Constants definitions */
#define RANK        2
#define DIM0 600
#define DIM1 1200
#define CHUNK_DIM0 ((DIM0 + 9) / 10)
#define CHUNK_DIM1 ((DIM1 + 9) / 10)
#define DATASETNAME1 "Data1"
#define DATASETNAME2 "Data2"
#define DATASETNAME3 "Data3"
#define DATASETNAME4 "Data4"
#define DATASETNAME5 "Data5"
#define DATASETNAME6 "Data6"
#define DATASETNAME7 "Data7"
#define DATASETNAME8 "Data8"
#define DATASETNAME9 "Data9"
#define MAX_ERR_REPORT 10 /* Maximum number of errors reported */

/* point selection order */
#define IN_ORDER     1
#define OUT_OF_ORDER 2

/* Hyperslab layout styles */
#define BYROW 1 /* divide into slabs of rows */
#define BYCOL 2 /* divide into blocks of columns */
#define ZROW  3 /* same as BYCOL except process 0 gets 0 rows */
#define ZCOL  4 /* same as BYCOL except process 0 gets 0 columns */

#define DXFER_COLLECTIVE_IO  0x1 /* Collective IO*/
#define DXFER_INDEPENDENT_IO 0x2 /* Independent IO collectively */

/*
 * VRFY: Verify if the condition val is true.
 * If val is not true, it prints error messages and calls MPI_Abort
 * to abort the program.
 */
#define VRFY_IMPL(val, mesg, rankvar)                                                                        \
    do {                                                                                                     \
        if (!val) {                                                                                          \
            printf("Proc %d: ", rankvar);                                                                    \
            printf("*** Parallel ERROR ***\n");                                                              \
            printf("    VRFY (%s) failed at line %4d in %s\n", mesg, (int)__LINE__, __FILE__);               \
            ++nerrors;                                                                                       \
            fflush(stdout);                                                                                  \
            printf("aborting MPI processes\n");                                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                    \
        }                                                                                                    \
    } while (0)

#define VRFY_G(val, mesg) VRFY_IMPL(val, mesg, mpi_rank_g)
#define VRFY(val, mesg)   VRFY_IMPL(val, mesg, mpi_rank)

#define H5FD_GDS_UNUSED(param) (void)(param)

typedef int DATATYPE;

static MPI_Comm comm = MPI_COMM_WORLD;
static MPI_Info info = MPI_INFO_NULL;
static int      mpi_rank;
static int      mpi_size;
int             nerrors = 0;
int             dxfer_coll_type = DXFER_COLLECTIVE_IO;

static void extend_writeInd_cuda(void);
static void dataset_writeAll_cuda(void);
static void dataset_readAll_cuda(void);
static void dataset_writeInd_cuda(void);
static void dataset_readInd_cuda(void);
static void extend_writeInd2_cuda(void);

/*
 * Tests for the HDF5 GDS VFD.
 */

/*
 * The following are various utility routines used by the tests.
 */

/*
 * Setup the dimensions of the hyperslab.
 * Two modes--by rows or by columns.
 * Assume dimension rank is 2.
 * BYROW    divide into slabs of rows
 * BYCOL    divide into blocks of columns
 * ZROW        same as BYROW except process 0 gets 0 rows
 * ZCOL        same as BYCOL except process 0 gets 0 columns
 */
static void
slab_set(int rank, int comm_size, hsize_t start[], hsize_t count[], hsize_t stride[], hsize_t block[],
         int mode)
{
    switch (mode) {
        case BYROW:
            /* Each process takes a slabs of rows. */
            block[0]  = (hsize_t)(DIM0 / comm_size);
            block[1]  = (hsize_t)DIM1;
            stride[0] = block[0];
            stride[1] = block[1];
            count[0]  = 1;
            count[1]  = 1;
            start[0]  = (hsize_t)rank * block[0];
            start[1]  = 0;
            break;
        case BYCOL:
            /* Each process takes a block of columns. */
            block[0]  = (hsize_t)DIM0;
            block[1]  = (hsize_t)(DIM1 / comm_size);
            stride[0] = block[0];
            stride[1] = block[1];
            count[0]  = 1;
            count[1]  = 1;
            start[0]  = 0;
            start[1]  = (hsize_t)rank * block[1];
            break;
        case ZROW:
            /* Similar to BYROW except process 0 gets 0 row */
            block[0]  = (hsize_t)(rank ? DIM0 / comm_size : 0);
            block[1]  = (hsize_t)DIM1;
            stride[0] = (rank ? block[0] : 1); /* avoid setting stride to 0 */
            stride[1] = block[1];
            count[0]  = 1;
            count[1]  = 1;
            start[0]  = (rank ? (hsize_t)rank * block[0] : 0);
            start[1]  = 0;
            break;
        case ZCOL:
            /* Similar to BYCOL except process 0 gets 0 column */
            block[0]  = (hsize_t)DIM0;
            block[1]  = (hsize_t)(rank ? DIM1 / comm_size : 0);
            stride[0] = block[0];
            stride[1] = (hsize_t)(rank ? block[1] : 1); /* avoid setting stride to 0 */
            count[0]  = 1;
            count[1]  = 1;
            start[0]  = 0;
            start[1]  = (rank ? (hsize_t)rank * block[1] : 0);
            break;
        default:
            /* Unknown mode.  Set it to cover the whole dataset. */
            printf("unknown slab_set mode (%d)\n", mode);
            block[0]  = (hsize_t)DIM0;
            block[1]  = (hsize_t)DIM1;
            stride[0] = block[0];
            stride[1] = block[1];
            count[0]  = 1;
            count[1]  = 1;
            start[0]  = 0;
            start[1]  = 0;
            break;
    }
}

/*
 * Setup the coordinates for point selection.
 */
void
point_set(hsize_t start[], hsize_t count[], hsize_t stride[], hsize_t block[], size_t num_points,
          hsize_t coords[], int order)
{
    hsize_t i, j, k = 0, m, n, s1, s2;

    assert(RANK == 2);

    if (OUT_OF_ORDER == order)
        k = (num_points * RANK) - 1;
    else if (IN_ORDER == order)
        k = 0;

    s1 = start[0];
    s2 = start[1];

    for (i = 0; i < count[0]; i++)
        for (j = 0; j < count[1]; j++)
            for (m = 0; m < block[0]; m++)
                for (n = 0; n < block[1]; n++)
                    if (OUT_OF_ORDER == order) {
                        coords[k--] = s2 + (stride[1] * j) + n;
                        coords[k--] = s1 + (stride[0] * i) + m;
                    }
                    else if (IN_ORDER == order) {
                        coords[k++] = s1 + stride[0] * i + m;
                        coords[k++] = s2 + stride[1] * j + n;
                    }
}

/*
 * Fill the dataset with trivial data for testing.
 * Assume dimension rank is 2 and data is stored contiguous.
 */
static void
dataset_fill(hsize_t start[], hsize_t block[], DATATYPE *dataset)
{
    DATATYPE *dataptr = dataset;
    hsize_t   i, j;

    /* put some trivial data in the data_array */
    for (i = 0; i < block[0]; i++) {
        for (j = 0; j < block[1]; j++) {
            *dataptr = (DATATYPE)((i + start[0]) * 100 + (j + start[1] + 1));
            dataptr++;
        }
    }
}

int
dataset_vrfy(hsize_t start[], hsize_t count[], hsize_t stride[], hsize_t block[], DATATYPE *dataset,
             DATATYPE *original)
{
    hsize_t i, j;
    int     vrfyerrs;

    /* unused */
    H5FD_GDS_UNUSED(count);
    H5FD_GDS_UNUSED(stride);

    vrfyerrs = 0;
    for (i = 0; i < block[0]; i++) {
        for (j = 0; j < block[1]; j++) {
            if (*dataset != *original) {
                if (vrfyerrs++ < MAX_ERR_REPORT) {
                    printf("Dataset Verify failed at [%lu][%lu](row %lu, col %lu): expect %d, got %d\n",
                             (unsigned long)i, (unsigned long)j, (unsigned long)(i + start[0]),
                             (unsigned long)(j + start[1]), *(original), *(dataset));
                }
                dataset++;
                original++;
            }
        }
    }
    if (vrfyerrs > MAX_ERR_REPORT)
        printf("[more errors ...]\n");
    if (vrfyerrs)
        printf("%d errors found in dataset_vrfy\n", vrfyerrs);
    return (vrfyerrs);
}

/*
 * Example of using the parallel HDF5 library to create two datasets
 * in one HDF5 files with parallel MPIO access support.
 * The Datasets are of sizes (number-of-mpi-processes x DIM0) x DIM1.
 * Each process controls only a slab of size DIM0 x DIM1 within each
 * dataset.
 */

static void
extend_writeInd_cuda(void)
{
    hid_t       fid;                /* HDF5 file ID */
    hid_t       acc_tpl;            /* File access templates */
    hid_t       sid;                /* Dataspace ID */
    hid_t       file_dataspace;     /* File dataspace ID */
    hid_t       mem_dataspace;      /* memory dataspace ID */
    hid_t       dataset1, dataset2; /* Dataset ID */
    hsize_t     dims[RANK];                                      /* dataset dim sizes */
    hsize_t     max_dims[RANK] = {H5S_UNLIMITED, H5S_UNLIMITED}; /* dataset maximum dim sizes */
    DATATYPE *  data_array1    = NULL;                           /* data buffer */
    hsize_t     chunk_dims[RANK];                                /* chunk sizes */
    hid_t       dataset_pl;                                      /* dataset create prop. list */

    hsize_t   start[RANK];      /* for hyperslab setting */
    hsize_t   count[RANK];      /* for hyperslab setting */
    hsize_t   stride[RANK];     /* for hyperslab setting */
    hsize_t   block[RANK];      /* for hyperslab setting */
    DATATYPE *cuda_buff = NULL; /* data buffer */

    herr_t ret; /* Generic return value */

    printf("Extend independent write test on file %s\n", FILENAME);

    /* setup chunk-size. Make sure sizes are > 0 */
    chunk_dims[0] = CHUNK_DIM0;
    chunk_dims[1] = CHUNK_DIM1;

    /* allocate memory for data buffer */
    data_array1 = (DATATYPE *)malloc(DIM0 * DIM1 * sizeof(DATATYPE));
    VRFY((data_array1 != NULL), "data_array1 malloc succeeded");

    /* -------------------
     * START AN HDF5 FILE
     * -------------------*/
    /* setup file access template */
    acc_tpl = H5Pcreate(H5P_FILE_ACCESS);
    VRFY((acc_tpl >= 0), "H5Pcreate H5P_FILE_ACCESS");
    ret = H5Pset_fapl_mpio(acc_tpl, comm, info);
    VRFY((ret >= 0), "H5Pset_fapl_mpio");
    ret = H5Pset_all_coll_metadata_ops(acc_tpl, true);
    VRFY((ret >= 0), "H5Pset_all_coll_metadata_ops");
    ret = H5Pset_coll_metadata_write(acc_tpl, true);
    VRFY((ret >= 0), "H5Pset_coll_metadata_write");

    /* Reduce the number of metadata cache slots, so that there are cache
     * collisions during the raw data I/O on the chunked dataset.  This stresses
     * the metadata cache and tests for cache bugs. -QAK
     */
    {
        int    mdc_nelmts;
        size_t rdcc_nelmts;
        size_t rdcc_nbytes;
        double rdcc_w0;

        ret = H5Pget_cache(acc_tpl, &mdc_nelmts, &rdcc_nelmts, &rdcc_nbytes, &rdcc_w0);
        VRFY((ret >= 0), "H5Pget_cache succeeded");
        mdc_nelmts = 4;
        ret        = H5Pset_cache(acc_tpl, mdc_nelmts, rdcc_nelmts, rdcc_nbytes, rdcc_w0);
        VRFY((ret >= 0), "H5Pset_cache succeeded");
    }

    /* create the file collectively */
    fid = H5Fcreate(FILENAME, H5F_ACC_TRUNC, H5P_DEFAULT, acc_tpl);
    VRFY((fid >= 0), "H5Fcreate succeeded");

    /* Release file-access template */
    ret = H5Pclose(acc_tpl);
    VRFY((ret >= 0), "");

    /* --------------------------------------------------------------
     * Define the dimensions of the overall datasets and create them.
     * ------------------------------------------------------------- */

    /* set up dataset storage chunk sizes and creation property list */
    dataset_pl = H5Pcreate(H5P_DATASET_CREATE);
    VRFY((dataset_pl >= 0), "H5Pcreate succeeded");
    ret = H5Pset_chunk(dataset_pl, RANK, chunk_dims);
    VRFY((ret >= 0), "H5Pset_chunk succeeded");

    /* setup dimensionality object */
    /* start out with no rows, extend it later. */
    dims[0] = dims[1] = 0;
    sid               = H5Screate_simple(RANK, dims, max_dims);
    VRFY((sid >= 0), "H5Screate_simple succeeded");

    /* create an extendible dataset collectively */
    dataset1 = H5Dcreate2(fid, DATASETNAME1, H5T_NATIVE_INT, sid, H5P_DEFAULT, dataset_pl, H5P_DEFAULT);
    VRFY((dataset1 >= 0), "H5Dcreate2 succeeded");

    /* create another extendible dataset collectively */
    dataset2 = H5Dcreate2(fid, DATASETNAME2, H5T_NATIVE_INT, sid, H5P_DEFAULT, dataset_pl, H5P_DEFAULT);
    VRFY((dataset2 >= 0), "H5Dcreate2 succeeded");

    /* release resource */
    H5Sclose(sid);
    H5Pclose(dataset_pl);

    /* -------------------------
     * Test writing to dataset1
     * -------------------------*/
    /* set up dimensions of the slab this process accesses */
    slab_set(mpi_rank, mpi_size, start, count, stride, block, BYROW);

    /* put some trivial data in the data_array */
    dataset_fill(start, block, data_array1);

    /* create a memory dataspace independently */
    mem_dataspace = H5Screate_simple(RANK, block, NULL);
    VRFY((mem_dataspace >= 0), "");

    /* Extend its current dim sizes before writing */
    dims[0] = DIM0;
    dims[1] = DIM1;
    ret     = H5Dset_extent(dataset1, dims);
    VRFY((ret >= 0), "H5Dset_extent succeeded");

    /* create a file dataspace independently */
    file_dataspace = H5Dget_space(dataset1);
    VRFY((file_dataspace >= 0), "H5Dget_space succeeded");
    ret = H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, stride, count, block);
    VRFY((ret >= 0), "H5Sset_hyperslab succeeded");

    cudaMalloc((void **)&cuda_buff, DIM0 * DIM1 * sizeof(DATATYPE));
    cudaMemcpy(cuda_buff, data_array1, DIM0 * DIM1 * sizeof(DATATYPE), cudaMemcpyHostToDevice);

    /* write data independently */
    ret = H5Dwrite(dataset1, H5T_NATIVE_INT, mem_dataspace, file_dataspace, H5P_DEFAULT, cuda_buff);
    VRFY((ret >= 0), "H5Dwrite succeeded");

    /* release resource */
    H5Sclose(file_dataspace);
    H5Sclose(mem_dataspace);

    /* -------------------------
     * Test writing to dataset2
     * -------------------------*/
    /* set up dimensions of the slab this process accesses */
    slab_set(mpi_rank, mpi_size, start, count, stride, block, BYCOL);

    /* put some trivial data in the data_array */
    dataset_fill(start, block, data_array1);

    /* create a memory dataspace independently */
    mem_dataspace = H5Screate_simple(RANK, block, NULL);
    VRFY((mem_dataspace >= 0), "");

    /* Try write to dataset2 beyond its current dim sizes.  Should fail. */
    H5E_BEGIN_TRY
    {
        /* create a file dataspace independently */
        file_dataspace = H5Dget_space(dataset2);
        VRFY((file_dataspace >= 0), "H5Dget_space succeeded");
        ret = H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, stride, count, block);
        VRFY((ret >= 0), "H5Sset_hyperslab succeeded");
        cudaMemcpy(cuda_buff, data_array1, DIM0 * DIM1 * sizeof(DATATYPE), cudaMemcpyHostToDevice);

        /* write data independently.  Should fail. */
        ret = H5Dwrite(dataset2, H5T_NATIVE_INT, mem_dataspace, file_dataspace, H5P_DEFAULT, cuda_buff);
        VRFY((ret < 0), "H5Dwrite failed as expected");
    }
    H5E_END_TRY;

    H5Sclose(file_dataspace);

    /* Extend dataset2 and try again.  Should succeed. */
    dims[0] = DIM0;
    dims[1] = DIM1;
    ret     = H5Dset_extent(dataset2, dims);
    VRFY((ret >= 0), "H5Dset_extent succeeded");

    /* create a file dataspace independently */
    file_dataspace = H5Dget_space(dataset2);
    VRFY((file_dataspace >= 0), "H5Dget_space succeeded");
    ret = H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, stride, count, block);
    VRFY((ret >= 0), "H5Sset_hyperslab succeeded");

    /* write data independently */
    ret = H5Dwrite(dataset2, H5T_NATIVE_INT, mem_dataspace, file_dataspace, H5P_DEFAULT, cuda_buff);
    VRFY((ret >= 0), "H5Dwrite succeeded");

    /* release resource */
    ret = H5Sclose(file_dataspace);
    VRFY((ret >= 0), "H5Sclose succeeded");
    ret = H5Sclose(mem_dataspace);
    VRFY((ret >= 0), "H5Sclose succeeded");

    /* close dataset collectively */
    ret = H5Dclose(dataset1);
    VRFY((ret >= 0), "H5Dclose1 succeeded");
    ret = H5Dclose(dataset2);
    VRFY((ret >= 0), "H5Dclose2 succeeded");

    /* close the file collectively */
    H5Fclose(fid);

    /* release data buffers */
    if (data_array1)
        free(data_array1);
    cudaFree(&cuda_buff);
}

static void
dataset_writeAll_cuda(void)
{
    hid_t       fid;                                    /* HDF5 file ID */
    hid_t       acc_tpl;                                /* File access templates */
    hid_t       xfer_plist;                             /* Dataset transfer properties list */
    hid_t       sid;                                    /* Dataspace ID */
    hid_t       file_dataspace;                         /* File dataspace ID */
    hid_t       mem_dataspace;                          /* memory dataspace ID */
    hid_t       dataset1, dataset2, dataset3, dataset4; /* Dataset ID */
    hid_t       dataset5, dataset6, dataset7;           /* Dataset ID */
    hid_t       datatype;                               /* Datatype ID */
    hsize_t     dims[RANK];                             /* dataset dim sizes */
    DATATYPE *  data_array1 = NULL;                     /* data buffer */

    DATATYPE *cuda_buff = NULL; /* data buffer */

    hsize_t start[RANK];               /* for hyperslab setting */
    hsize_t count[RANK], stride[RANK]; /* for hyperslab setting */
    hsize_t block[RANK];               /* for hyperslab setting */

    size_t   num_points;    /* for point selection */
    hsize_t *coords = NULL; /* for point selection */
    hsize_t  current_dims;  /* for point selection */

    herr_t ret; /* Generic return value */

    printf("Collective write test on file %s\n", FILENAME);

    /* set up the coords array selection */
    num_points = DIM1;
    coords     = (hsize_t *)malloc(DIM1 * RANK * sizeof(hsize_t));
    VRFY((coords != NULL), "coords malloc succeeded");

    /* allocate memory for data buffer */
    data_array1 = (DATATYPE *)malloc(DIM0 * DIM1 * sizeof(DATATYPE));
    VRFY((data_array1 != NULL), "data_array1 malloc succeeded");

    /* -------------------
     * START AN HDF5 FILE
     * -------------------*/
    /* setup file access template */
    acc_tpl = H5Pcreate(H5P_FILE_ACCESS);
    VRFY((acc_tpl >= 0), "H5Pcreate H5P_FILE_ACCESS");
    ret = H5Pset_fapl_mpio(acc_tpl, comm, info);
    VRFY((ret >= 0), "H5Pset_fapl_mpio");
    ret = H5Pset_all_coll_metadata_ops(acc_tpl, true);
    VRFY((ret >= 0), "H5Pset_all_coll_metadata_ops");
    ret = H5Pset_coll_metadata_write(acc_tpl, true);
    VRFY((ret >= 0), "H5Pset_coll_metadata_write");

    /* create the file collectively */
    fid = H5Fcreate(FILENAME, H5F_ACC_TRUNC, H5P_DEFAULT, acc_tpl);
    VRFY((fid >= 0), "H5Fcreate succeeded");

    /* Release file-access template */
    ret = H5Pclose(acc_tpl);
    VRFY((ret >= 0), "");

    /* --------------------------
     * Define the dimensions of the overall datasets
     * and create the dataset
     * ------------------------- */
    /* setup 2-D dimensionality object */
    dims[0] = DIM0;
    dims[1] = DIM1;
    sid     = H5Screate_simple(RANK, dims, NULL);
    VRFY((sid >= 0), "H5Screate_simple succeeded");

    /* create a dataset collectively */
    dataset1 = H5Dcreate2(fid, DATASETNAME1, H5T_NATIVE_INT, sid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    VRFY((dataset1 >= 0), "H5Dcreate2 succeeded");

    /* create another dataset collectively */
    datatype = H5Tcopy(H5T_NATIVE_INT);
    ret      = H5Tset_order(datatype, H5T_ORDER_LE);
    VRFY((ret >= 0), "H5Tset_order succeeded");

    dataset2 = H5Dcreate2(fid, DATASETNAME2, datatype, sid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    VRFY((dataset2 >= 0), "H5Dcreate2 2 succeeded");

    /* create a third dataset collectively */
    dataset3 = H5Dcreate2(fid, DATASETNAME3, H5T_NATIVE_INT, sid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    VRFY((dataset3 >= 0), "H5Dcreate2 succeeded");

    dataset5 = H5Dcreate2(fid, DATASETNAME7, H5T_NATIVE_INT, sid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    VRFY((dataset5 >= 0), "H5Dcreate2 succeeded");
    dataset6 = H5Dcreate2(fid, DATASETNAME8, H5T_NATIVE_INT, sid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    VRFY((dataset6 >= 0), "H5Dcreate2 succeeded");
    dataset7 = H5Dcreate2(fid, DATASETNAME9, H5T_NATIVE_INT, sid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    VRFY((dataset7 >= 0), "H5Dcreate2 succeeded");

    /* release 2-D space ID created */
    H5Sclose(sid);

    /* setup scalar dimensionality object */
    sid = H5Screate(H5S_SCALAR);
    VRFY((sid >= 0), "H5Screate succeeded");

    /* create a fourth dataset collectively */
    dataset4 = H5Dcreate2(fid, DATASETNAME4, H5T_NATIVE_INT, sid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    VRFY((dataset4 >= 0), "H5Dcreate2 succeeded");

    /* release scalar space ID created */
    H5Sclose(sid);

    /*
     * Set up dimensions of the slab this process accesses.
     */

    /* Dataset1: each process takes a block of rows. */
    slab_set(mpi_rank, mpi_size, start, count, stride, block, BYROW);

    /* create a file dataspace independently */
    file_dataspace = H5Dget_space(dataset1);
    VRFY((file_dataspace >= 0), "H5Dget_space succeeded");
    ret = H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, stride, count, block);
    VRFY((ret >= 0), "H5Sset_hyperslab succeeded");

    /* create a memory dataspace independently */
    mem_dataspace = H5Screate_simple(RANK, block, NULL);
    VRFY((mem_dataspace >= 0), "");

    /* fill the local slab with some trivial data */
    dataset_fill(start, block, data_array1);

    /* set up the collective transfer properties list */
    xfer_plist = H5Pcreate(H5P_DATASET_XFER);
    VRFY((xfer_plist >= 0), "H5Pcreate xfer succeeded");
    ret = H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
    VRFY((ret >= 0), "H5Pset_dxpl_mpio succeeded");
    if (dxfer_coll_type == DXFER_INDEPENDENT_IO) {
        ret = H5Pset_dxpl_mpio_collective_opt(xfer_plist, H5FD_MPIO_INDIVIDUAL_IO);
        VRFY((ret >= 0), "set independent IO collectively succeeded");
    }

    cudaMalloc((void **)&cuda_buff, DIM0 * DIM1 * sizeof(DATATYPE));
    cudaMemcpy(cuda_buff, data_array1, DIM0 * DIM1 * sizeof(DATATYPE), cudaMemcpyHostToDevice);

    /* write data collectively */
    ret = H5Dwrite(dataset1, H5T_NATIVE_INT, mem_dataspace, file_dataspace, xfer_plist, cuda_buff);
    VRFY((ret >= 0), "H5Dwrite dataset1 succeeded");

    /* setup dimensions again to writeAll with zero rows for process 0 */
    printf("writeAll by some with zero row\n");
    slab_set(mpi_rank, mpi_size, start, count, stride, block, ZROW);
    ret = H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, stride, count, block);
    VRFY((ret >= 0), "H5Sset_hyperslab succeeded");
    /* need to make mem_dataspace to match for process 0 */
    if (MAINPROCESS) {
        ret = H5Sselect_hyperslab(mem_dataspace, H5S_SELECT_SET, start, stride, count, block);
        VRFY((ret >= 0), "H5Sset_hyperslab mem_dataspace succeeded");
    }
    ret = H5Dwrite(dataset1, H5T_NATIVE_INT, mem_dataspace, file_dataspace, xfer_plist, cuda_buff);
    VRFY((ret >= 0), "H5Dwrite dataset1 by ZROW succeeded");

    /* release all temporary handles. */
    /* Could have used them for dataset2 but it is cleaner */
    /* to create them again.*/
    H5Sclose(file_dataspace);
    H5Sclose(mem_dataspace);
    H5Pclose(xfer_plist);

    /* Dataset2: each process takes a block of columns. */
    slab_set(mpi_rank, mpi_size, start, count, stride, block, BYCOL);

    /* put some trivial data in the data_array */
    dataset_fill(start, block, data_array1);

    /* create a file dataspace independently */
    file_dataspace = H5Dget_space(dataset1);
    VRFY((file_dataspace >= 0), "H5Dget_space succeeded");
    ret = H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, stride, count, block);
    VRFY((ret >= 0), "H5Sset_hyperslab succeeded");

    /* create a memory dataspace independently */
    mem_dataspace = H5Screate_simple(RANK, block, NULL);
    VRFY((mem_dataspace >= 0), "");

    /* fill the local slab with some trivial data */
    dataset_fill(start, block, data_array1);

    /* set up the collective transfer properties list */
    xfer_plist = H5Pcreate(H5P_DATASET_XFER);
    VRFY((xfer_plist >= 0), "");
    ret = H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
    VRFY((ret >= 0), "H5Pcreate xfer succeeded");
    if (dxfer_coll_type == DXFER_INDEPENDENT_IO) {
        ret = H5Pset_dxpl_mpio_collective_opt(xfer_plist, H5FD_MPIO_INDIVIDUAL_IO);
        VRFY((ret >= 0), "set independent IO collectively succeeded");
    }

    /* Raafat: Collective write: prob 1
        when using cuda_buff instead of data_array1: ret < 0
    */
    cudaMemcpy(cuda_buff, data_array1, DIM0 * DIM1 * sizeof(DATATYPE), cudaMemcpyHostToDevice);
    /* write data independently */
    ret = H5Dwrite(dataset2, H5T_NATIVE_INT, mem_dataspace, file_dataspace, xfer_plist, cuda_buff);
    VRFY((ret >= 0), "H5Dwrite dataset2 succeeded");

    /* setup dimensions again to writeAll with zero columns for process 0 */
    printf("writeAll by some with zero col\n");
    slab_set(mpi_rank, mpi_size, start, count, stride, block, ZCOL);
    ret = H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, stride, count, block);
    VRFY((ret >= 0), "H5Sset_hyperslab succeeded");
    /* need to make mem_dataspace to match for process 0 */
    if (MAINPROCESS) {
        ret = H5Sselect_hyperslab(mem_dataspace, H5S_SELECT_SET, start, stride, count, block);
        VRFY((ret >= 0), "H5Sset_hyperslab mem_dataspace succeeded");
    }
    /* Raafat: Collective write: prob 2
        when using cuda_buff instead of data_array1: ret < 0
    */
    ret = H5Dwrite(dataset1, H5T_NATIVE_INT, mem_dataspace, file_dataspace, xfer_plist, cuda_buff);
    VRFY((ret >= 0), "H5Dwrite dataset1 by ZCOL succeeded");

    /* release all temporary handles. */
    /* Could have used them for dataset3 but it is cleaner */
    /* to create them again.*/
    H5Sclose(file_dataspace);
    H5Sclose(mem_dataspace);
    H5Pclose(xfer_plist);

    /* Dataset3: each process takes a block of rows, except process zero uses "none" selection. */
    slab_set(mpi_rank, mpi_size, start, count, stride, block, BYROW);

    /* create a file dataspace independently */
    file_dataspace = H5Dget_space(dataset3);
    VRFY((file_dataspace >= 0), "H5Dget_space succeeded");
    if (MAINPROCESS) {
        ret = H5Sselect_none(file_dataspace);
        VRFY((ret >= 0), "H5Sselect_none file_dataspace succeeded");
    } /* end if */
    else {
        ret = H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, stride, count, block);
        VRFY((ret >= 0), "H5Sselect_hyperslab succeeded");
    } /* end else */

    /* create a memory dataspace independently */
    mem_dataspace = H5Screate_simple(RANK, block, NULL);
    VRFY((mem_dataspace >= 0), "");
    if (MAINPROCESS) {
        ret = H5Sselect_none(mem_dataspace);
        VRFY((ret >= 0), "H5Sselect_none mem_dataspace succeeded");
    } /* end if */

    /* fill the local slab with some trivial data */
    dataset_fill(start, block, data_array1);

    /* set up the collective transfer properties list */
    xfer_plist = H5Pcreate(H5P_DATASET_XFER);
    VRFY((xfer_plist >= 0), "");
    ret = H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
    VRFY((ret >= 0), "H5Pcreate xfer succeeded");
    if (dxfer_coll_type == DXFER_INDEPENDENT_IO) {
        ret = H5Pset_dxpl_mpio_collective_opt(xfer_plist, H5FD_MPIO_INDIVIDUAL_IO);
        VRFY((ret >= 0), "set independent IO collectively succeeded");
    }

    cudaMemcpy(cuda_buff, data_array1, DIM0 * DIM1 * sizeof(DATATYPE), cudaMemcpyHostToDevice);

    /* write data collectively */
    ret = H5Dwrite(dataset3, H5T_NATIVE_INT, mem_dataspace, file_dataspace, xfer_plist, cuda_buff);
    VRFY((ret >= 0), "H5Dwrite dataset3 succeeded");

    /* write data collectively (with datatype conversion) */
    /* Raafat: Collective write: prob 3
        when using cuda_buff instead of data_array1: Segmentation fault
        change H5T_NATIVE_UCHAR to H5T_NATIVE_INT make it work
    */
    ret = H5Dwrite(dataset3, H5T_NATIVE_UCHAR, mem_dataspace, file_dataspace, xfer_plist, cuda_buff);
    VRFY((ret >= 0), "H5Dwrite dataset3 succeeded");

    /* release all temporary handles. */
    /* Could have used them for dataset4 but it is cleaner */
    /* to create them again.*/
    H5Sclose(file_dataspace);
    H5Sclose(mem_dataspace);
    H5Pclose(xfer_plist);

    /* Dataset4: each process writes no data, except process zero uses "all" selection. */
    /* Additionally, these are in a scalar dataspace */

    /* create a file dataspace independently */
    file_dataspace = H5Dget_space(dataset4);
    VRFY((file_dataspace >= 0), "H5Dget_space succeeded");
    if (MAINPROCESS) {
        ret = H5Sselect_none(file_dataspace);
        VRFY((ret >= 0), "H5Sselect_all file_dataspace succeeded");
    } /* end if */
    else {
        ret = H5Sselect_all(file_dataspace);
        VRFY((ret >= 0), "H5Sselect_none succeeded");
    } /* end else */

    /* create a memory dataspace independently */
    mem_dataspace = H5Screate(H5S_SCALAR);
    VRFY((mem_dataspace >= 0), "");
    if (MAINPROCESS) {
        ret = H5Sselect_none(mem_dataspace);
        VRFY((ret >= 0), "H5Sselect_all mem_dataspace succeeded");
    } /* end if */
    else {
        ret = H5Sselect_all(mem_dataspace);
        VRFY((ret >= 0), "H5Sselect_none succeeded");
    } /* end else */

    /* fill the local slab with some trivial data */
    dataset_fill(start, block, data_array1);

    /* set up the collective transfer properties list */
    xfer_plist = H5Pcreate(H5P_DATASET_XFER);
    VRFY((xfer_plist >= 0), "");
    ret = H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
    VRFY((ret >= 0), "H5Pcreate xfer succeeded");
    if (dxfer_coll_type == DXFER_INDEPENDENT_IO) {
        ret = H5Pset_dxpl_mpio_collective_opt(xfer_plist, H5FD_MPIO_INDIVIDUAL_IO);
        VRFY((ret >= 0), "set independent IO collectively succeeded");
    }

    cudaMemcpy(cuda_buff, data_array1, DIM0 * DIM1 * sizeof(DATATYPE), cudaMemcpyHostToDevice);

    /* write data collectively */
    ret = H5Dwrite(dataset4, H5T_NATIVE_INT, mem_dataspace, file_dataspace, xfer_plist, cuda_buff);
    VRFY((ret >= 0), "H5Dwrite dataset4 succeeded");

    /* Raafat: Collective write: prob 4
        when using cuda_buff instead of data_array1: Segmentation fault
        change H5T_NATIVE_UCHAR to H5T_NATIVE_INT make it work
    */
    /* write data collectively (with datatype conversion) */
    ret = H5Dwrite(dataset4, H5T_NATIVE_UCHAR, mem_dataspace, file_dataspace, xfer_plist, cuda_buff);
    VRFY((ret >= 0), "H5Dwrite dataset4 succeeded");

    /* release all temporary handles. */
    H5Sclose(file_dataspace);
    H5Sclose(mem_dataspace);
    H5Pclose(xfer_plist);

    if (data_array1)
        free(data_array1);
    data_array1 = (DATATYPE *)malloc(DIM0 * DIM1 * sizeof(DATATYPE));
    VRFY((data_array1 != NULL), "data_array1 malloc succeeded");

    cudaFree(&cuda_buff);
    cudaMalloc((void **)&cuda_buff, DIM0 * DIM1 * sizeof(DATATYPE));

    block[0]  = 1;
    block[1]  = DIM1;
    stride[0] = 1;
    stride[1] = DIM1;
    count[0]  = 1;
    count[1]  = 1;
    start[0]  = DIM0 / mpi_size * mpi_rank;
    start[1]  = 0;

    dataset_fill(start, block, data_array1);

    /* Dataset5: point selection in File - Hyperslab selection in Memory*/
    /* create a file dataspace independently */
    point_set(start, count, stride, block, num_points, coords, OUT_OF_ORDER);
    file_dataspace = H5Dget_space(dataset5);
    VRFY((file_dataspace >= 0), "H5Dget_space succeeded");
    ret = H5Sselect_elements(file_dataspace, H5S_SELECT_SET, num_points, coords);
    VRFY((ret >= 0), "H5Sselect_elements succeeded");

    start[0]      = 0;
    start[1]      = 0;
    mem_dataspace = H5Dget_space(dataset5);
    VRFY((mem_dataspace >= 0), "H5Dget_space succeeded");
    ret = H5Sselect_hyperslab(mem_dataspace, H5S_SELECT_SET, start, stride, count, block);
    VRFY((ret >= 0), "H5Sset_hyperslab succeeded");

    /* set up the collective transfer properties list */
    xfer_plist = H5Pcreate(H5P_DATASET_XFER);
    VRFY((xfer_plist >= 0), "");
    ret = H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
    VRFY((ret >= 0), "H5Pcreate xfer succeeded");
    if (dxfer_coll_type == DXFER_INDEPENDENT_IO) {
        ret = H5Pset_dxpl_mpio_collective_opt(xfer_plist, H5FD_MPIO_INDIVIDUAL_IO);
        VRFY((ret >= 0), "set independent IO collectively succeeded");
    }

    cudaMemcpy(cuda_buff, data_array1, DIM0 * DIM1 * sizeof(DATATYPE), cudaMemcpyHostToDevice);

    /* Raafat: Collective write: prob 5
        when using cuda_buff instead of data_array1: ret < 0
    */
    /* write data collectively */
    ret = H5Dwrite(dataset5, H5T_NATIVE_INT, mem_dataspace, file_dataspace, xfer_plist, cuda_buff);
    VRFY((ret >= 0), "H5Dwrite dataset5 succeeded");

    /* release all temporary handles. */
    H5Sclose(file_dataspace);
    H5Sclose(mem_dataspace);
    H5Pclose(xfer_plist);

    /* Dataset6: point selection in File - Point selection in Memory*/
    /* create a file dataspace independently */
    start[0] = DIM0 / mpi_size * mpi_rank;
    start[1] = 0;
    point_set(start, count, stride, block, num_points, coords, OUT_OF_ORDER);
    file_dataspace = H5Dget_space(dataset6);
    VRFY((file_dataspace >= 0), "H5Dget_space succeeded");
    ret = H5Sselect_elements(file_dataspace, H5S_SELECT_SET, num_points, coords);
    VRFY((ret >= 0), "H5Sselect_elements succeeded");

    start[0] = 0;
    start[1] = 0;
    point_set(start, count, stride, block, num_points, coords, IN_ORDER);
    mem_dataspace = H5Dget_space(dataset6);
    VRFY((mem_dataspace >= 0), "H5Dget_space succeeded");
    ret = H5Sselect_elements(mem_dataspace, H5S_SELECT_SET, num_points, coords);
    VRFY((ret >= 0), "H5Sselect_elements succeeded");

    /* set up the collective transfer properties list */
    xfer_plist = H5Pcreate(H5P_DATASET_XFER);
    VRFY((xfer_plist >= 0), "");
    ret = H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
    VRFY((ret >= 0), "H5Pcreate xfer succeeded");
    if (dxfer_coll_type == DXFER_INDEPENDENT_IO) {
        ret = H5Pset_dxpl_mpio_collective_opt(xfer_plist, H5FD_MPIO_INDIVIDUAL_IO);
        VRFY((ret >= 0), "set independent IO collectively succeeded");
    }

    /* Raafat: Collective write: prob 6
        when using cuda_buff instead of data_array1: ret < 0
    */
    /* write data collectively */
    ret = H5Dwrite(dataset6, H5T_NATIVE_INT, mem_dataspace, file_dataspace, xfer_plist, cuda_buff);
    VRFY((ret >= 0), "H5Dwrite dataset6 succeeded");

    /* release all temporary handles. */
    H5Sclose(file_dataspace);
    H5Sclose(mem_dataspace);
    H5Pclose(xfer_plist);

    /* Dataset7: point selection in File - All selection in Memory*/
    /* create a file dataspace independently */
    start[0] = DIM0 / mpi_size * mpi_rank;
    start[1] = 0;
    point_set(start, count, stride, block, num_points, coords, IN_ORDER);
    file_dataspace = H5Dget_space(dataset7);
    VRFY((file_dataspace >= 0), "H5Dget_space succeeded");
    ret = H5Sselect_elements(file_dataspace, H5S_SELECT_SET, num_points, coords);
    VRFY((ret >= 0), "H5Sselect_elements succeeded");

    current_dims  = num_points;
    mem_dataspace = H5Screate_simple(1, &current_dims, NULL);
    VRFY((mem_dataspace >= 0), "mem_dataspace create succeeded");

    ret = H5Sselect_all(mem_dataspace);
    VRFY((ret >= 0), "H5Sselect_all succeeded");

    /* set up the collective transfer properties list */
    xfer_plist = H5Pcreate(H5P_DATASET_XFER);
    VRFY((xfer_plist >= 0), "");
    ret = H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
    VRFY((ret >= 0), "H5Pcreate xfer succeeded");
    if (dxfer_coll_type == DXFER_INDEPENDENT_IO) {
        ret = H5Pset_dxpl_mpio_collective_opt(xfer_plist, H5FD_MPIO_INDIVIDUAL_IO);
        VRFY((ret >= 0), "set independent IO collectively succeeded");
    }

    /* write data collectively */
    ret = H5Dwrite(dataset7, H5T_NATIVE_INT, mem_dataspace, file_dataspace, xfer_plist, cuda_buff);
    VRFY((ret >= 0), "H5Dwrite dataset7 succeeded");

    /* release all temporary handles. */
    H5Sclose(file_dataspace);
    H5Sclose(mem_dataspace);
    H5Pclose(xfer_plist);

    /*
     * All writes completed.  Close datasets collectively
     */
    ret = H5Dclose(dataset1);
    VRFY((ret >= 0), "H5Dclose1 succeeded");
    ret = H5Dclose(dataset2);
    VRFY((ret >= 0), "H5Dclose2 succeeded");
    ret = H5Dclose(dataset3);
    VRFY((ret >= 0), "H5Dclose3 succeeded");
    ret = H5Dclose(dataset4);
    VRFY((ret >= 0), "H5Dclose4 succeeded");
    ret = H5Dclose(dataset5);
    VRFY((ret >= 0), "H5Dclose5 succeeded");
    ret = H5Dclose(dataset6);
    VRFY((ret >= 0), "H5Dclose6 succeeded");
    ret = H5Dclose(dataset7);
    VRFY((ret >= 0), "H5Dclose7 succeeded");

    /* close the file collectively */
    H5Fclose(fid);

    /* release data buffers */
    if (coords)
        free(coords);
    if (data_array1)
        free(data_array1);
    cudaFree(&cuda_buff);
}

static void
dataset_readAll_cuda(void)
{
    hid_t       fid;                                              /* HDF5 file ID */
    hid_t       acc_tpl;                                          /* File access templates */
    hid_t       xfer_plist;                                       /* Dataset transfer properties list */
    hid_t       file_dataspace;                                   /* File dataspace ID */
    hid_t       mem_dataspace;                                    /* memory dataspace ID */
    hid_t       dataset1, dataset2, dataset5, dataset6, dataset7; /* Dataset ID */
    DATATYPE *  data_array1  = NULL;                              /* data buffer */
    DATATYPE *  data_origin1 = NULL;                              /* expected data buffer */
    DATATYPE *  cuda_buff = NULL; /* data buffer */

    hsize_t start[RANK];               /* for hyperslab setting */
    hsize_t count[RANK], stride[RANK]; /* for hyperslab setting */
    hsize_t block[RANK];               /* for hyperslab setting */

    size_t   num_points;    /* for point selection */
    hsize_t *coords = NULL; /* for point selection */
    int      i, j, k;

    herr_t ret; /* Generic return value */

    printf("Collective read test on file %s\n", FILENAME);

    /* set up the coords array selection */
    num_points = DIM1;
    coords     = (hsize_t *)malloc(DIM0 * DIM1 * RANK * sizeof(hsize_t));
    VRFY((coords != NULL), "coords malloc succeeded");

    /* allocate memory for data buffer */
    data_array1 = (DATATYPE *)malloc(DIM0 * DIM1 * sizeof(DATATYPE));
    VRFY((data_array1 != NULL), "data_array1 malloc succeeded");
    data_origin1 = (DATATYPE *)malloc(DIM0 * DIM1 * sizeof(DATATYPE));
    VRFY((data_origin1 != NULL), "data_origin1 malloc succeeded");

    /* -------------------
     * OPEN AN HDF5 FILE
     * -------------------*/
    /* setup file access template */
    acc_tpl = H5Pcreate(H5P_FILE_ACCESS);
    VRFY((acc_tpl >= 0), "H5Pcreate H5P_FILE_ACCESS");
    ret = H5Pset_fapl_mpio(acc_tpl, comm, info);
    VRFY((ret >= 0), "H5Pset_fapl_mpio");
    ret = H5Pset_all_coll_metadata_ops(acc_tpl, true);
    VRFY((ret >= 0), "H5Pset_all_coll_metadata_ops");
    ret = H5Pset_coll_metadata_write(acc_tpl, true);
    VRFY((ret >= 0), "H5Pset_coll_metadata_write");

    /* open the file collectively */
    fid = H5Fopen(FILENAME, H5F_ACC_RDONLY, acc_tpl);
    VRFY((fid >= 0), "H5Fopen succeeded");

    /* Release file-access template */
    ret = H5Pclose(acc_tpl);
    VRFY((ret >= 0), "");

    /* --------------------------
     * Open the datasets in it
     * ------------------------- */
    /* open the dataset1 collectively */
    dataset1 = H5Dopen2(fid, DATASETNAME1, H5P_DEFAULT);
    VRFY((dataset1 >= 0), "H5Dopen2 succeeded");

    /* open another dataset collectively */
    dataset2 = H5Dopen2(fid, DATASETNAME2, H5P_DEFAULT);
    VRFY((dataset2 >= 0), "H5Dopen2 2 succeeded");

    /* open another dataset collectively */
    dataset5 = H5Dopen2(fid, DATASETNAME7, H5P_DEFAULT);
    VRFY((dataset5 >= 0), "H5Dopen2 5 succeeded");
    dataset6 = H5Dopen2(fid, DATASETNAME8, H5P_DEFAULT);
    VRFY((dataset6 >= 0), "H5Dopen2 6 succeeded");
    dataset7 = H5Dopen2(fid, DATASETNAME9, H5P_DEFAULT);
    VRFY((dataset7 >= 0), "H5Dopen2 7 succeeded");

    /*
     * Set up dimensions of the slab this process accesses.
     */

    /* Dataset1: each process takes a block of columns. */
    slab_set(mpi_rank, mpi_size, start, count, stride, block, BYCOL);

    /* create a file dataspace independently */
    file_dataspace = H5Dget_space(dataset1);
    VRFY((file_dataspace >= 0), "H5Dget_space succeeded");
    ret = H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, stride, count, block);
    VRFY((ret >= 0), "H5Sset_hyperslab succeeded");

    /* create a memory dataspace independently */
    mem_dataspace = H5Screate_simple(RANK, block, NULL);
    VRFY((mem_dataspace >= 0), "");

    /* fill dataset with test data */
    dataset_fill(start, block, data_origin1);

    /* set up the collective transfer properties list */
    xfer_plist = H5Pcreate(H5P_DATASET_XFER);
    VRFY((xfer_plist >= 0), "");
    ret = H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
    VRFY((ret >= 0), "H5Pcreate xfer succeeded");
    if (dxfer_coll_type == DXFER_INDEPENDENT_IO) {
        ret = H5Pset_dxpl_mpio_collective_opt(xfer_plist, H5FD_MPIO_INDIVIDUAL_IO);
        VRFY((ret >= 0), "set independent IO collectively succeeded");
    }

    cudaMalloc((void **)&cuda_buff, DIM0 * DIM1 * sizeof(DATATYPE));

    /* Raafat: Collective read prob 1
        read data collectively
    */
    ret = H5Dread(dataset1, H5T_NATIVE_INT, mem_dataspace, file_dataspace, xfer_plist, cuda_buff);
    VRFY((ret >= 0), "H5Dread dataset1 succeeded");

    cudaMemcpy(data_array1, cuda_buff, DIM0 * DIM1 * sizeof(DATATYPE), cudaMemcpyDeviceToHost);
    /* verify the read data with original expected data */
    ret = dataset_vrfy(start, count, stride, block, data_array1, data_origin1);
    if (ret)
        nerrors++;

    /* setup dimensions again to readAll with zero columns for process 0 */
    printf("readAll by some with zero col\n");
    slab_set(mpi_rank, mpi_size, start, count, stride, block, ZCOL);
    ret = H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, stride, count, block);
    VRFY((ret >= 0), "H5Sset_hyperslab succeeded");
    /* need to make mem_dataspace to match for process 0 */
    if (MAINPROCESS) {
        ret = H5Sselect_hyperslab(mem_dataspace, H5S_SELECT_SET, start, stride, count, block);
        VRFY((ret >= 0), "H5Sset_hyperslab mem_dataspace succeeded");
    }

    /* Raafat: Collective read prob 2
    */
    ret = H5Dread(dataset1, H5T_NATIVE_INT, mem_dataspace, file_dataspace, xfer_plist, cuda_buff);
    VRFY((ret >= 0), "H5Dread dataset1 by ZCOL succeeded");

    cudaMemcpy(data_array1, cuda_buff, DIM0 * DIM1 * sizeof(DATATYPE), cudaMemcpyDeviceToHost);
    /* verify the read data with original expected data */
    ret = dataset_vrfy(start, count, stride, block, data_array1, data_origin1);
    if (ret)
        nerrors++;

    /* release all temporary handles. */
    /* Could have used them for dataset2 but it is cleaner */
    /* to create them again.*/
    H5Sclose(file_dataspace);
    H5Sclose(mem_dataspace);
    H5Pclose(xfer_plist);

    /* Dataset2: each process takes a block of rows. */
    slab_set(mpi_rank, mpi_size, start, count, stride, block, BYROW);

    /* create a file dataspace independently */
    file_dataspace = H5Dget_space(dataset1);
    VRFY((file_dataspace >= 0), "H5Dget_space succeeded");
    ret = H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, stride, count, block);
    VRFY((ret >= 0), "H5Sset_hyperslab succeeded");

    /* create a memory dataspace independently */
    mem_dataspace = H5Screate_simple(RANK, block, NULL);
    VRFY((mem_dataspace >= 0), "");

    /* fill dataset with test data */
    dataset_fill(start, block, data_origin1);

    /* set up the collective transfer properties list */
    xfer_plist = H5Pcreate(H5P_DATASET_XFER);
    VRFY((xfer_plist >= 0), "");
    ret = H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
    VRFY((ret >= 0), "H5Pcreate xfer succeeded");
    if (dxfer_coll_type == DXFER_INDEPENDENT_IO) {
        ret = H5Pset_dxpl_mpio_collective_opt(xfer_plist, H5FD_MPIO_INDIVIDUAL_IO);
        VRFY((ret >= 0), "set independent IO collectively succeeded");
    }

    /* read data collectively */
    ret = H5Dread(dataset2, H5T_NATIVE_INT, mem_dataspace, file_dataspace, xfer_plist, cuda_buff);
    VRFY((ret >= 0), "H5Dread dataset2 succeeded");

    cudaMemcpy(data_array1, cuda_buff, DIM0 * DIM1 * sizeof(DATATYPE), cudaMemcpyDeviceToHost);
    /* verify the read data with original expected data */
    ret = dataset_vrfy(start, count, stride, block, data_array1, data_origin1);
    if (ret)
        nerrors++;

    /* setup dimensions again to readAll with zero rows for process 0 */
    printf("readAll by some with zero row\n");
    slab_set(mpi_rank, mpi_size, start, count, stride, block, ZROW);
    ret = H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, stride, count, block);
    VRFY((ret >= 0), "H5Sset_hyperslab succeeded");
    /* need to make mem_dataspace to match for process 0 */
    if (MAINPROCESS) {
        ret = H5Sselect_hyperslab(mem_dataspace, H5S_SELECT_SET, start, stride, count, block);
        VRFY((ret >= 0), "H5Sset_hyperslab mem_dataspace succeeded");
    }
    ret = H5Dread(dataset1, H5T_NATIVE_INT, mem_dataspace, file_dataspace, xfer_plist, cuda_buff);
    VRFY((ret >= 0), "H5Dread dataset1 by ZROW succeeded");

    cudaMemcpy(data_array1, cuda_buff, DIM0 * DIM1 * sizeof(DATATYPE), cudaMemcpyDeviceToHost);
    /* verify the read data with original expected data */
    ret = dataset_vrfy(start, count, stride, block, data_array1, data_origin1);
    if (ret)
        nerrors++;

    /* release all temporary handles. */
    H5Sclose(file_dataspace);
    H5Sclose(mem_dataspace);
    H5Pclose(xfer_plist);

    if (data_array1)
        free(data_array1);
    if (data_origin1)
        free(data_origin1);
    data_array1 = (DATATYPE *)malloc(DIM0 * DIM1 * sizeof(DATATYPE));
    VRFY((data_array1 != NULL), "data_array1 malloc succeeded");
    data_origin1 = (DATATYPE *)malloc(DIM0 * DIM1 * sizeof(DATATYPE));
    VRFY((data_origin1 != NULL), "data_origin1 malloc succeeded");

    if (NULL != cuda_buff)
        cudaFree(&cuda_buff);
    cudaMalloc((void **)&cuda_buff, DIM0 * DIM1 * sizeof(DATATYPE));

    block[0]  = 1;
    block[1]  = DIM1;
    stride[0] = 1;
    stride[1] = DIM1;
    count[0]  = 1;
    count[1]  = 1;
    start[0]  = DIM0 / mpi_size * mpi_rank;
    start[1]  = 0;

    dataset_fill(start, block, data_origin1);

    /* Dataset5: point selection in memory - Hyperslab selection in file*/
    /* create a file dataspace independently */
    file_dataspace = H5Dget_space(dataset5);
    VRFY((file_dataspace >= 0), "H5Dget_space succeeded");
    ret = H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, stride, count, block);
    VRFY((ret >= 0), "H5Sset_hyperslab succeeded");

    start[0] = 0;
    start[1] = 0;
    point_set(start, count, stride, block, num_points, coords, OUT_OF_ORDER);
    mem_dataspace = H5Dget_space(dataset5);
    VRFY((mem_dataspace >= 0), "H5Dget_space succeeded");
    ret = H5Sselect_elements(mem_dataspace, H5S_SELECT_SET, num_points, coords);
    VRFY((ret >= 0), "H5Sselect_elements succeeded");

    /* set up the collective transfer properties list */
    xfer_plist = H5Pcreate(H5P_DATASET_XFER);
    VRFY((xfer_plist >= 0), "");
    ret = H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
    VRFY((ret >= 0), "H5Pcreate xfer succeeded");
    if (dxfer_coll_type == DXFER_INDEPENDENT_IO) {
        ret = H5Pset_dxpl_mpio_collective_opt(xfer_plist, H5FD_MPIO_INDIVIDUAL_IO);
        VRFY((ret >= 0), "set independent IO collectively succeeded");
    }

    /* Raafat: Collective read prob 3
    */
    /* read data collectively */
    ret = H5Dread(dataset5, H5T_NATIVE_INT, mem_dataspace, file_dataspace, xfer_plist, cuda_buff);
    VRFY((ret >= 0), "H5Dread dataset5 succeeded");

    cudaMemcpy(data_array1, cuda_buff, DIM0 * DIM1 * sizeof(DATATYPE), cudaMemcpyDeviceToHost);
    ret = dataset_vrfy(start, count, stride, block, data_array1, data_origin1);
    if (ret)
        nerrors++;

    /* release all temporary handles. */
    H5Sclose(file_dataspace);
    H5Sclose(mem_dataspace);
    H5Pclose(xfer_plist);

    if (data_array1)
        free(data_array1);
    data_array1 = (DATATYPE *)malloc(DIM0 * DIM1 * sizeof(DATATYPE));
    VRFY((data_array1 != NULL), "data_array1 malloc succeeded");

    if (NULL != cuda_buff)
        cudaFree(&cuda_buff);
    cudaMalloc((void **)&cuda_buff, DIM0 * DIM1 * sizeof(DATATYPE));

    /* Dataset6: point selection in File - Point selection in Memory*/
    /* create a file dataspace independently */
    start[0] = DIM0 / mpi_size * mpi_rank;
    start[1] = 0;
    point_set(start, count, stride, block, num_points, coords, IN_ORDER);
    file_dataspace = H5Dget_space(dataset6);
    VRFY((file_dataspace >= 0), "H5Dget_space succeeded");
    ret = H5Sselect_elements(file_dataspace, H5S_SELECT_SET, num_points, coords);
    VRFY((ret >= 0), "H5Sselect_elements succeeded");

    start[0] = 0;
    start[1] = 0;
    point_set(start, count, stride, block, num_points, coords, OUT_OF_ORDER);
    mem_dataspace = H5Dget_space(dataset6);
    VRFY((mem_dataspace >= 0), "H5Dget_space succeeded");
    ret = H5Sselect_elements(mem_dataspace, H5S_SELECT_SET, num_points, coords);
    VRFY((ret >= 0), "H5Sselect_elements succeeded");

    /* set up the collective transfer properties list */
    xfer_plist = H5Pcreate(H5P_DATASET_XFER);
    VRFY((xfer_plist >= 0), "");
    ret = H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
    VRFY((ret >= 0), "H5Pcreate xfer succeeded");
    if (dxfer_coll_type == DXFER_INDEPENDENT_IO) {
        ret = H5Pset_dxpl_mpio_collective_opt(xfer_plist, H5FD_MPIO_INDIVIDUAL_IO);
        VRFY((ret >= 0), "set independent IO collectively succeeded");
    }

    /* Raafat: Collective read prob 4
    */
    /* read data collectively */
    ret = H5Dread(dataset6, H5T_NATIVE_INT, mem_dataspace, file_dataspace, xfer_plist, cuda_buff);
    VRFY((ret >= 0), "H5Dread dataset6 succeeded");

    cudaMemcpy(data_array1, cuda_buff, DIM0 * DIM1 * sizeof(DATATYPE), cudaMemcpyDeviceToHost);
    ret = dataset_vrfy(start, count, stride, block, data_array1, data_origin1);
    if (ret)
        nerrors++;

    /* release all temporary handles. */
    H5Sclose(file_dataspace);
    H5Sclose(mem_dataspace);
    H5Pclose(xfer_plist);

    if (data_array1)
        free(data_array1);
    data_array1 = (DATATYPE *)malloc(DIM0 * DIM1 * sizeof(DATATYPE));
    VRFY((data_array1 != NULL), "data_array1 malloc succeeded");

    if (NULL != cuda_buff)
        cudaFree(&cuda_buff);
    cudaMalloc((void **)&cuda_buff, DIM0 * DIM1 * sizeof(DATATYPE));

    /* Dataset7: point selection in memory - All selection in file*/
    /* create a file dataspace independently */
    file_dataspace = H5Dget_space(dataset7);
    VRFY((file_dataspace >= 0), "H5Dget_space succeeded");
    ret = H5Sselect_all(file_dataspace);
    VRFY((ret >= 0), "H5Sselect_all succeeded");

    num_points = DIM0 * DIM1;
    k          = 0;
    for (i = 0; i < DIM0; i++) {
        for (j = 0; j < DIM1; j++) {
            coords[k++] = i;
            coords[k++] = j;
        }
    }
    mem_dataspace = H5Dget_space(dataset7);
    VRFY((mem_dataspace >= 0), "H5Dget_space succeeded");
    ret = H5Sselect_elements(mem_dataspace, H5S_SELECT_SET, num_points, coords);
    VRFY((ret >= 0), "H5Sselect_elements succeeded");

    /* set up the collective transfer properties list */
    xfer_plist = H5Pcreate(H5P_DATASET_XFER);
    VRFY((xfer_plist >= 0), "");
    ret = H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
    VRFY((ret >= 0), "H5Pcreate xfer succeeded");
    if (dxfer_coll_type == DXFER_INDEPENDENT_IO) {
        ret = H5Pset_dxpl_mpio_collective_opt(xfer_plist, H5FD_MPIO_INDIVIDUAL_IO);
        VRFY((ret >= 0), "set independent IO collectively succeeded");
    }

    /* read data collectively */
    ret = H5Dread(dataset7, H5T_NATIVE_INT, mem_dataspace, file_dataspace, xfer_plist, cuda_buff);
    VRFY((ret >= 0), "H5Dread dataset7 succeeded");

    cudaMemcpy(data_array1, cuda_buff, DIM0 * DIM1 * sizeof(DATATYPE), cudaMemcpyDeviceToHost);

    start[0] = DIM0 / mpi_size * mpi_rank;
    start[1] = 0;
    ret      = dataset_vrfy(start, count, stride, block, data_array1 + (DIM0 / mpi_size * DIM1 * mpi_rank),
                       data_origin1);
    if (ret)
        nerrors++;

    /* release all temporary handles. */
    H5Sclose(file_dataspace);
    H5Sclose(mem_dataspace);
    H5Pclose(xfer_plist);

    /*
     * All reads completed.  Close datasets collectively
     */
    ret = H5Dclose(dataset1);
    VRFY((ret >= 0), "H5Dclose1 succeeded");
    ret = H5Dclose(dataset2);
    VRFY((ret >= 0), "H5Dclose2 succeeded");
    ret = H5Dclose(dataset5);
    VRFY((ret >= 0), "H5Dclose5 succeeded");
    ret = H5Dclose(dataset6);
    VRFY((ret >= 0), "H5Dclose6 succeeded");
    ret = H5Dclose(dataset7);
    VRFY((ret >= 0), "H5Dclose7 succeeded");

    /* close the file collectively */
    H5Fclose(fid);

    /* release data buffers */
    if (coords)
        free(coords);
    if (data_array1)
        free(data_array1);
    if (data_origin1)
        free(data_origin1);
    if (NULL != cuda_buff)
        cudaFree(&cuda_buff);
}

static void
dataset_writeInd_cuda(void)
{
    hid_t       fid;                /* HDF5 file ID */
    hid_t       acc_tpl;            /* File access templates */
    hid_t       sid;                /* Dataspace ID */
    hid_t       file_dataspace;     /* File dataspace ID */
    hid_t       mem_dataspace;      /* memory dataspace ID */
    hid_t       dataset1, dataset2; /* Dataset ID */
    hsize_t     dims[RANK];         /* dataset dim sizes */
    DATATYPE *  data_array1 = NULL; /* data buffer */
    DATATYPE *  cuda_buff = NULL; /* data buffer */

    hsize_t start[RANK];               /* for hyperslab setting */
    hsize_t count[RANK], stride[RANK]; /* for hyperslab setting */
    hsize_t block[RANK];               /* for hyperslab setting */

    herr_t ret; /* Generic return value */

    printf("Independent write test on file %s\n", FILENAME);

    /* allocate memory for data buffer */
    data_array1 = (DATATYPE *)malloc(DIM0 * DIM1 * sizeof(DATATYPE));
    VRFY((data_array1 != NULL), "data_array1 malloc succeeded");

    /* ----------------------------------------
     * CREATE AN HDF5 FILE WITH PARALLEL ACCESS
     * ---------------------------------------*/
    /* setup file access template */
    acc_tpl = H5Pcreate(H5P_FILE_ACCESS);
    VRFY((acc_tpl >= 0), "H5Pcreate H5P_FILE_ACCESS");
    ret = H5Pset_fapl_mpio(acc_tpl, comm, info);
    VRFY((ret >= 0), "H5Pset_fapl_mpio");
    ret = H5Pset_all_coll_metadata_ops(acc_tpl, true);
    VRFY((ret >= 0), "H5Pset_all_coll_metadata_ops");
    ret = H5Pset_coll_metadata_write(acc_tpl, true);
    VRFY((ret >= 0), "H5Pset_coll_metadata_write");

    /* create the file collectively */
    fid = H5Fcreate(FILENAME, H5F_ACC_TRUNC, H5P_DEFAULT, acc_tpl);
    VRFY((fid >= 0), "H5Fcreate succeeded");

    /* Release file-access template */
    ret = H5Pclose(acc_tpl);
    VRFY((ret >= 0), "");

    /* ---------------------------------------------
     * Define the dimensions of the overall datasets
     * and the slabs local to the MPI process.
     * ------------------------------------------- */
    /* setup dimensionality object */
    dims[0] = DIM0;
    dims[1] = DIM1;
    sid     = H5Screate_simple(RANK, dims, NULL);
    VRFY((sid >= 0), "H5Screate_simple succeeded");

    /* create a dataset collectively */
    dataset1 = H5Dcreate2(fid, DATASETNAME1, H5T_NATIVE_INT, sid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    VRFY((dataset1 >= 0), "H5Dcreate2 succeeded");

    /* create another dataset collectively */
    dataset2 = H5Dcreate2(fid, DATASETNAME2, H5T_NATIVE_INT, sid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    VRFY((dataset2 >= 0), "H5Dcreate2 succeeded");

    /*
     * To test the independent orders of writes between processes, all
     * even number processes write to dataset1 first, then dataset2.
     * All odd number processes write to dataset2 first, then dataset1.
     */

    /* set up dimensions of the slab this process accesses */
    slab_set(mpi_rank, mpi_size, start, count, stride, block, BYROW);

    /* put some trivial data in the data_array */
    dataset_fill(start, block, data_array1);

    cudaMalloc((void **)&cuda_buff, DIM0 * DIM1 * sizeof(DATATYPE));
    cudaMemcpy(cuda_buff, data_array1, DIM0 * DIM1 * sizeof(DATATYPE), cudaMemcpyHostToDevice);

    /* create a file dataspace independently */
    file_dataspace = H5Dget_space(dataset1);
    VRFY((file_dataspace >= 0), "H5Dget_space succeeded");
    ret = H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, stride, count, block);
    VRFY((ret >= 0), "H5Sset_hyperslab succeeded");

    /* create a memory dataspace independently */
    mem_dataspace = H5Screate_simple(RANK, block, NULL);
    VRFY((mem_dataspace >= 0), "");

    /* write data independently */
    ret = H5Dwrite(dataset1, H5T_NATIVE_INT, mem_dataspace, file_dataspace, H5P_DEFAULT, cuda_buff);
    VRFY((ret >= 0), "H5Dwrite dataset1 succeeded");
    /* write data independently */
    ret = H5Dwrite(dataset2, H5T_NATIVE_INT, mem_dataspace, file_dataspace, H5P_DEFAULT, cuda_buff);
    VRFY((ret >= 0), "H5Dwrite dataset2 succeeded");

    /* setup dimensions again to write with zero rows for process 0 */
    printf("writeInd by some with zero row\n");
    slab_set(mpi_rank, mpi_size, start, count, stride, block, ZROW);
    ret = H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, stride, count, block);
    VRFY((ret >= 0), "H5Sset_hyperslab succeeded");
    /* need to make mem_dataspace to match for process 0 */
    if (MAINPROCESS) {
        ret = H5Sselect_hyperslab(mem_dataspace, H5S_SELECT_SET, start, stride, count, block);
        VRFY((ret >= 0), "H5Sset_hyperslab mem_dataspace succeeded");
    }
    if ((mpi_rank / 2) * 2 != mpi_rank) {
        ret = H5Dwrite(dataset1, H5T_NATIVE_INT, mem_dataspace, file_dataspace, H5P_DEFAULT, cuda_buff);
        VRFY((ret >= 0), "H5Dwrite dataset1 by ZROW succeeded");
    }

    /* release dataspace ID */
    H5Sclose(file_dataspace);

    /* close dataset collectively */
    ret = H5Dclose(dataset1);
    VRFY((ret >= 0), "H5Dclose1 succeeded");
    ret = H5Dclose(dataset2);
    VRFY((ret >= 0), "H5Dclose2 succeeded");

    /* release all IDs created */
    H5Sclose(sid);

    /* close the file collectively */
    H5Fclose(fid);

    /* release data buffers */
    if (data_array1)
        free(data_array1);
    cudaFree(&cuda_buff);
}

/* Example of using the parallel HDF5 library to read a dataset */
static void
dataset_readInd_cuda(void)
{
    hid_t       fid;                 /* HDF5 file ID */
    hid_t       acc_tpl;             /* File access templates */
    hid_t       file_dataspace;      /* File dataspace ID */
    hid_t       mem_dataspace;       /* memory dataspace ID */
    hid_t       dataset1, dataset2;  /* Dataset ID */
    DATATYPE *  data_array1  = NULL; /* data buffer */
    DATATYPE *  data_origin1 = NULL; /* expected data buffer */
    DATATYPE *  cuda_buff = NULL; /* data buffer */

    hsize_t start[RANK];               /* for hyperslab setting */
    hsize_t count[RANK], stride[RANK]; /* for hyperslab setting */
    hsize_t block[RANK];               /* for hyperslab setting */

    herr_t ret; /* Generic return value */

    printf("Independent read test on file %s\n", FILENAME);

    /* allocate memory for data buffer */
    data_array1 = (DATATYPE *)malloc(DIM0 * DIM1 * sizeof(DATATYPE));
    VRFY((data_array1 != NULL), "data_array1 malloc succeeded");
    data_origin1 = (DATATYPE *)malloc(DIM0 * DIM1 * sizeof(DATATYPE));
    VRFY((data_origin1 != NULL), "data_origin1 malloc succeeded");

    /* setup file access template */
    acc_tpl = H5Pcreate(H5P_FILE_ACCESS);
    VRFY((acc_tpl >= 0), "H5Pcreate H5P_FILE_ACCESS");
    ret = H5Pset_fapl_mpio(acc_tpl, comm, info);
    VRFY((ret >= 0), "H5Pset_fapl_mpio");
    ret = H5Pset_all_coll_metadata_ops(acc_tpl, true);
    VRFY((ret >= 0), "H5Pset_all_coll_metadata_ops");
    ret = H5Pset_coll_metadata_write(acc_tpl, true);
    VRFY((ret >= 0), "H5Pset_coll_metadata_write");

    /* open the file collectively */
    fid = H5Fopen(FILENAME, H5F_ACC_RDONLY, acc_tpl);
    VRFY((fid >= 0), "");

    /* Release file-access template */
    ret = H5Pclose(acc_tpl);
    VRFY((ret >= 0), "");

    /* open the dataset1 collectively */
    dataset1 = H5Dopen2(fid, DATASETNAME1, H5P_DEFAULT);
    VRFY((dataset1 >= 0), "");

    /* open another dataset collectively */
    dataset2 = H5Dopen2(fid, DATASETNAME1, H5P_DEFAULT);
    VRFY((dataset2 >= 0), "");

    /* set up dimensions of the slab this process accesses */
    slab_set(mpi_rank, mpi_size, start, count, stride, block, BYROW);

    /* create a file dataspace independently */
    file_dataspace = H5Dget_space(dataset1);
    VRFY((file_dataspace >= 0), "");
    ret = H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, stride, count, block);
    VRFY((ret >= 0), "");

    /* create a memory dataspace independently */
    mem_dataspace = H5Screate_simple(RANK, block, NULL);
    VRFY((mem_dataspace >= 0), "");

    /* fill dataset with test data */
    dataset_fill(start, block, data_origin1);

    cudaMalloc((void **)&cuda_buff, DIM0 * DIM1 * sizeof(DATATYPE));

    /* read data independently */
    ret = H5Dread(dataset1, H5T_NATIVE_INT, mem_dataspace, file_dataspace, H5P_DEFAULT, cuda_buff);
    VRFY((ret >= 0), "");

    cudaMemcpy(data_array1, cuda_buff, DIM0 * DIM1 * sizeof(DATATYPE), cudaMemcpyDeviceToHost);

    /* verify the read data with original expected data */
    ret = dataset_vrfy(start, count, stride, block, data_array1, data_origin1);
    if (ret)
        nerrors++;

    /* read data independently */
    ret = H5Dread(dataset2, H5T_NATIVE_INT, mem_dataspace, file_dataspace, H5P_DEFAULT, cuda_buff);
    VRFY((ret >= 0), "");

    cudaMemcpy(data_array1, cuda_buff, DIM0 * DIM1 * sizeof(DATATYPE), cudaMemcpyDeviceToHost);

    /* verify the read data with original expected data */
    ret = dataset_vrfy(start, count, stride, block, data_array1, data_origin1);
    if (ret)
        nerrors++;

    /* close dataset collectively */
    ret = H5Dclose(dataset1);
    VRFY((ret >= 0), "");
    ret = H5Dclose(dataset2);
    VRFY((ret >= 0), "");

    /* release all IDs created */
    H5Sclose(file_dataspace);

    /* close the file collectively */
    H5Fclose(fid);

    /* release data buffers */
    if (data_array1)
        free(data_array1);
    if (data_origin1)
        free(data_origin1);
    cudaFree(&cuda_buff);
}

static void
extend_writeInd2_cuda(void)
{
    hid_t       fid;             /* HDF5 file ID */
    hid_t       fapl;            /* File access templates */
    hid_t       fs;              /* File dataspace ID */
    hid_t       ms;              /* Memory dataspace ID */
    hid_t       dataset;         /* Dataset ID */
    hsize_t     orig_size  = 10; /* Original dataset dim size */
    hsize_t     new_size   = 20; /* Extended dataset dim size */
    hsize_t     one        = 1;
    hsize_t     max_size   = H5S_UNLIMITED; /* dataset maximum dim size */
    hsize_t     chunk_size = 16384;         /* chunk size */
    hid_t       dcpl;                       /* dataset create prop. list */
    int         written[10],                /* Data to write */
        retrieved[10];                      /* Data read in */
    int    i;                               /* Local index variable */
    herr_t ret;                             /* Generic return value */
    int *  cuda_written   = NULL;           /* data buffer */
    int *  cuda_retrieved = NULL;           /* data buffer */

    cudaMalloc((void **)&cuda_written, 10 * sizeof(int));
    cudaMalloc((void **)&cuda_retrieved, 10 * sizeof(int));

    printf("Extend independent write test #2 on file %s\n", FILENAME);

    /* -------------------
     * START AN HDF5 FILE
     * -------------------*/
    /* setup file access template */
    fapl = H5Pcreate(H5P_FILE_ACCESS);
    VRFY((fapl >= 0), "H5Pcreate H5P_FILE_ACCESS");
    ret = H5Pset_fapl_mpio(fapl, comm, info);
    VRFY((ret >= 0), "H5Pset_fapl_mpio");
    ret = H5Pset_all_coll_metadata_ops(fapl, true);
    VRFY((ret >= 0), "H5Pset_all_coll_metadata_ops");
    ret = H5Pset_coll_metadata_write(fapl, true);
    VRFY((ret >= 0), "H5Pset_coll_metadata_write");

    /* create the file collectively */
    fid = H5Fcreate(FILENAME, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    VRFY((fid >= 0), "H5Fcreate succeeded");

    /* Release file-access template */
    ret = H5Pclose(fapl);
    VRFY((ret >= 0), "H5Pclose succeeded");

    /* --------------------------------------------------------------
     * Define the dimensions of the overall datasets and create them.
     * ------------------------------------------------------------- */

    /* set up dataset storage chunk sizes and creation property list */
    dcpl = H5Pcreate(H5P_DATASET_CREATE);
    VRFY((dcpl >= 0), "H5Pcreate succeeded");
    ret = H5Pset_chunk(dcpl, 1, &chunk_size);
    VRFY((ret >= 0), "H5Pset_chunk succeeded");

    /* setup dimensionality object */
    fs = H5Screate_simple(1, &orig_size, &max_size);
    VRFY((fs >= 0), "H5Screate_simple succeeded");

    /* create an extendible dataset collectively */
    dataset = H5Dcreate2(fid, DATASETNAME1, H5T_NATIVE_INT, fs, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    VRFY((dataset >= 0), "H5Dcreat2e succeeded");

    /* release resource */
    ret = H5Pclose(dcpl);
    VRFY((ret >= 0), "H5Pclose succeeded");

    /* -------------------------
     * Test writing to dataset
     * -------------------------*/
    /* create a memory dataspace independently */
    ms = H5Screate_simple(1, &orig_size, &max_size);
    VRFY((ms >= 0), "H5Screate_simple succeeded");

    /* put some trivial data in the data_array */
    for (i = 0; i < (int)orig_size; i++)
        written[i] = i;

    cudaMemcpy(cuda_written, written, 10 * sizeof(int), cudaMemcpyHostToDevice);

    ret = H5Dwrite(dataset, H5T_NATIVE_INT, ms, fs, H5P_DEFAULT, cuda_written);
    VRFY((ret >= 0), "H5Dwrite succeeded");

    /* -------------------------
     * Read initial data from dataset.
     * -------------------------*/
    ret = H5Dread(dataset, H5T_NATIVE_INT, ms, fs, H5P_DEFAULT, cuda_retrieved);
    VRFY((ret >= 0), "H5Dread succeeded");
    cudaMemcpy(retrieved, cuda_retrieved, 10 * sizeof(int), cudaMemcpyDeviceToHost);

    for (i = 0; i < (int)orig_size; i++)
        if (written[i] != retrieved[i]) {
            printf("Line #%d: written!=retrieved: written[%d]=%d, retrieved[%d]=%d\n", __LINE__, i,
                     written[i], i, retrieved[i]);
            nerrors++;
        }

    /* -------------------------
     * Extend the dataset & retrieve new dataspace
     * -------------------------*/
    ret = H5Dset_extent(dataset, &new_size);
    VRFY((ret >= 0), "H5Dset_extent succeeded");
    ret = H5Sclose(fs);
    VRFY((ret >= 0), "H5Sclose succeeded");
    fs = H5Dget_space(dataset);
    VRFY((fs >= 0), "H5Dget_space succeeded");

    /* -------------------------
     * Write to the second half of the dataset
     * -------------------------*/
    for (i = 0; i < (int)orig_size; i++)
        written[i] = orig_size + i;

    ret = H5Sselect_hyperslab(fs, H5S_SELECT_SET, &orig_size, NULL, &one, &orig_size);
    VRFY((ret >= 0), "H5Sselect_hyperslab succeeded");
    cudaMemcpy(cuda_written, written, 10 * sizeof(int), cudaMemcpyHostToDevice);
    ret = H5Dwrite(dataset, H5T_NATIVE_INT, ms, fs, H5P_DEFAULT, cuda_written);
    VRFY((ret >= 0), "H5Dwrite succeeded");

    /* -------------------------
     * Read the new data
     * -------------------------*/
    ret = H5Dread(dataset, H5T_NATIVE_INT, ms, fs, H5P_DEFAULT, cuda_retrieved);
    VRFY((ret >= 0), "H5Dread succeeded");
    cudaMemcpy(retrieved, cuda_retrieved, 10 * sizeof(int), cudaMemcpyDeviceToHost);

    for (i = 0; i < (int)orig_size; i++)
        if (written[i] != retrieved[i]) {
            printf("Line #%d: written!=retrieved: written[%d]=%d, retrieved[%d]=%d\n", __LINE__, i,
                     written[i], i, retrieved[i]);
            nerrors++;
        }

    /* Close dataset collectively */
    ret = H5Dclose(dataset);
    VRFY((ret >= 0), "H5Dclose succeeded");

    /* Close the file collectively */
    ret = H5Fclose(fid);
    VRFY((ret >= 0), "H5Fclose succeeded");

    cudaFree(&cuda_written);
    cudaFree(&cuda_retrieved);
}

int
main(int argc, char **argv)
{
    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &mpi_rank);

    H5open();

    if (MAINPROCESS) {
        printf("==========================\n");
        printf("GDS VFD tests\n");
        printf("==========================\n\n");
    }

    extend_writeInd_cuda();
    dataset_writeAll_cuda();
    dataset_readAll_cuda();
    dataset_writeInd_cuda();
    dataset_readInd_cuda();
    extend_writeInd2_cuda();

    if (nerrors)
        goto exit;

    if (MAINPROCESS)
        puts("All GDS VFD tests passed\n");

exit:
    if (nerrors)
        if (MAINPROCESS)
            printf("*** %d TEST ERROR%s OCCURRED ***\n", nerrors, nerrors > 1 ? "S" : "");

    H5E_BEGIN_TRY
    {
        H5Fdelete(FILENAME, H5P_DEFAULT);
    }
    H5E_END_TRY;

    H5close();

    MPI_Finalize();

    exit((nerrors ? EXIT_FAILURE : EXIT_SUCCESS));
}
