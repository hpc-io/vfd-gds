#include <hdf5.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "H5FDgds.h"

#define FILENAME "gds_simple_dset_write.h5"
#define DIM_SIZE 10

int
main(int argc, char **argv)
{
    hsize_t  i, dims[] = { DIM_SIZE, DIM_SIZE };
    hid_t    file_id = H5I_INVALID_HID;
    hid_t    fapl_id = H5I_INVALID_HID;
    hid_t    dset_id = H5I_INVALID_HID;
    hid_t    space_id = H5I_INVALID_HID;
    int      data[DIM_SIZE * DIM_SIZE];
    int     *cuda_buf = NULL;

    fapl_id = H5Pcreate(H5P_FILE_ACCESS);

    H5Pset_fapl_gds(fapl_id, MBOUNDARY_DEF, FBSIZE_DEF, CBSIZE_DEF);

    file_id = H5Fcreate(FILENAME, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);

    space_id = H5Screate_simple(2, dims, NULL);

    dset_id = H5Dcreate2(file_id, "dset", H5T_NATIVE_INT, space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    for (i = 0; i < dims[0] * dims[1]; i++)
        data[i] = i;

    /* Allocate memory on CUDA device and copy data into that buffer */
    cudaMalloc((void **)&cuda_buf, DIM_SIZE * DIM_SIZE * sizeof(int));
    cudaMemcpy(cuda_buf, data, DIM_SIZE * DIM_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    /* Write dataset using buffer allocated on CUDA device */
    H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, cuda_buf);

    cudaFree(&cuda_buf);

    H5Sclose(space_id);
    H5Dclose(dset_id);
    H5Pclose(fapl_id);
    H5Fclose(file_id);

    return 0;
}

