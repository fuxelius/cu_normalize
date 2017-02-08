
#include <cuda_runtime.h>
#include "common/common.h"
#include <stdio.h>
#include <stdint.h>
#include <curand.h>
#include <curand_kernel.h>
#include "math.h"
#include "struct.h"
#include "makros.h"

// ====================================================== DATA STRUCTURES ==================================================================
// error_table[mag_idx] ??
__device__ float *error_table; //[META_SIZE * CHUNK_SIZE]; meta_size * chunk_size= 100*1024=102400 threads (410 kbyte)

// set the errors VERY high as initialization; first round will get them normal; no initialization step nessesary
__device__ void initialize_error_table(int meta_size, int chunk_size) {
    error_table =(float*) malloc(meta_size*chunk_size * sizeof(float));

    for (int idx=0; idx<(meta_size * chunk_size); idx++) {
        error_table[idx] = 1000000;
    }
}
// ----------------------------------------------------------------------------------------------------
__device__ rand_record *rand_table; //[META_SIZE]; contains random values

// set everything to 1 in the first round; no initialization step nessesary
__device__ void initialize_rand_table(int meta_size) {
    rand_table =(rand_record*) malloc(meta_size * sizeof(rand_record));

    for (int meta_idx=0; meta_idx<meta_size; meta_idx++) {
        rand_table[meta_idx].rand_1 = 1.00;  // 1 makes x0 unchanged in first round of point_square_GPU
        rand_table[meta_idx].rand_2 = 1.00;  // 1 makes y0 unchanged in first round of point_square_GPU
        rand_table[meta_idx].rand_3 = 1.00;  // 1 makes scale_r unchanged in first round of point_square_GPU
        rand_table[meta_idx].rand_4 = 1.00;  // 1 makes scale_y_axis unchanged in first round of point_square_GPU
        rand_table[meta_idx].rand_5 = 1.00;  // 1 makes theta unchanged in first round of point_square_GPU
    }
}
// ========================================================== RANDOMIZE ====================================================================
__device__ float randomize(void) { // Return a random number between 0-1, make a simple implementation hmm use CURAND
    return 1;
}


// ======================================================== SUM A VECTOR ===================================================================
// this is probably very efficient ... if running on 100 processors in paralell ... and only 1024 loops ;)
__device__ float sum_vector(int chunk_idx, int chunk_size) {
    int off_set = chunk_idx*chunk_size;
    float sum = 0;

    for (int idx=off_set; idx < off_set+chunk_size; idx++) {
        sum = sum + error_table[idx];
    }

    printf("Sum vector: chunk_idx %i (%f)\n", chunk_idx, sum);

    return sqrtf(sum);
}


// ======================================================== POINT SQUARE ==================================================================
// CUDA implementation, hold the number of (mxt, myt) pairs <= 1024 to fit on a single SM, important for calculating the sum??!!
__global__ void point_square_GPU(chunk_record *chunk_table, int chunk_len,
                                     mag_record *mag_table, int mag_len,
                                            int chunk_size, int meta_idx) {




    // <------------------------------------------ uppdetera denna för meta_idx som blir offset för idx, chunk_idx !!!!!!!!!!!!
    //




    // int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // int chunk_idx = idx / chunk_size; // = meta_idx*META_SIZE  + idx / chunk_size
    // int mag_idx = idx;                // = meta_idx*chunk_size + idx

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int chunk_idx = meta_idx*META_SIZE  + idx / chunk_size;
    int mag_idx = meta_idx*chunk_size + idx;

    //printf("Point Square mag_idx=%i, chunk_idx=%i, meta_idx=%i\n", mag_idx, chunk_idx, meta_idx);

    if ((idx < mag_len) && !(mag_table[mag_idx].disable)) {
        // mag_table
        short mxt = mag_table[mag_idx].mxt;
        short myt = mag_table[mag_idx].myt;

        // // chunk_table
        float x0            = chunk_table[chunk_idx].x0;            // multiply with rand_table[].rand_1
        float y0            = chunk_table[chunk_idx].y0;            // multiply with rand_table[].rand_2
        float scale_r       = chunk_table[chunk_idx].scale_r;       // multiply with rand_table[].rand_3
        float scale_y_axis  = chunk_table[chunk_idx].scale_y_axis;  // multiply with rand_table[].rand_4
        float theta         = chunk_table[chunk_idx].theta;         // multiply with rand_table[].rand_5

        // chunk_table; ga_uppsala2
        // float x0            = 16;
        // float y0            = -124;
        // float scale_r       = 0.0041;
        // float scale_y_axis  = 1.045;
        // float theta         = 0;

        //printf("raw,%f,%f\n", mxt, myt);

        float trans_x = (mxt - x0);  // move plane to set origo to in middle of ellipse
        float trans_y = (myt - y0);  // move plane to set origo to in middle of ellipse

        //printf("trans,%f,%f\n", trans_x, trans_y);

        // Vector is individual for each rotated pair (mx,my)
        // unless they are situated on a perfect circle which
        // measured values are NOT!! (otherwise v would be r)

        // https://en.wikipedia.org/wiki/Rotation_(mathematics)
        // https://en.wikipedia.org/wiki/C_mathematical_functions

        float rotate_x = trans_x * cosf(theta) - trans_y * sinf(theta);  // Rotate theta degrees
        float rotate_y = trans_x * sinf(theta) + trans_y * cosf(theta);  // Rotate theta degrees

        //printf("rotate,%f,%f\n", rotate_x, rotate_y);

        float scale_x = rotate_x;
        float scale_y = rotate_y * scale_y_axis;                         // Scale y-axis to make the ellips to a cicle

        //printf("scale,%f,%f\n", scale_x, scale_y); // För R-plot

        rotate_x = scale_x * cosf(-theta) - scale_y * sinf(-theta);      // Rotate -theta degrees (back)
        rotate_y = scale_x * sinf(-theta) + scale_y * cosf(-theta);      // Rotate -theta degrees (back)

        //printf("rotate back ,%f,%f\n", rotate_x, rotate_y);

        float normalized_x = rotate_x * scale_r;                        // Returns x,y normalized to unit circle
        float normalized_y = rotate_y * scale_r;

        //printf("normalized ,%f,%f\n", normalized_x, normalized_y);

        //printf("cuda,%f,%f\n", scale_x, scale_y);

        float quad_error = powf(sqrtf(powf(normalized_x,2) + powf(normalized_y,2)) - 1,2); // Returns square error from unity cicle

        //printf("quad_error,%f\n", quad_error);

        error_table[idx] = quad_error;

        //printf("first test x=%i, y=%i, x+y=%i\n", 3, 4, first_test(3,4));

     }
}


// ======================================================== PARENT LAUNCH ==================================================================
// this is main() for CUDA
__global__ void cuda_main(chunk_record *chunk_table, int chunk_len, mag_record *mag_table, int mag_len, meta_record *meta_table, int meta_len) {
    printf("Device Launch\n");

    for (int meta_idx=0; meta_idx<META_SIZE; meta_idx++) {
        initialize_error_table(META_SIZE, CHUNK_SIZE); // all values set to 100000
        initialize_rand_table(META_SIZE);  // all values set to 1.00

        for (int round=0; round<1; round++) { // iterate 100.000 times
            printf("meta_idx=%i, round=%i\n", meta_idx, round);

            // int meta_idx = 0;
            // int left_chunk_idx = 0;
            // int right_chunk_idx = 0;
            //
            // left_chunk_idx  = meta_table[meta_idx].left_chunk_idx;
            // right_chunk_idx = meta_table[meta_idx].right_chunk_idx;

            int dimx = BLOCK_SIZE; // Set in struct.h, should be smaller than chunk_size
            dim3 block(dimx, 1);
            dim3 grid((META_SIZE * CHUNK_SIZE + BLOCK_SIZE - 1)/ BLOCK_SIZE, 1);
            point_square_GPU<<<grid,block>>>(chunk_table, chunk_len, mag_table, mag_len, CHUNK_SIZE, meta_idx);

            cudaDeviceSynchronize();

            int chunk_idx = 0;

            float lsq = sum_vector(chunk_idx, CHUNK_SIZE);

            cudaDeviceSynchronize();

            // if lsq < chunk_table[idx].lsq
            //    update x0,y0 ... with the randomized values
            //    chunk_table[idx].iter++
            // else
            //    chunk_table[idx].iter++
            //
            // update random_table
            //
            // rerun everything :)

            printf("LSQ=%f for chunk_idx=%i\n", lsq, chunk_idx);
        }
    }


    free(error_table);
    free(rand_table);
}


// ========================================================= HOST LAUNCH ==================================================================
void host_launch(chunk_record *chunk_table, int chunk_len, mag_record *mag_table, int mag_len, meta_record *meta_table, int meta_len) {
    printf("Host Launch:\n");

    cuda_main<<<1,1>>>(chunk_table, chunk_len, mag_table, mag_len, meta_table, meta_len); // Kernel running on only one single core

}
