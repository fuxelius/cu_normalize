
#include <cuda_runtime.h>
#include "common/common.h"
#include <stdio.h>
#include <stdint.h>
#include "math.h"
#include "struct.h"
#include "makros.h"

// ====================================================== DATA STRUCTURES ==================================================================
// error_table[chunk_idx*CHUNK_SIZE + threadIdx.x] ??
// error_table[idx]
__device__ float error_table[META_SIZE * CHUNK_SIZE]; // meta_chunk_size * chunk_size = 102400 threads (410 kbyte)


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
//CUDA implementation, hold the number of (mxt,myt) pairs <= 1024 to fit on a single SM, important for calculating the sum??!!
__global__ void point_square_GPU(chunk_record *chunk_table, int chunk_len,
                                     mag_record *mag_table, int mag_len,
                                                            int chunk_size) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int chunk_idx = idx / chunk_size; // whole number
    int mag_idx = idx;

    printf("Point Square idx=%i chunk_idx=%i\n", idx, chunk_idx);

    if ((idx < mag_len) && !(mag_table[mag_idx].disable)) {
        // mag_table
        short mxt = mag_table[mag_idx].mxt;
        short myt = mag_table[mag_idx].myt;

        // // chunk_table
        // float x0            = chunk_table[chunk_idx].x0;
        // float y0            = chunk_table[chunk_idx].y0;
        // float scale_r       = chunk_table[chunk_idx].scale_r;
        // float scale_y_axis  = chunk_table[chunk_idx].scale_y_axis;
        // float theta         = chunk_table[chunk_idx].theta;

        // chunk_table; ga_uppsala2
        float x0            = 16;
        float y0            = -124;
        float scale_r       = 0.0041;
        float scale_y_axis  = 1.045;
        float theta         = 0;

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

        // Write back result
        //mag_table[mag_idx].normalized_x = normalized_x;
        //mag_table[mag_idx].normalized_y = normalized_y;
        //mag_table[mag_idx].quad_error   = quad_error;
        //result_table[mag_idx].mfv = normalized_x;
        //result_table[mag_idx].rho = normalized_y;

        //printf("first test x=%i, y=%i, x+y=%i\n", 3, 4, first_test(3,4));

     }
}


// ======================================================== PARENT LAUNCH =================================================================
__global__ void parent_launch(chunk_record *chunk_table, int chunk_len,
                                  mag_record *mag_table, int mag_len,
                                  meta_record *meta_table, int meta_len,
                                                         int chunk_size) {

    printf("Parent Launch: %i\n", threadIdx.x);

    int meta_idx = 0;
    int left_chunk_idx = 0;
    int right_chunk_idx = 0;

    left_chunk_idx  = meta_table[meta_idx].left_chunk_idx;
    right_chunk_idx = meta_table[meta_idx].right_chunk_idx;

    point_square_GPU<<<1, ((right_chunk_idx-left_chunk_idx+1) * chunk_size)>>>(chunk_table, chunk_len, mag_table, mag_len, CHUNK_SIZE);

    cudaDeviceSynchronize();

    int chunk_idx = 0;

    float sum = sum_vector(chunk_idx, CHUNK_SIZE);

    cudaDeviceSynchronize();

    printf("LSQ=%f for chunk_idx=%i\n", sum, chunk_idx);
}


// ========================================================= HOST LAUNCH ==================================================================
void host_launch(chunk_record *chunk_table, int chunk_len,
                                  mag_record *mag_table, int mag_len,
                                  meta_record *meta_table, int meta_len,
                                                         int chunk_size) {
    printf("Host Launch:\n");

    parent_launch<<<1,1>>>(chunk_table, chunk_len, mag_table, mag_len, meta_table, meta_len, chunk_size);
}
