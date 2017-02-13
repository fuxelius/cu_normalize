
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
// error_table[mag_idx]
__device__ float *error_table; // [META_SIZE * CHUNK_SIZE] meta_size * chunk_size= 100*1024=102400 threads (410 kbyte)

__device__ void initialize_error_table(int meta_size, int chunk_size) {
    error_table =(float*) malloc(meta_size*chunk_size * sizeof(float));
}

// ----------------------------------------------------------------------------------------------------

// Use the same random numbers for all sets we are working with
__device__ float rand_table[5]; // contains 5 random values for altering x0, y0, ...

// first round x0, y0, ... is untouched and is then set to its initial values from histogram
__device__ void initialize_rand_table(void) {
    rand_table[0] = 1;
    rand_table[1] = 1;
    rand_table[2] = 1;
    rand_table[3] = 1;
    rand_table[4] = 1;
}
// ================================================ RANDOM NUMBER GENERATOR (RNG) ==========================================================

__global__ void setup_kernel(curandState * state, unsigned long seed) {
    int id = threadIdx.x;
    curand_init(seed, id, 0, &state[id]);
}

// Normal (Gaussian) distribution around 0.0f with stdandard deviation of 1.0f
__global__ void generate_rand(curandState* globalState) {
    int idx = threadIdx.x;
    curandState localState = globalState[idx];
    //float RANDOM = curand_uniform( &localState );
    float RANDOM = curand_normal(&localState);
    globalState[idx] = localState;

    float randy = RANDOM/10 + 1;

    if (randy > 0.7 && randy < 1.3) {
        //printf("1>random=%f\n", randy);
        rand_table[idx] = randy;
    }
    else {
        //printf("2>random=%f (%f)\n", 1.0, randy);
        rand_table[idx] = 1.00;
    }
    globalState[idx] = localState;
}

// ======================================================== SUM A VECTOR ===================================================================
// this is probably very efficient ... if running on 100 processors in paralell ... and only 1024 loops ;)

//__global__ void sum_vector_eval(int meta_idx, chunk_record *chunk_table, int chunk_len, int chunk_size, int meta_len) {
__global__ void sum_vector_eval(int meta_idx, chunk_record *chunk_table, int chunk_len, int chunk_size) {
    int idx = threadIdx.x;
    int chunk_idx = meta_idx*META_SIZE + idx;
    int off_set = idx*chunk_size;

    if (chunk_idx < chunk_len) { // cut out all other created threads based on threadIdx.x
    //if (chunk_idx < chunk_len && meta_idx < meta_len) { // cut out all other created threads based on threadIdx.x <------------------------------ kolla upp denna
        float sum = 0;
        for (int error_idx=off_set; error_idx < (off_set+chunk_size); error_idx++) {
            sum = sum + error_table[error_idx];
        }

        float lsq = sqrtf(sum);

        // if approximation is better update chunk_table[chunk_idx]
        if (lsq < chunk_table[chunk_idx].lsq) {

            printf("----->Updated meta_idx=%i chunk_idx=%i old_lsq=%f new_lsq=%f\n", meta_idx, chunk_idx, chunk_table[chunk_idx].lsq, lsq);

            chunk_table[chunk_idx].x0           = chunk_table[chunk_idx].x0 * rand_table[0];
            chunk_table[chunk_idx].y0           = chunk_table[chunk_idx].y0 * rand_table[1];
            chunk_table[chunk_idx].scale_r      = chunk_table[chunk_idx].scale_r * rand_table[2];
            chunk_table[chunk_idx].scale_y_axis = chunk_table[chunk_idx].scale_y_axis * rand_table[3];
            chunk_table[chunk_idx].theta        = chunk_table[chunk_idx].theta * rand_table[4];

            chunk_table[chunk_idx].lsq = lsq;
            chunk_table[chunk_idx].iter_cnt++;
        }

        //printf("----->Sum vector: chunk_idx=%i lsq=%f chunk_lsq=%f\n", chunk_idx, lsq, chunk_table[chunk_idx].lsq);
    }
}

// ======================================================== POINT SQUARE ==================================================================
// CUDA implementation, hold the number of (mxt, myt) pairs <= 1024 to fit on a single SM, important for calculating the sum??!!

__global__ void point_square(chunk_record *chunk_table, int chunk_len, mag_record *mag_table, int mag_len, int chunk_size, int meta_idx) {
//__global__ void point_square(chunk_record *chunk_table, int chunk_len, mag_record *mag_table, int mag_len, int chunk_size, int meta_idx, int meta_len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // position inside meta_idx
    int chunk_idx = meta_idx*META_SIZE + idx / chunk_size;
    int error_idx = idx;
    int mag_idx = meta_idx*META_SIZE*chunk_size + idx;

    //printf("Point Square: meta_idx=%i idx=%i error_idx=%i chunk_idx=%i mag_idx=%i\n", meta_idx, idx, error_idx, chunk_idx, mag_idx);

    // cut out all other created threads based on threadIdx.x, otherwise they WILL write out of bound -- and krashes :(
    if (!mag_table[error_idx].disable && mag_idx < mag_len && chunk_idx < chunk_len && error_idx < META_SIZE*chunk_size) {
    //if (mag_idx < mag_len && chunk_idx < chunk_len && meta_idx < meta_len) { // cut out all other created threads based on threadIdx.x
    //if (idx < META_SIZE*chunk_size && chunk_idx < chunk_len && meta_idx < meta_len) { // cut out all other created threads based on threadIdx.x

        // mag_table
        short mxt = mag_table[mag_idx].mxt;
        short myt = mag_table[mag_idx].myt;

        // // chunk_table, temporary values
        float x0            = chunk_table[chunk_idx].x0           * rand_table[0]; // rand_table is 1 first round
        float y0            = chunk_table[chunk_idx].y0           * rand_table[1]; // rand_table is 1 first round
        float scale_r       = chunk_table[chunk_idx].scale_r      * rand_table[2]; // rand_table is 1 first round
        float scale_y_axis  = chunk_table[chunk_idx].scale_y_axis * rand_table[3]; // rand_table is 1 first round
        float theta         = chunk_table[chunk_idx].theta        * rand_table[4]; // rand_table is 1 first round

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

        // if (! mag_table[error_idx].disable) {
        //     error_table[error_idx] = quad_error;
        // }
        // else {
        //     error_table[error_idx] = 0;
        // }

        error_table[error_idx] = quad_error;

        //printf("first test x=%i, y=%i, x+y=%i\n", 3, 4, first_test(3,4));
     }
     else {
        error_table[error_idx] = 0;
     }
}


// ======================================================== PARENT LAUNCH ==================================================================
// this is main() for CUDA
__global__ void cuda_main(chunk_record *chunk_table, int chunk_len, mag_record *mag_table, int mag_len, meta_record *meta_table, int meta_len) {

    printf("Device Launch\n");

    // for (int i = 0; i < chunk_len; i++) {
    //     //printf("chunk_idx=%i left_mag_idx=%i right_mag_idx=%i\n", i, chunk_table[i].left_mag_idx, chunk_table[i].right_mag_idx);
    //     printf("chunk_idx=%i chunk_lsq=%f\n", i, chunk_table[i].lsq );
    //
    // }

    // ----------------------------- RANDOMIZE SETUP  -------------------------------------------
    int N = 5; // generate 5 random numbers
    curandState* devStates = (curandState*) malloc( N*sizeof(curandState));

    // setup seeds
    setup_kernel<<<1,N>>>(devStates, clock64());
    // ------------------------------------------------------------------------------------------

    int max_iter = 200; // Maximum iteration depth 100,000

    for (int meta_idx=0; meta_idx<meta_len; meta_idx++) {
        initialize_error_table(META_SIZE, CHUNK_SIZE);      //
        initialize_rand_table();                            // all values set to 1.00

        for (int round=0; round<max_iter; round++) {        // iterate 100.000 times
            printf("meta_idx=%i, round=%i\n", meta_idx, round);

            dim3 grid((META_SIZE * CHUNK_SIZE + BLOCK_SIZE - 1)/ BLOCK_SIZE, 1);
            point_square<<<grid,BLOCK_SIZE>>>(chunk_table, chunk_len, mag_table, mag_len, CHUNK_SIZE, meta_idx);
            //point_square<<<grid,BLOCK_SIZE>>>(chunk_table, chunk_len, mag_table, mag_len, CHUNK_SIZE, meta_idx, meta_len);

            //cudaDeviceSynchronize();

            //sum_vector_eval<<<1,META_SIZE>>>(meta_idx, chunk_table, chunk_len, CHUNK_SIZE, meta_len); // summera alla paralellt och uppdatera chunk
            sum_vector_eval<<<1,META_SIZE>>>(meta_idx, chunk_table, chunk_len, CHUNK_SIZE); // summera alla paralellt och uppdatera chunk

            //cudaDeviceSynchronize();

            // generate random numbers
            generate_rand<<<1,N>>>(devStates);

            //cudaDeviceSynchronize();

            printf("random=%f\n", rand_table[0]);
            printf("random=%f\n", rand_table[1]);
            printf("random=%f\n", rand_table[2]);
            printf("random=%f\n", rand_table[3]);
            printf("random=%f\n\n", rand_table[4]);

            //printf("LSQ=%f for chunk_idx=%i\n", lsq, chunk_idx);
        }
    }

    free(devStates);
    free(error_table);

    // analyse x0, y0, scale_r, scale_y, theta => interpolate & extrapolate
    //
    // write results to result_table (maybe switch back to host for loading it?!)
}

// ========================================================== HOST LAUNCH ==================================================================
void host_launch(chunk_record *chunk_table, int chunk_len, mag_record *mag_table, int mag_len, meta_record *meta_table, int meta_len) {
    printf("Host Launch:\n");
    cuda_main<<<1,1>>>(chunk_table, chunk_len, mag_table, mag_len, meta_table, meta_len); // Kernel running on only one single core
}
