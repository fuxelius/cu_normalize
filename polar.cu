
#include <cuda_runtime.h>
#include "common/common.h"
#include <stdio.h>
#include <stdint.h>
#include <curand.h>
#include <curand_kernel.h>
#include "math.h"
#include "struct.h"
#include "makros.h"
//#include "math_constants.h"

#define PI ((float)3.141592653)

__global__ void rec2polar(result_record *result_table, chunk_record *chunk_table, int chunk_len, mag_record *mag_table, int mag_len, int chunk_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // position inside meta_idx
    int chunk_idx =  idx / chunk_size;
    int mag_idx = idx;

    //printf("Point Square: meta_idx=%i idx=%i error_idx=%i chunk_idx=%i mag_idx=%i\n", meta_idx, idx, error_idx, chunk_idx, mag_idx);

    // cut out all other created threads based on threadIdx.x, otherwise they WILL write out of bound -- and krashes :(
    if (mag_idx < mag_len && chunk_idx < chunk_len) {

        // mag_table
        short mxt = mag_table[mag_idx].mxt;
        short myt = mag_table[mag_idx].myt;

        // // chunk_table, temporary values
        float x0            = chunk_table[chunk_idx].x0;
        float y0            = chunk_table[chunk_idx].y0;
        float scale_r       = chunk_table[chunk_idx].scale_r;
        float scale_y_axis  = chunk_table[chunk_idx].scale_y_axis;
        float theta         = chunk_table[chunk_idx].theta;

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

        float normalized_x = rotate_x * scale_r;                         // Returns x,y normalized to unit circle
        float normalized_y = rotate_y * scale_r;

        //printf("normalized ,%f,%f\n", normalized_x, normalized_y);

        // https://en.wikipedia.org/wiki/Polar_coordinate_system
        float mfv = sqrtf(powf(normalized_x,2) + powf(normalized_y,2));
        float rho = atan2(normalized_y,normalized_x); // turn x to NORTH by adding PI/2

        if (rho < 0) {
            rho = rho + 2*PI;
        }

        //if (!mag_table[mag_idx].disable) { // <------------------------------------------------------------ temporary only for printouts
            result_table[mag_idx].mfv = mfv;
            result_table[mag_idx].rho = rho;
        //}

     }
}
