#include <cuda_runtime.h>
#include "common/common.h"

#include <stdio.h>
#include <stdint.h>
#include <sqlite3.h>

#include "struct.h"
#include "makros.h"

#include "histogram.h"
#include "slice2arc.h"
#include "kinetics.h"

#include "point_square.h"
#include "device_info.h"



// plot_raw_filtered print all raw data between left_chunk_idx and right_chunk_idx with outliers excluded.
// used for creating plots to R to debug and analysis. to run from BASH and inside main
// Must first run:
// 1) kinetics2record - the kinetics file to datastructure magtable
// 2) gps2chunk_record  - Creates arcs pointing into mag_table
// 3) histogram       - cut off outliers and mark it in mag_table[idx].outlier
void plot_raw_filtered(chunk_record *chunk_table, int *chunk_len, mag_record *mag_table, int *mag_len, int left_chunk_idx, int right_chunk_idx) {
    short mxt;
    short myt;

    puts("mxt, myt");

    for (int mag_idx = chunk_table[left_chunk_idx].left_mag_idx; mag_idx <= chunk_table[right_chunk_idx].right_mag_idx; mag_idx++) {
        mxt = mag_table[mag_idx].mxt;
        myt = mag_table[mag_idx].myt;

        if (!mag_table[mag_idx].disable) {
            printf("%i,%i\n", mxt, myt);
        }
    }
  }



int main(int argc, char *argv[]) {

    fprintf(stderr,"\n\n                               *** OSAN POSITIONING 2017 v0.01 ***\n\n");

    print_device_info();    // Print out all relevant CUDA device information

    if (argc != 2) {
       fprintf(stderr,"Usage:\n");
       fprintf(stderr,"normalize <database>\n\n");
       exit(1);
    }

    // set up CUDA device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    char buffer_Z[100];  // string buffer

    int mag_len;
    //struct mag_record *mag_table = NULL; // mag_table is of length kinetics_len
    mag_record *mag_table = NULL;

    int chunk_len;
    //struct  *chunk_table = NULL; // chunk_table is of length chunk_len
    chunk_record *chunk_table = NULL;


    // skriv ut en text här hur man refererar till programmet om man publicerar
    // vetenskapliga resultat. OSAN POSITIONING; H-H. Fuxelius

    sprintf(buffer_Z,"%s",argv[1]);   // *.sqlite3

    // Reads in magnetometer data from database (table kinetics) to magtable and returns
    // table length kinetics_len
    kinetics2record(buffer_Z, &mag_table, &mag_len);

    // Creates an chunk_table which is a partitioning of mxt, myt of a chunk_size
    // chunk_table[].left_mag_idx and chunk_table[].right_mag_idx points out each arcs border
    // These arcs partition the entire mag_table
    int arc_size = 1024;           // Should be a multiple of BLOCK_SIZE=256; CUDA stuff
    slice2arc_record(&chunk_table, &chunk_len, mag_table, mag_len, arc_size);


    #ifdef DEBUG_INFO_1
        // Proves that the pointers are correct in chunk_table
        puts("seqid | seqid | mag_idx | mag_idx | seqid | seqid");

        int left_idx, right_idx;
        for (int rec_cnt=0; rec_cnt<chunk_len; rec_cnt++) {
            printf("++-> %u | %u | %u |",rec_cnt, chunk_table[rec_cnt].left_seq_id, chunk_table[rec_cnt].right_seq_id);
            printf(" %u | %u | ", chunk_table[rec_cnt].left_mag_idx, chunk_table[rec_cnt].right_mag_idx);

            left_idx = chunk_table[rec_cnt].left_mag_idx;
            right_idx = chunk_table[rec_cnt].right_mag_idx;

            printf("%u | %u \n", mag_table[left_idx].seq_id, mag_table[right_idx].seq_id);

        }
    #endif


    // Run histogram on each arc, and store its results in chunk_table
    int bin   = 5;
    int range = 100; // => (-500,+500)
    int cut_off = 5;
    for (int arc_idx=0; arc_idx<chunk_len; arc_idx++) {
        histogram(chunk_table, &chunk_len, mag_table, &mag_len, arc_idx, bin, range, cut_off);
    }

    // print out the info in all arcs
    #ifdef DEBUG_INFO_1
        for (int arc_idx=0; arc_idx<chunk_len; arc_idx++) {
            printf("arc_idx %i\n", arc_idx);
            printf("left_mag_idx %i\n", chunk_table[arc_idx].left_mag_idx);
            printf("right_mag_idx %i\n", chunk_table[arc_idx].right_mag_idx);
            printf("x0 %f\n", chunk_table[arc_idx].x0);
            printf("y0 %f\n", chunk_table[arc_idx].y0);
            printf("scale_r %f\n", chunk_table[arc_idx].scale_r);
            printf("scale_y_axis %f\n", chunk_table[arc_idx].scale_y_axis);
            printf("theta %f\n", chunk_table[arc_idx].theta);
            printf("disable %i\n\n", chunk_table[arc_idx].disable);
        }
    #endif
    //--------------------------------------------------------------------------

    int arc_idx = 0;

    // tested model
    short mxt      =  200;
    short myt      =  -30;
    float x0       =   16;
    float y0       =  -124;
    float scale_r  = 0.0041;
    float scale_y  = 1.045; // 1.045
    float rotate   = 0.0;

    float normalized_x;  // Return value
    float normalized_y;  // Return value
    float quad_error;    // Return value

    for (int mag_idx = chunk_table[arc_idx].left_mag_idx; mag_idx <= chunk_table[arc_idx].right_mag_idx; mag_idx++) {
        mxt = mag_table[mag_idx].mxt;
        myt = mag_table[mag_idx].myt;

        if (!mag_table[mag_idx].disable) {
            point_square(mxt, myt, x0, y0, scale_r, scale_y, rotate, &normalized_x, &normalized_y, &quad_error);
            //printf("%f,%f\n", normalized_x, normalized_y);
        }
    }


      //point_square_GPU(&chunk_table, chunk_len, &mag_table, mag_len, arc_size);


    // malloc device global memory
    mag_record *d_mag_table;
    size_t mag_bytes = mag_len * sizeof(mag_record);
    CHECK(cudaMalloc((void **)&d_mag_table, mag_bytes));
    CHECK(cudaMemcpy(d_mag_table, mag_table, mag_bytes, cudaMemcpyHostToDevice));

    chunk_record *d_chunk_table;
    size_t arc_bytes = chunk_len * sizeof(chunk_record);
    CHECK(cudaMalloc((void **)&d_chunk_table, arc_bytes));
    CHECK(cudaMemcpy(d_chunk_table, chunk_table, arc_bytes, cudaMemcpyHostToDevice));


    // invoke kernel at host side
    int dimx = BLOCK_SIZE; // < 1024
    dim3 block(dimx, 1);
    dim3 grid(mag_len / block.x + 1, 1);
    //dim3 grid(800, 1);

    //point_square_GPU(&chunk_table, chunk_len, &mag_table, mag_len, arc_size);

    point_square_GPU<<<grid, block>>>(d_chunk_table, chunk_len, d_mag_table, mag_len, arc_size);

    CHECK(cudaDeviceSynchronize());

    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(mag_table, d_mag_table, mag_bytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(chunk_table, d_chunk_table, arc_bytes, cudaMemcpyDeviceToHost));

    // free device global memory
    CHECK(cudaFree(d_mag_table));
    CHECK(cudaFree(d_chunk_table));

    // reset device
    CHECK(cudaDeviceReset());

    free(mag_table);
    free(chunk_table);

    return 0;

}
