#include <cuda_runtime.h>
#include "common/common.h"

#include <stdio.h>
#include <stdint.h>
#include <sqlite3.h>

#include "struct.h"
#include "makros.h"

#include "histogram.h"
#include "slice2chunk.h"
#include "slice2meta_record.h"
#include "kinetics.h"

#include "cuda_magix.h"
#include "device_info.h"
#include "polar.h"
#include "x0y0_histogram.h"


// plot_raw_filtered print all raw data between left_chunk_idx and right_chunk_idx with outliers excluded.
// used for creating plots to R to debug and analysis. to run from BASH and inside main
// Must first run:
// 1) kinetics2record - the kinetics file to datastructure magtable
// 2) gps2chunk_record  - Creates chunks pointing into mag_table
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
   // skriv ut en text här hur man refererar till programmet om man publicerar
   // vetenskapliga resultat. OSAN POSITIONING; H-H. Fuxelius
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
    sprintf(buffer_Z,"%s",argv[1]);   // *.sqlite3

    int mag_len;
    mag_record *mag_table = NULL; // potentionally very big; only load in the end for transfering results back to host

    // This table keeps the returned results on host
    // initialize to mfv and rho to zeros when filling in seq_id
    result_record *result_table = NULL;   // identical length to mag_record(mag_len);byggs och returneras i kinetics2record

    int chunk_len;                        // number of chunks that divides entire mag table in CHUNK_SIZE
    chunk_record *chunk_table = NULL;

    int meta_len;                         // number of meta_records that divides entire chunk_table in META_SIZE pieces
    meta_record *meta_table = NULL;

    // Reads in magnetometer data from database (table kinetics) to magtable and returns
    // table length kinetics_len
    kinetics2record(buffer_Z, &mag_table, &result_table, &mag_len);

    // Creates an chunk_table which is a partitioning of mxt, myt of a chunk_size
    // chunk_table[].left_mag_idx and chunk_table[].right_mag_idx points out each chunk border
    // These chunk partition the entire mag_table
    // Should be a multiple of BLOCK_SIZE (now set to 256); CUDA stuff (= 20 minuter)
    slice2chunk_record(&chunk_table, &chunk_len, mag_table, mag_len, CHUNK_SIZE);

    // keep the legth of a chunk in a metatable
    slice2meta_record(&meta_table, &meta_len, chunk_len, META_SIZE);

    #ifdef DEBUG_INFO_1
        // Proves that the pointers are correct in chunk_table
        puts("seqid | seqid | mag_idx | mag_idx | seqid | seqid");

        int left_idx, right_idx;
        for (int rec_cnt=0; rec_cnt<chunk_len; rec_cnt++) {
            printf("++-> %u | %u | %u |",rec_cnt, chunk_table[rec_cnt].left_mag_idx, chunk_table[rec_cnt].right_mag_idx);
            //(" %u | %u | ", chunk_table[rec_cnt].left_mag_idx, chunk_table[rec_cnt].right_mag_idx);

            left_idx = chunk_table[rec_cnt].left_mag_idx;
            right_idx = chunk_table[rec_cnt].right_mag_idx;

            printf("%u | %u \n", mag_table[left_idx].mxt, mag_table[right_idx].myt);

        }
    #endif


    // Run histogram on each chunk, and store its results in chunk_table
    int bin   = 5;      // size of each bin
    int range = 100;    // => (-500,+500)
    int cut_off = 5;    // cut off on both sides of origo where < cut_off in a bin
    for (int chunk_idx=0; chunk_idx<chunk_len; chunk_idx++) {
        histogram(chunk_table, &chunk_len, mag_table, &mag_len, chunk_idx, bin, range, cut_off);
    }

    // print out the info in all chunks
    #ifdef DEBUG_INFO_1
        for (int chunk_idx=0; chunk_idx<chunk_len; chunk_idx++) {
            printf("chunk_idx %i\n", chunk_idx);
            printf("left_mag_idx %i\n", chunk_table[chunk_idx].left_mag_idx);
            printf("right_mag_idx %i\n", chunk_table[chunk_idx].right_mag_idx);
            printf("x0 %f\n", chunk_table[chunk_idx].x0);
            printf("y0 %f\n", chunk_table[chunk_idx].y0);
            printf("scale_r %f\n", chunk_table[chunk_idx].scale_r);
            printf("scale_y_axis %f\n", chunk_table[chunk_idx].scale_y_axis);
            printf("theta %f\n", chunk_table[chunk_idx].theta);
            printf("disable %i\n\n", chunk_table[chunk_idx].outlier);
        }
    #endif
    //--------------------------------------------------------------------------

    // ============================================ CUDA START 1 ============================================


    mag_record *d_mag_table;
    size_t mag_bytes = mag_len * sizeof(mag_record);
    cudaMalloc((void **)&d_mag_table, mag_bytes);
    cudaMemcpy(d_mag_table, mag_table, mag_bytes, cudaMemcpyHostToDevice);

    chunk_record *d_chunk_table;
    size_t chunk_bytes = chunk_len * sizeof(chunk_record);
    cudaMalloc((void **)&d_chunk_table, chunk_bytes);
    cudaMemcpy(d_chunk_table, chunk_table, chunk_bytes, cudaMemcpyHostToDevice);

    meta_record *d_meta_table;
    size_t meta_bytes = meta_len * sizeof(meta_record);
    cudaMalloc((void **)&d_meta_table, meta_bytes);
    cudaMemcpy(d_meta_table, meta_table, meta_bytes, cudaMemcpyHostToDevice);

    result_record *d_result_table;
    size_t result_bytes = mag_len * sizeof(result_record);
    cudaMalloc((void **)&d_result_table, result_bytes);
    cudaMemcpy(d_result_table, result_table, result_bytes, cudaMemcpyHostToDevice);

    host_launch(d_chunk_table, chunk_len, d_mag_table, mag_len, d_meta_table, meta_len); // <--------- MAIN() CUDA CALL (ONLY ONE THREAD)

    CHECK(cudaDeviceSynchronize());

    cudaMemcpy(chunk_table, d_chunk_table, chunk_bytes, cudaMemcpyDeviceToHost); // Get it back here, NOW!!!!





    // ============================================ CUDA END ============================================



    // for (int i=0; i<mag_len; i++) {
    //   printf("cuda,%f,%f\n", result_table[i].mfv, result_table[i].rho);
    // }

    // for (int i=0; i<chunk_len; i++) {
    //   printf("chunk_idx \t%i\n", i);
    //   printf("x0 \t\t%f\n", chunk_table[i].x0);
    //   printf("y0 \t\t%f\n", chunk_table[i].y0);
    //   printf("scale_r \t%f\n", chunk_table[i].scale_r);
    //   printf("scale_y_axis \t%f\n", chunk_table[i].scale_y_axis);
    //   printf("theta \t\t%f\n\n", chunk_table[i].theta);
    //   printf("iter_cnt \t%f\n\n", chunk_table[i].iter_cnt);
    // }

    printf("chunk_idx, x0, y0\n");
    for (int i=0; i<chunk_len; i++) {
      printf("%i, %f, %f\n", i, chunk_table[i].x0, chunk_table[i].y0);
    }


    // ============================================ X0Y0 HISTOGRAM ============================================

    // since each value represent CHUNK_SIZE (1024) mag values the number of x0y0 will be very low and it is
    // impossible to make any statistics:
    // Therefore: set cut_off = 1 for low lengths on chunk_table

    // x0y0_bin x0y0_cut_off<------------------------------------------------------------------------------------- måste gå att ändra via argv som input värde

    int x0y0_range = 100;    // => (-500,+500)
    int x0y0_bin;
    int x0y0_cut_off;

    if (chunk_len < 100) {
        x0y0_bin   = 10;      // size of each bin
        x0y0_cut_off = 5;    // cut off on both sides of origo where < cut_off in a bin     <------------------------------ MUST BE 0 at short tests in time
    }
    else {
        x0y0_bin   = 5;      // size of each bin
        x0y0_cut_off = 5;    // cut off on both sides of origo where < cut_off in a bin     <------------------------------ MUST BE 0 at short tests in time
    }

    // 1) calculate the geometric mid-point of non-outliers; x0, y0, scale-r, scale-y and theta (theta can be turned 2*PI - a problem)
    // 2) update all entries in chunk_table
    x0y0_histogram(chunk_table, chunk_len, x0y0_bin, x0y0_range, x0y0_cut_off);


    // ============================================ CUDA START 2 ============================================


    dim3 grid((mag_len + BLOCK_SIZE - 1)/ BLOCK_SIZE, 1);
    rec2polar<<<grid,BLOCK_SIZE>>>(d_result_table, d_chunk_table, chunk_len, d_mag_table, mag_len, CHUNK_SIZE); // record_len = mag_len

    CHECK(cudaDeviceSynchronize()); // behövs denna här ??

    cudaMemcpy(result_table, d_result_table, result_bytes, cudaMemcpyDeviceToHost); // Get it back here, NOW!!!!

    printf("mfv, rho\n");
    for (int i=0; i<mag_len; i++ ) {
        float mfv = result_table[i].mfv;
        float rho = result_table[i].rho;
        printf("%f, %f\n", mfv, rho);
    }



    // 2) skriv tillbaka till databasen



    // ============================================ CUDA END ============================================

    // free device global memory
    CHECK(cudaFree(d_mag_table));     // denna hanterar free på både host och device
    CHECK(cudaFree(d_chunk_table));   // denna hanterar free på både host och device
    CHECK(cudaFree(d_meta_table));    // denna hanterar free på både host och device

    CHECK(cudaGetLastError());

    // reset device
    CHECK(cudaDeviceReset());

    free(mag_table);
    free(chunk_table);
    free(meta_table);
    free(result_table);

    return 0;
}
