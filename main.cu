#include <cuda_runtime.h>
#include "common/common.h"

#include <stdio.h>
#include <stdint.h>
#include <sqlite3.h>

#include "struct.h"
#include "makros.h"

#include "histogram.h"
#include "gps2arc.h"
#include "kinetics.h"

#include "point_square.h"
#include "device_info.h"

// plot_raw_filtered print all raw data between left_arc_idx and right_arc_idx with outliers excluded.
// used for creating plots to R to debug and analysis. to run from BASH and inside main
// Must first run:
// 1) kinetics2record - the kinetics file to datastructure magtable
// 2) gps2arc_record  - Creates arcs pointing into mag_table
// 3) histogram       - cut off outliers and mark it in mag_table[idx].outlier
void plot_raw_filtered(struct arc_record **arc_table, int *arc_len, struct mag_record **mag_table, int *mag_len, int left_arc_idx, int right_arc_idx) {
    float mxt;
    float myt;

    puts("mxt, myt");

    for (int mag_idx = (*arc_table)[left_arc_idx].left_mag_idx; mag_idx <= (*arc_table)[right_arc_idx].right_mag_idx; mag_idx++) {
        mxt = (*mag_table)[mag_idx].mxt;
        myt = (*mag_table)[mag_idx].myt;

        if (!(*mag_table)[mag_idx].outlier) {
            printf("%f,%f\n", mxt, myt);
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


    char buffer_Z[100];  // string buffer

    int mag_len;
    struct mag_record *mag_table = NULL; // mag_table is of length kinetics_len

    int arc_len;
    struct arc_record *arc_table = NULL; // arc_table is of length arc_len



    // skriv ut en text här hur man refererar till programmet om man publicerar
    // vetenskapliga resultat. OSAN POSITIONING; H-H. Fuxelius


    sprintf(buffer_Z,"%s",argv[1]);   // *.sqlite3



    // Reads in magnetometer data from database (table kinetics) to magtable and returns
    // table length kinetics_len
    kinetics2record(buffer_Z, &mag_table, &mag_len);


    // for (int row_cnt=0; row_cnt<kinetics_len; row_cnt++) {
    //     printf(">> %u | ",mag_table[row_cnt].seq_id);
    //     printf("%f | ",mag_table[row_cnt].mxt);
    //     printf("%f\n",mag_table[row_cnt].myt);
    // }


    // Creates an arc_table which is a partitioning of mxt, myt between gps positions
    // arc_table[].left_mag_idx and arc_table[].right_mag_idx points out each arcs border
    // These arcs partition the entire mag_table

    gps2arc_record(buffer_Z, &arc_table, &arc_len, &mag_table, &mag_len);

    #ifdef DEBUG_INFO_1
        // Proves that the pointers are correct in arc_table
        puts("seqid | seqid | mag_idx | mag_idx | seqid | seqid");

        int left_idx, right_idx;
        for (int rec_cnt=0; rec_cnt<arc_len; rec_cnt++) {
            printf("++-> %u | %u | %u |",rec_cnt, arc_table[rec_cnt].left_seq_id, arc_table[rec_cnt].right_seq_id);
            printf(" %u | %u | ", arc_table[rec_cnt].left_mag_idx, arc_table[rec_cnt].right_mag_idx);

            left_idx = arc_table[rec_cnt].left_mag_idx;
            right_idx = arc_table[rec_cnt].right_mag_idx;

            printf("%u | %u \n", mag_table[left_idx].seq_id, mag_table[right_idx].seq_id);

        }
    #endif





    //------------------------------- experiment -------------------------------

    // 1) this code operates on all arcs, one at a time
    for (int arc_idx = 0; arc_idx < arc_len; arc_idx++) {

        // bin=5; range=200 => (-1000,1000); cut_off<5
        //histogram(&arc_table, &arc_len, &mag_table, &mag_len, 0, 0, 5, 100, 5);
    }

    // int left_arc_idx = 0;
    // int right_arc_idx = 18;



    int left_arc_idx = 0;
    int right_arc_idx = arc_len-1;

    printf("arc_len=%i", arc_len-1);

    // 2) all mag pointsfrom arc 0-18
    histogram(&arc_table, &arc_len, &mag_table, &mag_len, left_arc_idx, right_arc_idx, 5, 100, 5);

    // ---> funktion för att skriva ut antalet arcs i hela databasen
    // ---> skriv funktion för att plotta i R (seq_id,mxt,myt)
    // ---> plotta alla origo från alla arcs och se hur dom hamnar:
    //plot_raw_filtered(&arc_table, &arc_len, &mag_table, &mag_len, left_arc_idx, right_arc_idx);

    // använd -v, verbose mode för att skriva ut ALL debug info i en katalog

    // 3) all mag points from in arc 2 is negative
    //histogram(&arc_table, &arc_len, &mag_table, &mag_len, 2, 2, 5, 100, 3);

    // 4) all mag points from in arc 2 is negative
    //histogram(&arc_table, &arc_len, &mag_table, &mag_len, 17, 17, 5, 100, 3);

    //--------------------------------------------------------------------------



    // tested model
    float mxt      =  200;
    float myt      =  -30;
    float x0       =   16;
    float y0       =  -124;
    float scale_r  = 0.0041;
    float scale_y  = 1.045; // 1.045
    float rotate   = 0.0;

    float normalized_x;  // Return value
    float normalized_y;  // Return value
    float quad_error;    // Return value

    for (int mag_idx = arc_table[left_arc_idx].left_mag_idx; mag_idx <= arc_table[right_arc_idx].right_mag_idx; mag_idx++) {
        mxt = mag_table[mag_idx].mxt;
        myt = mag_table[mag_idx].myt;

        if (!mag_table[mag_idx].outlier) {
            point_square(mxt, myt, x0, y0, scale_r, scale_y, rotate, &normalized_x, &normalized_y, &quad_error);
            //printf("%f,%f\n", normalized_x, normalized_y);
        }
    }




    free(mag_table);
    free(arc_table);

    return 0;

}
