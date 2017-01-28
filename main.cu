

#include <stdio.h>
#include <stdint.h>
#include <sqlite3.h>

#include "struct.h"
#include "makros.h"

#include "histogram.h"
#include "gps2arc.h"
#include "kinetics.h"



int main(int argc, char *argv[]) {
    char buffer_Z[100];  // string buffer

    int mag_len;
    struct mag_record *mag_table = NULL; // mag_table is of length kinetics_len

    int arc_len;
    struct arc_record *arc_table = NULL; // arc_table is of length arc_len

    fprintf(stderr,"\n   *** OSAN POSITIONING 2017 v0.01 ***\n\n");

    if (argc != 2) {
       fprintf(stderr,"Usage:\n");
       fprintf(stderr,"normalize <database>\n\n");
       exit(1);
    }

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

    gps2arc_record(buffer_Z, &arc_table, &arc_len, &mag_table, &mag_len);   // <-------------------------------- krashar

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


    // bin=5; range=200 => (-1000,1000); cut_off=2
    histogram(&arc_table, &arc_len, &mag_table, &mag_len, 5, 200, 2);

    free(mag_table);

    free(arc_table);

    return 0;

}
