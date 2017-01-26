

#include <stdio.h>
#include <stdint.h>
#include <sqlite3.h>

#include "struct.h"
#include "gps2arc.h"
#include "kinetics.h"
#include "makros.h"

// kompilera
// nvcc main.cu gps2arc.cu kinetics.cu -o normalize -lsqlite3

int main(int argc, char *argv[]) {
    char buffer_Z[100];  // string buffer

    int kinetics_len;
    struct mag_record *mag_table = NULL; // mag_table is of length kinetics_len

    int arc_len;
    struct arc_record *arc_table = NULL; // arc_table is of length arc_len

    fprintf(stderr,"\n   *** OSAN POSITIONING 2017 v0.01 ***\n\n");

    if (argc != 2) {
       fprintf(stderr,"Usage:\n");
       fprintf(stderr,"Normalize <filename>\n\n");
       exit(1);
    }

    sprintf(buffer_Z,"%s",argv[1]);   // *.sqlite3

    kinetics2record(buffer_Z, &mag_table, &kinetics_len);

    // for (int row_cnt=0; row_cnt<kinetics_len; row_cnt++) {
    //     printf(">> %u | ",mag_table[row_cnt].seq_id);
    //     printf("%f | ",mag_table[row_cnt].mxt);
    //     printf("%f\n",mag_table[row_cnt].myt);
    // }

    gps2arc_record(buffer_Z, &arc_table, &arc_len, &mag_table, &kinetics_len);


    puts("Can not hold me back");

    for (int rec_cnt=0; rec_cnt<arc_len; rec_cnt++) {
        printf("++->%u | %u | %u | ",rec_cnt, arc_table[rec_cnt].left_seq_id, arc_table[rec_cnt].right_seq_id);
        printf("%u | %u \n", arc_table[rec_cnt].left_mag_idx, arc_table[rec_cnt].right_mag_idx);
    }

    //printf("length fux %u \n\n",kinetics_len);

    return 0;

}
