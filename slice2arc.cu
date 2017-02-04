
#include <stdio.h>
#include <stdint.h>
#include <sqlite3.h>

#include "struct.h"
#include "makros.h"



// slice2arc_record calculate the pointer index into mag_table and defines all arcs, returned in chunk_table
// chunk_size is the number of elements in each arc fitted to WARP_SIZE for ultimate CUDA performance
int slice2arc_record(chunk_record **chunk_table, int *chunk_len, mag_record *mag_table, int mag_len, int chunk_size) {

    //*chunk_size = (wished_size/WARP_SIZE) * WARP_SIZE; // approximately the same size as chunk_size
    *chunk_len =  mag_len/chunk_size + 1;  //

    //printf("chunk_size=%i, chunk_len=%i, mag_len=%i\n\n", chunk_size, *chunk_len, mag_len);

    chunk_record *new_table = (chunk_record*) malloc((*chunk_len) * sizeof(chunk_record));

    int left_mag_idx;
    int right_mag_idx;

    for (int arc_idx=0; arc_idx < (*chunk_len) - 1; arc_idx++) {
        left_mag_idx = arc_idx * chunk_size;
        right_mag_idx = arc_idx * chunk_size + chunk_size - 1;

        new_table[arc_idx].left_mag_idx  = left_mag_idx;
        new_table[arc_idx].right_mag_idx = right_mag_idx;

        //new_table[arc_idx].left_seq_id = mag_table[left_mag_idx].seq_id;
        //new_table[arc_idx].right_seq_id = mag_table[right_mag_idx].seq_id;

        //printf("arc_idx=%i, left_mag_idx=%i, right_mag_idx=%i\n", arc_idx, left_mag_idx, right_mag_idx);
    }

    // create last arc by hand here
    if (right_mag_idx < mag_len) {
        int arc_idx = (*chunk_len) - 1;
        left_mag_idx = arc_idx * chunk_size;
        right_mag_idx = mag_len -1;

        new_table[arc_idx].left_mag_idx  = left_mag_idx;
        new_table[arc_idx].right_mag_idx = right_mag_idx;

        //new_table[arc_idx].left_seq_id = mag_table[left_mag_idx].seq_id;
        //new_table[arc_idx].right_seq_id = mag_table[right_mag_idx].seq_id;

        //printf("arc_idx=%i, left_mag_idx=%i, right_mag_idx=%i\n", arc_idx, left_mag_idx, right_mag_idx);
    }

    *chunk_table = new_table;

    return 0;
}
