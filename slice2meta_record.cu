/*
 *     slice2meta_record.cu
 * 
 *          Description:  Rectangular to polar normalization, in CUDA-C
 *          Author:       Hans-Henrik Fuxelius
 *          Date:         Uppsala, 2017-02-17
 *          License:      MIT
 *          Version:      1.0
 */

#include <stdio.h>
#include <stdint.h>
#include "struct.h"
#include "makros.h"

int slice2meta_record(meta_record **meta_table, int *meta_len, int chunk_len, int meta_size) {

    if (chunk_len%meta_size == 0) {
        *meta_len = chunk_len / meta_size;
    }
    else {
        *meta_len = chunk_len / meta_size + 1;
    }

    printf("\nchunk_len=%i, META_SIZE=%i, meta_len=%i\n", chunk_len, meta_size, *meta_len);

    meta_record *new_table= (meta_record*) malloc((*meta_len) * sizeof(meta_record));

    int left_meta_idx  = 0;
    int right_meta_idx = 0;

    for (int meta_idx=0; meta_idx<(*meta_len) - 1; meta_idx++) {
      left_meta_idx = meta_idx * meta_size;
      right_meta_idx = meta_idx * meta_size + meta_size - 1;

        new_table[meta_idx].left_chunk_idx = left_meta_idx;
        new_table[meta_idx].right_chunk_idx = right_meta_idx;
        printf("1>meta_idx=%i, left_chunk_idx=%i, right_chunk_idx=%i\n", meta_idx, left_meta_idx, right_meta_idx);
    }
    if (right_meta_idx < chunk_len) {
        int meta_idx = *meta_len - 1;
        left_meta_idx = meta_idx * meta_size;
        right_meta_idx = chunk_len -1;

        new_table[meta_idx].left_chunk_idx  = left_meta_idx;
        new_table[meta_idx].right_chunk_idx = right_meta_idx;
        printf("2>meta_idx=%i, left_chunk_idx=%i, right_chunk_idx=%i\n", meta_idx, left_meta_idx, right_meta_idx);
    }

    *meta_table = new_table;

    return 0;
}
