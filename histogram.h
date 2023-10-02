/*
 *     histogram.h
 * 
 *          Description:  Rectangular to polar normalization, in CUDA-C
 *          Author:       Hans-Henrik Fuxelius
 *          Date:         Uppsala, 2017-02-17
 *          License:      MIT
 *          Version:      1.0
 */

int histogram(chunk_record *chunk_table, int *chunk_len, mag_record *mag_table, int *mag_len, int chunk_idx,
                                                                                          int bin, int range, int cut_off);
