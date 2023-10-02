/*
 *     x0y0_histogram.h
 * 
 *          Description:  Rectangular to polar normalization, in CUDA-C
 *          Author:       Hans-Henrik Fuxelius
 *          Date:         Uppsala, 2017-02-17
 *          License:      MIT
 *          Version:      1.0
 */

int x0y0_histogram(chunk_record *chunk_table, int chunk_len, int bin, int range, int cut_off);
