/*
 *     cuda_magix.h
 * 
 *          Description:  Rectangular to polar normalization, in CUDA-C
 *          Author:       Hans-Henrik Fuxelius
 *          Date:         Uppsala, 2017-02-17
 *          License:      MIT
 *          Version:      1.0
 */

void host_launch(chunk_record *chunk_table, int chunk_len, mag_record *mag_table, int mag_len, meta_record *meta_table, int meta_len);
