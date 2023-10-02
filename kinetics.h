/*
 *     kinetics.h
 * 
 *          Description:  Rectangular to polar normalization, in CUDA-C
 *          Author:       Hans-Henrik Fuxelius
 *          Date:         Uppsala, 2017-02-17
 *          License:      MIT
 *          Version:      1.0
 */

int kinetics2record(char *db_name, mag_record **mag_table, result_record **result_table, int *kinetics_len);
