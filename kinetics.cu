
#include <stdio.h>
#include <stdint.h>
#include <sqlite3.h>

#include "struct.h"
#include "makros.h"

// Load sqlite database table kinetics to record in memory
int kinetics2record(char *db_name, struct mag_record **mag_table, int *kinetics_len) {
    sqlite3 *conn;
    sqlite3_stmt *res;
    int error = 0;
    //const char *errMSG;
    const char *tail;

    //int kinetics_len;          // The length of the kinetics table
    int row_cnt;

    error = sqlite3_open(db_name, &conn);
    if (error) {
        puts("Can not open database");
        exit(0);
    }

    error = sqlite3_prepare_v2(conn,"SELECT count(seq_id) FROM kinetics",1000, &res, &tail);

    if (error != SQLITE_OK) {
        puts("We did not get any data!");
        exit(0);
    }

    while (sqlite3_step(res) == SQLITE_ROW) {
        *kinetics_len = sqlite3_column_int(res, 0);
        // printf("%u|\n", *kinetics_len);
    }

    struct mag_record *new_table = (struct mag_record*) malloc((*kinetics_len) * sizeof(struct mag_record));

    error = sqlite3_prepare_v2(conn,"SELECT seq_id, mxt, myt FROM kinetics ORDER BY seq_id",1000, &res, &tail);

    if (error != SQLITE_OK) {
        puts("We did not get any data!");
        exit(0);
    }

    row_cnt = 0;
    while (sqlite3_step(res) == SQLITE_ROW) {
        //printf("%u | ", sqlite3_column_int(res, 0));
        //printf("%f | ",  (float)sqlite3_column_double(res, 1));  // dessa bör castas till CUDA single precision = float????
        //printf("%f\n", (float)sqlite3_column_double(res, 2)); // dessa bör castas till CUDA single precision

        new_table[row_cnt].seq_id = sqlite3_column_int(res, 0);
        new_table[row_cnt].mxt = sqlite3_column_double(res, 1);
        new_table[row_cnt].myt = sqlite3_column_double(res, 2);
        new_table[row_cnt].outlier = 0;

        // printf("%u | ",new_table[row_cnt].seq_id);
        // printf("%f | ",new_table[row_cnt].mxt);
        // printf("%f\n",new_table[row_cnt].myt);

        row_cnt++;
    }

    // puts("==========================");
    //
    // printf("We received %d records.\n", *kinetics_len);

    sqlite3_finalize(res);

    sqlite3_close(conn);

    *mag_table = new_table;

    return 0;
}
