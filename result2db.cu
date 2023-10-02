/*
 *     result2db.cu
 * 
 *          Description:  Rectangular to polar normalization, in CUDA-C
 *          Author:       Hans-Henrik Fuxelius
 *          Date:         Uppsala, 2017-02-17
 *          License:      MIT
 *          Version:      1.0
 */

#include <stdio.h>
#include <stdint.h>
#include <sqlite3.h>

#include "struct.h"
#include "makros.h"

// Load sqlite database table kinetics to record in memory
int result2db(char *db_name, result_record *result_table, int result_len) {
    // ----------------------- SQLITE VARIABLES
    int retval = 0;                // Create an int variable for storing the return code for each call
    char query[500];               // holder for sql queries

    sqlite3 *conn;
    //sqlite3_stmt *res = NULL;
    int error = 0;
    //const char *errMSG;
    //const char *tail;
    //
    retval = sqlite3_open(db_name, &conn);
    if (retval) {
        puts("Can not open database");
        exit(0);
    }

    // // --------------------------- OPTIMIZATIONS ---------------------------
    //
    // Optimize for maximum INSERT speed
    sprintf(query, "PRAGMA synchronous=OFF");
    retval = retval = sqlite3_exec(conn, query, 0, 0, 0);
    if (retval) {
        puts("ERROR: PRAGMA synchronous=OFF");
        exit(0);
    }

    // Write to memory, then to disk
    retval = sqlite3_exec(conn, "BEGIN TRANSACTION", NULL, NULL, NULL);
    if (retval) {
        puts("ERROR: BEGIN TRANSACTION");
        exit(0);
    }

    //printf("seq_id, mfv, rho\n");
    for (int i=0; i<result_len; i++ ) {
        int seq_id = result_table[i].seq_id;
        float mfv  = result_table[i].mfv;
        float rho  = result_table[i].rho;
        //printf("%i, %f, %f\n",seq_id, mfv, rho);

        sprintf(query, "UPDATE kinetics SET mfv=%f, heading=%f WHERE seq_id = %i", mfv, rho, seq_id);
        retval = sqlite3_exec(conn, query ,0, 0, 0);
        if (retval) {
            puts("ERROR: UPDATE");
            exit(0);
        }
    }

    sqlite3_exec(conn, "COMMIT TRANSACTION", NULL, NULL, NULL);
    sqlite3_close(conn);                  // Close the DB handle to free memory

    return 0;
}
