
#include <stdio.h>
#include <stdint.h>
#include <sqlite3.h>

#include "struct.h"
#include "makros.h"

// Load sqlite database table sql and kinetics to record in memory
int gps2arc_record(char *db_name, struct arc_record **arc_table, int *arc_len, struct mag_record **mag_table, int *mag_len) {
    sqlite3 *conn;
    sqlite3_stmt *res;
    int error = 0;
    const char *errMSG;
    const char *tail;

    //int kinetics_len;          // The length of the kinetics table
    //int row_cnt;
    int gps_cnt;
    //int arc_cnt = 0;
    //int state = 0;               // State machine
    //int seq_id;

    error = sqlite3_open(db_name, &conn);
    if (error) {
        puts("Can not open database");
        exit(0);
    }

    error = sqlite3_prepare_v2(conn,"SELECT count(seq_id) AS gps_cnt FROM gps",1000, &res, &tail);

    if (error != SQLITE_OK) {
        puts("We did not get any data!");
        exit(0);
    }

    while (sqlite3_step(res) == SQLITE_ROW) {
        gps_cnt = sqlite3_column_int(res, 0);
        *arc_len = gps_cnt - 1;
        //printf("gps_cnt %u\n", gps_cnt);
        //printf("arc_len %u\n", *arc_len);
    }

    struct arc_record *new_table = (struct arc_record*) malloc((*arc_len) * sizeof(struct arc_record));

    error = sqlite3_prepare_v2(conn,"SELECT seq_id, token FROM event WHERE token='gps' OR token='kinetics' ORDER BY seq_id",1000, &res, &tail);

    if (error != SQLITE_OK) {
        puts("We did not get any data!");
        exit(0);
    }

    int arc_cnt = 0;  // Counter for arc_table values
    int state = 0;
    char token[100];
    int seq_id;
    int seq_id_prev; // Previous seq_id value

    // An array that holds all index to arcs in mag_table
    int *arc_idx = (int*) malloc((*arc_len) * 2 * sizeof(int) + 1);  // <------ lägger till +1 extra arc_idx[arc_cnt2] som dyker upp efter sista gps som left_ptr
    int arc_cnt2 = 0; // Counter for arc_idx array                               '-----> D LEFT kinetics | 3346'

    int row_cnt = 0;
    while (sqlite3_step(res) == SQLITE_ROW) {
        seq_id = sqlite3_column_int(res, 0);
        sprintf(token,"%s", sqlite3_column_text(res, 1));

        #ifdef DEBUG_INFO_0
            printf("%u | ",seq_id);
            printf("%s | \n", token);
        #endif

        //implement state machine and populates arc_table
        if (state == 0 && (strcmp("kinetics", token) == 0)) {
            #ifdef DEBUG_INFO_0
                printf("-----> A %s | \n", token);
            #endif

            state = 0;
            continue;
        }

        if (state == 0 && (strcmp("gps", token) == 0)) {
            #ifdef DEBUG_INFO_0
                printf("-----> B %s | \n", token);
            #endif

            state = 1;
            continue;
        }

        if (state == 1 && (strcmp("gps", token) == 0)) {
            #ifdef DEBUG_INFO_0
                printf("-----> C %s | \n", token);
            #endif

            state = 0;
            continue;
        }

        if (state == 1 && (strcmp("kinetics", token) == 0)) {
            #ifdef DEBUG_INFO_0
                printf("-----> D LEFT %s | %u\n", token, seq_id);
            #endif

            new_table[arc_cnt].left_seq_id = seq_id;
            seq_id_prev = seq_id;

            arc_idx[arc_cnt2] = seq_id;
            arc_cnt2++;

            state = 2;
            continue;
        }

        if (state == 2 && (strcmp("kinetics", token) == 0)) {
            #ifdef DEBUG_INFO_0
                printf("-----> E %s | \n", token);
            #endif

            seq_id_prev = seq_id;

            state = 2;
            continue;
        }

        if (state == 2 && (strcmp("gps", token) == 0)) {

            #ifdef DEBUG_INFO_0
                printf("-----> F RIGHT %s |%u\n", token, seq_id_prev);
            #endif

            new_table[arc_cnt].right_seq_id = seq_id_prev;
            arc_cnt++;

            arc_idx[arc_cnt2] = seq_id_prev;
            arc_cnt2++;

            state = 1;
            continue;
        }

        row_cnt++;
    }

    #ifdef DEBUG_INFO_0
        //DEBUG: print all indexes to arc_table
        for (int i=0; i < (2 * (*arc_len) + 1); i++) {   // +1 här ser man det extra entryt från *arc_idx=... (????? 38 3346)
            printf("????? %u %u \n", i, arc_idx[i]);
        }
    #endif

    int forward = 0; // shifts one step forward for each idx found in arc_idx
    // Find all indexes for an arc into the mag_table
    for(int idx=0; idx < *mag_len; idx++) {

        #ifdef DEBUG_INFO_0
            printf(">>>>> %u %u \n", idx, (*mag_table)[idx].seq_id);
        #endif

        if ((forward < (2 * (*arc_len))) && (arc_idx[forward] == (*mag_table)[idx].seq_id)) {

            #ifdef DEBUG_INFO_0
                printf("::::: ----->%u %u %u \n", forward, idx, (*mag_table)[idx].seq_id);
            #endif

            if ((forward % 2) == 0) { // idx is even, modulo operator
                new_table[forward/2].left_mag_idx = idx;

                #ifdef DEBUG_INFO_0
                    printf("&&&&&&>%u %u left\n", forward/2, idx);
                #endif
            }
            else {
                new_table[forward/2].right_mag_idx = idx;

                #ifdef DEBUG_INFO_0
                  printf("&&&&&&>%u %u right\n", forward/2 , idx);
                #endif
            }

            forward++;
        }
    }

    free(arc_idx); // <-------*** Error in `./cu_normalize11': double free or corruption (!prev): 0x00000000008c7060 ***

    sqlite3_finalize(res);

    sqlite3_close(conn);

    *arc_table = new_table;

    return 0;
}
