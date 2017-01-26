#include <stdio.h>
#include <stdint.h>
#include <sqlite3.h>

// kompilera
// nvcc cu_normalize.cu -o cu_normalize -lsqlite3

typedef struct mag_record {  // Magnetometer data implement as an array of structs
    int seq_id;
    float mxt;    // CUDA single precision
    float myt;    // CUDA single precision
    bool outlier; // Set outliers to 1 otherwise 0
} helu;

typedef struct arc_record {  // Magnetometer data implement as an array of structs
    int left_seq_id;  // överflödig ..
    int right_seq_id; // överflödig ..
    int left_mag_idx;  //left index of an arc in mag_record[]
    int right_mag_idx; //right index of an arc in mag_record[]
    // More data associated with an arc
} helu2;

// Load sqlite database table kinetics to record in memory
int kinetics2record(char *db_name, struct mag_record **mag_table, int *kinetics_len) {
    sqlite3 *conn;
    sqlite3_stmt *res;
    int error = 0;
    const char *errMSG;
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
        printf("%u|\n", *kinetics_len);
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
        //printf("%f | ",  (float)sqlite3_column_double(res, 1));  // dessa bör castas till CUDA single precision
        //printf("%f\n", (float)sqlite3_column_double(res, 2)); // dessa bör castas till CUDA single precision

        new_table[row_cnt].seq_id = sqlite3_column_int(res, 0);
        new_table[row_cnt].mxt = sqlite3_column_double(res, 1);
        new_table[row_cnt].myt = sqlite3_column_double(res, 2);
        new_table[row_cnt].outlier = 0;

        printf("%u | ",new_table[row_cnt].seq_id);
        printf("%f | ",new_table[row_cnt].mxt);
        printf("%f\n",new_table[row_cnt].myt);

        row_cnt++;
    }

    puts("==========================");

    printf("We received %d records.\n", *kinetics_len);

    sqlite3_finalize(res);

    sqlite3_close(conn);

    *mag_table = new_table;

    return 0;
}


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
        printf("gps_cnt %u\n", gps_cnt);
        printf("arc_len %u\n", *arc_len);
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
    int *arc_idx = (int*) malloc((*arc_len) * 2 * sizeof(int));  // <-----------------------------------------------
    int arc_cnt2 = 0; // Counter for arc_idx array         // <-----------------------------------------------

    int row_cnt = 0;
    while (sqlite3_step(res) == SQLITE_ROW) {
        seq_id = sqlite3_column_int(res, 0);
        sprintf(token,"%s", sqlite3_column_text(res, 1));

        printf("%u | ",seq_id);
        printf("%s | \n", token);

        //implement state machine and populates arc_table
        if (state == 0 && (strcmp("kinetics", token) == 0)) {
            printf("-----> A %s | \n", token);

            state = 0;
            continue;
        }

        if (state == 0 && (strcmp("gps", token) == 0)) {
            printf("-----> B %s | \n", token);

            state = 1;
            continue;
        }

        if (state == 1 && (strcmp("gps", token) == 0)) {
            printf("-----> C %s | \n", token);

            state = 0;
            continue;
        }

        if (state == 1 && (strcmp("kinetics", token) == 0)) {
            printf("-----> D LEFT %s | %u\n", token, seq_id);

            new_table[arc_cnt].left_seq_id = seq_id;
            seq_id_prev = seq_id;

            arc_idx[arc_cnt2] = seq_id;
            arc_cnt2++;

            state = 2;
            continue;
        }

        if (state == 2 && (strcmp("kinetics", token) == 0)) {
            printf("-----> E %s | \n", token);

            seq_id_prev = seq_id;

            state = 2;
            continue;
        }

        if (state == 2 && (strcmp("gps", token) == 0)) {
            printf("-----> F RIGHT %s |%u\n", token, seq_id_prev);

            new_table[arc_cnt].right_seq_id = seq_id_prev;
            arc_cnt++;

            arc_idx[arc_cnt2] = seq_id_prev;
            arc_cnt2++;

            state = 1;
            continue;
        }

        row_cnt++;
    }

    // DEBUG: print all indexes to arc_table
    for (int i=0; i < (2 * (*arc_len)); i++) {
        printf("::::: %u %u \n", i, arc_idx[i]);
    }

    int forward = 0; // shifts one step forward for each idx found in arc_idx
    // Find all indexes for an arc into the mag_table
    for(int idx=0; idx < *mag_len; idx++) {
        printf(">>>>> %u %u \n", idx, (*mag_table)[idx].seq_id);
        if ((forward < (2 * (*arc_len))) && (arc_idx[forward] == (*mag_table)[idx].seq_id)) {
            printf("::::: ----->%u %u %u \n", forward, idx, (*mag_table)[idx].seq_id);

            if ((forward % 2) == 0) { // idx is even, modulo operator
                new_table[forward/2].left_mag_idx = idx;
                printf("&&&&&&>%u %u left\n", forward/2, idx);
            }
            else {
                new_table[forward/2].right_mag_idx = idx;
                printf("&&&&&&>%u %u right\n", forward/2 , idx);
            }

            forward++;
        }
    }

    //free(arc_idx); // <-------*** Error in `./cu_normalize11': double free or corruption (!prev): 0x00000000008c7060 ***

    sqlite3_finalize(res);

    sqlite3_close(conn);

    *arc_table = new_table;

    return 0;
}


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

    for (int row_cnt=0; row_cnt<kinetics_len; row_cnt++) {
        printf(">> %u | ",mag_table[row_cnt].seq_id);
        printf("%f | ",mag_table[row_cnt].mxt);
        printf("%f\n",mag_table[row_cnt].myt);
    }

    gps2arc_record(buffer_Z, &arc_table, &arc_len, &mag_table, &kinetics_len);

    for (int rec_cnt=0; rec_cnt<arc_len; rec_cnt++) {
        printf("++->%u | %u | %u | ",rec_cnt, arc_table[rec_cnt].left_seq_id, arc_table[rec_cnt].right_seq_id);
        printf("%u | %u \n", arc_table[rec_cnt].left_mag_idx, arc_table[rec_cnt].right_mag_idx);
    }

    printf("length fux %u \n\n",kinetics_len);

    // printf("%u | ",magtable[555].seq_id);
    // printf("%f | ",magtable[555].mxt);
    // printf("%f\n",magtable[555].myt);



    return 0;

}
