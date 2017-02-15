
#include <stdio.h>
#include <stdint.h>
#include <sqlite3.h>

#include "struct.h"
#include "makros.h"


// Check if hist_table has an entry in table > cut_off
// Used to see if the histogram is skewed into only one side of origo
// False if reaching the end without finding a value higher then cut_off in table, otherwise return index
int has_cut_off_xy(int* hist_table, int range, int cut_off) {
    for (int i=range - 1; i >= 0; i--) {
        //printf("i = %u\n", i);
        if(hist_table[i] > cut_off) {
            return i;
        }
    }
    return -1; // nothing found
}

int has_cut_off_xy_rev(int* hist_table, int range, int cut_off) {
    for (int i=0; i<range; i++) {
        //printf("i = %u\n", i);
        if(hist_table[i] > cut_off) {
            return i;
        }
    }
    return -1; // nothing found
}

//====================================================================================================================================
// histogram takes the left and right index into magtable from chunk_table to pick out magrecords (mxt,myt) . It loops
// over it from left to right and creates a histogram in four hist_table:s. First all outliers are removed and the the seed
// values for the iterative parameters are set. In a later stage a CUDA function takes the seed values to do an  iteration
// over each chunk and set the values (mfv, rho) for each chunk
// left_chunk_idx  = pointer into left chunk in mag_table
// right_chunk_idx = pointer into right chunk of another or the same chunk in mag_table
// bin = the size of each bin (now 5)
// range = the number of bins in positive and negative direction (now 100)
// cut_off = outer bins are cut off if lower than cut_off (now 3)

int x0y0_histogram(chunk_record *chunk_table, int chunk_len, int bin, int range, int cut_off) {

    // int left_chunk_idx  = chunk_idx;
    // int right_chunk_idx = chunk_idx;

    //puts("\n>All chunks in mag_table");

    int *hist_table_x0_pos = (int*) malloc(range * sizeof(int)); // positive values
    int *hist_table_x0_neg = (int*) malloc(range * sizeof(int)); // negative values
    int *hist_table_y0_pos = (int*) malloc(range * sizeof(int)); // positive values
    int *hist_table_y0_neg = (int*) malloc(range * sizeof(int)); // negative values

    memset(hist_table_x0_pos, 0, range * sizeof(int));  // initialize to zeros
    memset(hist_table_x0_neg, 0, range * sizeof(int));  // initialize to zeros
    memset(hist_table_y0_pos, 0, range * sizeof(int));  // initialize to zeros
    memset(hist_table_y0_neg, 0, range * sizeof(int));  // initialize to zeros

    //puts("\n>Next chunk");

    // float mxt;
    // float myt;
    //
    // int mxt_idx; // mxt indexed to hist_tables
    // int myt_idx; // mxt indexed to hist_tables

    float x0;
    float y0;

    int x0_idx; // mxt indexed to hist_tables
    int y0_idx; // mxt indexed to hist_tables

    // travese all mag_records between left_chunk_idx and right_chunk_idx, constituting multiple chunks. f.ex. 3 chunks at a time

    for (int chunk_idx = 0; chunk_idx < chunk_len; chunk_idx++) {
    //for (int mag_idx = chunk_table[left_chunk_idx].left_mag_idx; mag_idx <= chunk_table[right_chunk_idx].right_mag_idx; mag_idx++)

        // mxt = mag_table[mag_idx].mxt;
        // myt = mag_table[mag_idx].myt;

        x0 = chunk_table[chunk_idx].x0;
        y0 = chunk_table[chunk_idx].y0;

        //printf("> %u\t%f\t%f", mag_idx, mxt, myt);

        // build up the histogram here cut out everything out of range
        if (x0 >= 0) {
            x0_idx = (int)(x0/bin);
            //printf("\tmxt_idx: +%u ", mxt_idx);

            if (x0_idx < range) {
                // addera till hist_table_mxt_pos
                hist_table_x0_pos[x0_idx]++; // add +1 to the bin

            }
        }
        else { // mxt < 0
            x0_idx = (int)(-x0/bin);
            //printf("\tmxt_idx: -%u", mxt_idx);

            if (x0_idx < range) {
                // addera till hist_table_mxt_pos
                hist_table_x0_neg[x0_idx]++; // add +1 to the bin

            }
        }

        if (y0 >= 0) {
            y0_idx = (int)(y0/bin);
            //printf("\tmyt_idx: +%u\n", myt_idx);

            if (y0_idx < range) {
                // addera till hist_table_mxt_pos
                hist_table_y0_pos[y0_idx]++; // add +1 to the bin

            }
        }
        else { // myt < 0
            y0_idx = (int)(-y0/bin);
            //printf("\tmyt_idx: -%u\n", myt_idx);

            if (y0_idx < range) {
                // addera till hist_table_mxt_pos
                hist_table_y0_neg[y0_idx]++; // add +1 to the bin

            }
        }
    }

    int x0_pos_idx = has_cut_off_xy(hist_table_x0_pos, range, cut_off);
    int x0_pos_idx_rev = has_cut_off_xy_rev(hist_table_x0_pos, range, cut_off);
    int x0_neg_idx = has_cut_off_xy(hist_table_x0_neg, range, cut_off);
    int x0_neg_idx_rev = has_cut_off_xy_rev(hist_table_x0_neg, range, cut_off);
    int y0_pos_idx = has_cut_off_xy(hist_table_y0_pos, range, cut_off);
    int y0_pos_idx_rev = has_cut_off_xy_rev(hist_table_y0_pos, range, cut_off);
    int y0_neg_idx = has_cut_off_xy(hist_table_y0_neg, range, cut_off);
    int y0_neg_idx_rev = has_cut_off_xy_rev(hist_table_y0_neg, range, cut_off);


    #ifdef DEBUG_INFO_2
        //puts("\nmxt+");
        for (int i=0; i<range; i++) {

            //printf(" %u", hist_table_mxt_pos[i]);

        }
        //printf("\nforvard=%d (# %d) (value %d)", mxt_pos_idx, hist_table_mxt_pos[mxt_pos_idx], (bin) * mxt_pos_idx);
        //printf("\nreverse=%d (# %d) (value %d)\n\n", mxt_pos_idx_rev, hist_table_mxt_pos[mxt_pos_idx_rev], (bin) * mxt_pos_idx_rev);


        //puts("mxt-");
        for (int i=0; i<range; i++) {

            //printf(" %u", hist_table_mxt_neg[i]);

        }
        //printf("\nposition=%d (# %d) (value %d)", mxt_neg_idx, hist_table_mxt_neg[mxt_neg_idx], (-bin) * mxt_neg_idx);
        //printf("\nreverse=%d (# %d) (value %d)\n\n", mxt_neg_idx_rev, hist_table_mxt_neg[mxt_neg_idx_rev], (bin) * mxt_neg_idx_rev);


        //puts("myt+");
        for (int i=0; i<range; i++) {

            //printf(" %u", hist_table_myt_pos[i]);

        }
        //printf("\nposition=%d (# %d) (value %d)", myt_pos_idx, hist_table_myt_pos[myt_pos_idx], (bin) * myt_pos_idx);
        //printf("\nreverse=%d (# %d) (value %d)\n\n", myt_pos_idx_rev, hist_table_myt_pos[myt_pos_idx_rev], (bin) * myt_pos_idx_rev);

        //puts("myt-");
        for (int i=0; i<range; i++) {

            //printf(" %u", hist_table_myt_neg[i]);

        }
        //printf("\nposition=%d (# %d) (value %d)", myt_neg_idx, hist_table_myt_neg[myt_neg_idx], (-bin) * myt_neg_idx);
        //printf("\nreverse=%d (# %d) (value %d)\n\n", myt_neg_idx_rev, hist_table_myt_pos[myt_neg_idx_rev], (bin) * myt_neg_idx_rev);
    #endif


    // Find the boundaries of the histogram
    float x0_boundry_high = 0;
    float x0_boundry_low = 0;
    float y0_boundry_high = 0;
    float y0_boundry_low = 0;

    // if elements > cut_off exist on both sides of origo
    if (x0_pos_idx != -1 && x0_neg_idx != -1) {  // both sides of origo
        x0_boundry_high = (float) (bin+1) * x0_pos_idx;
        x0_boundry_low  = (float) (-bin-1) * x0_neg_idx;
    }
    else if (x0_pos_idx != -1) { // only positive side of origo
            x0_boundry_high = (float) (bin+1) * x0_pos_idx;
            x0_boundry_low  = (float) (bin-1) * x0_pos_idx_rev;
        }
        else { // mxt_neg_idx != -1, only negative side of origo
            x0_boundry_high = (float) (-bin+1) * x0_neg_idx_rev;
            x0_boundry_low  = (float) (-bin-1) * x0_neg_idx;
        }

    printf("\nx_limit=<%f,%f>\n", x0_boundry_low, x0_boundry_high);

    // if elements > cut_off exist on both sides of origo
    if (y0_pos_idx != -1 && y0_neg_idx != -1) {  // both sides of origo
        y0_boundry_high = (float) (bin+1) * y0_pos_idx;
        y0_boundry_low  = (float) (-bin-1) * y0_neg_idx;
    }
    else if (y0_pos_idx != -1) { // only positive side of origo
            y0_boundry_high = (float) (bin+1) * y0_pos_idx;
            y0_boundry_low  = (float) (bin-1) * y0_pos_idx_rev;
        }
        else { // myt_neg_idx != -1, only negative side of origo
            y0_boundry_high = (float) (-bin+1) * y0_neg_idx_rev;
            y0_boundry_low  = (float) (-bin-1) * y0_neg_idx;
        }

    printf("y_limit=<%f,%f>\n\n", y0_boundry_low, y0_boundry_high);

    printf("x0,y0\n");

    // 1) loop throught mxt,myt and set the outliers in mag_table->outlier
    //for (int mag_idx = chunk_table[left_chunk_idx].left_mag_idx; mag_idx <= chunk_table[right_chunk_idx].right_mag_idx; mag_idx++)
    for (int chunk_idx = 0; chunk_idx < chunk_len; chunk_idx++) {
        x0 = chunk_table[chunk_idx].x0;
        y0 = chunk_table[chunk_idx].y0;

        if (!(x0_boundry_low < x0 && x0 < x0_boundry_high && y0_boundry_low < y0 && y0 < y0_boundry_high)) {
            //printf("outlier %f,%f\n", x0, y0);
            // mark outlier
            chunk_table[chunk_idx].outlier = 1; // Set outlier true
        }
        else {
            printf("%f,%f\n", x0, y0);
            chunk_table[chunk_idx].outlier = 0; // Set outlier false
        }
    }

    float x0_sum = 0;               // seed_x första gången i iterationen
    float y0_sum = 0;               // seed_y första gången i iterationen
    float scale_r_sum = 0;          // seed_scale_r första gången i iterationen
    float scale_y_axis_sum = 0;     // = 1 första gången i iterationen
    float theta_sum = 0;            // = 0 första gången i iterationen

    int N=0; // counter

    for (int chunk_idx = 0; chunk_idx < chunk_len; chunk_idx++) {
        if (chunk_table[chunk_idx].outlier == 0) {
            x0_sum += chunk_table[chunk_idx].x0;
            y0_sum += chunk_table[chunk_idx].y0;
            scale_r_sum += chunk_table[chunk_idx].scale_r;
            scale_y_axis_sum += chunk_table[chunk_idx].scale_y_axis;
            theta_sum += chunk_table[chunk_idx].theta;
            N++;
        }
    }

    printf("x_0: %f\n", x0_sum/N);
    printf("y_0: %f\n", y0_sum/N);
    printf("scale_r: %f\n", scale_r_sum/N);
    printf("scale_y_axis: %f\n", scale_y_axis_sum/N);
    printf("theta:% f\n", theta_sum/N);

    // inter/extra-polera fram x0y0 för alla outliers

    free(hist_table_x0_pos);
    free(hist_table_x0_neg);
    free(hist_table_y0_pos);
    free(hist_table_y0_neg);

    return 0;
}
