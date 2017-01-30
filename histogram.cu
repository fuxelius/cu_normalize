
#include <stdio.h>
#include <stdint.h>
#include <sqlite3.h>

#include "struct.h"
#include "makros.h"


// Check if hist_table has an entry in table > cut_off
// Used to see if the histogram is skewed into only one side of origo
// False if reaching the end without finding a value higher then cut_off in table, otherwise return index
int has_cut_off(int** hist_table, int range, int cut_off) {
    for (int i=range - 1; i >= 0; i--) {
        //printf("i = %u\n", i);
        if((*hist_table)[i] > cut_off) {
            return i;
        }
    }
    return -1; // nothing found
}

int has_cut_off_rev(int** hist_table, int range, int cut_off) {
    for (int i=0; i<range; i++) {
        //printf("i = %u\n", i);
        if((*hist_table)[i] > cut_off) {
            return i;
        }
    }
    return -1; // nothing found
}

//====================================================================================================================================
// histogram takes the left and right index into magtable from arc_table to pick out magrecords (mxt,myt) . It loops
// over it from left to right and creates a histogram in four hist_table:s. First all outliers are removed and the the seed
// values for the iterative parameters are set. In a later stage a CUDA function takes the seed values to do an  iteration
// over each arc and set the values (mfv, rho) for each arc
// left_arc_idx  = pointer into left arc in mag_table
// right_arc_idx = pointer into right arc of another or the same arc in mag_table
// bin = the size of each bin (now 5)
// range = the number of bins in positive and negative direction (now 100)
// cut_off = outer bins are cut off if lower than cut_off (now 3)

int histogram(struct arc_record **arc_table, int *arc_len, struct mag_record **mag_table, int *mag_len,
                                             int left_arc_idx, int right_arc_idx, int bin, int range, int cut_off) {

    puts("\n>All arcs in mag_table");

    int *hist_table_mxt_pos = (int*) malloc(range * sizeof(int)); // positive values
    int *hist_table_mxt_neg = (int*) malloc(range * sizeof(int)); // negative values
    int *hist_table_myt_pos = (int*) malloc(range * sizeof(int)); // positive values
    int *hist_table_myt_neg = (int*) malloc(range * sizeof(int)); // negative values

    memset(hist_table_mxt_pos, 0, range * sizeof(int));  // initialize to zeros
    memset(hist_table_mxt_neg, 0, range * sizeof(int));  // initialize to zeros
    memset(hist_table_myt_pos, 0, range * sizeof(int));  // initialize to zeros
    memset(hist_table_myt_neg, 0, range * sizeof(int));  // initialize to zeros

    puts("\n>Next arc");

    float mxt;
    float myt;

    int mxt_idx; // mxt indexed to hist_tables
    int myt_idx; // mxt indexed to hist_tables

    // travese all mag_records between left_arc_idx and right_arc_idx, constituting multiple arcs. f.ex. 3 arcs at a time

    for (int mag_idx = (*arc_table)[left_arc_idx].left_mag_idx; mag_idx <= (*arc_table)[right_arc_idx].right_mag_idx; mag_idx++) {

        mxt = (*mag_table)[mag_idx].mxt;
        myt = (*mag_table)[mag_idx].myt;

        printf("> %u\t%f\t%f", mag_idx, mxt, myt);

        // build up the histogram here
        if (mxt >= 0) {
            mxt_idx = (int)(mxt/bin);
            printf("\tmxt_idx: +%u ", mxt_idx);

            if (mxt_idx < range) {
                // addera till hist_table_mxt_pos
                hist_table_mxt_pos[mxt_idx]++; // add +1 to the bin

            }
        }
        else { // mxt < 0
            mxt_idx = (int)(-mxt/bin);
            printf("\tmxt_idx: -%u", mxt_idx);

            if (mxt_idx < range) {
                // addera till hist_table_mxt_pos
                hist_table_mxt_neg[mxt_idx]++; // add +1 to the bin

            }
        }

        if (myt >= 0) {
            myt_idx = (int)(myt/bin);
            printf("\tmyt_idx: +%u\n", myt_idx);

            if (myt_idx < range) {
                // addera till hist_table_mxt_pos
                hist_table_myt_pos[myt_idx]++; // add +1 to the bin

            }
        }
        else { // myt < 0
            myt_idx = (int)(-myt/bin);
            printf("\tmyt_idx: -%u\n", myt_idx);

            if (myt_idx < range) {
                // addera till hist_table_mxt_pos
                hist_table_myt_neg[myt_idx]++; // add +1 to the bin

            }
        }

    }

    int mxt_pos_idx = has_cut_off(&hist_table_mxt_pos, range, cut_off);
    int mxt_pos_idx_rev = has_cut_off_rev(&hist_table_mxt_pos, range, cut_off);
    int mxt_neg_idx = has_cut_off(&hist_table_mxt_neg, range, cut_off);
    int mxt_neg_idx_rev = has_cut_off_rev(&hist_table_mxt_neg, range, cut_off);
    int myt_pos_idx = has_cut_off(&hist_table_myt_pos, range, cut_off);
    int myt_pos_idx_rev = has_cut_off_rev(&hist_table_myt_pos, range, cut_off);
    int myt_neg_idx = has_cut_off(&hist_table_myt_neg, range, cut_off);
    int myt_neg_idx_rev = has_cut_off_rev(&hist_table_myt_neg, range, cut_off);


    #ifdef DEBUG_INFO_2
        puts("\nmxt+");
        for (int i=0; i<range; i++) {

            printf(" %u", hist_table_mxt_pos[i]);

        }
        printf("\nforvard=%d (# %d) (value %d)", mxt_pos_idx, hist_table_mxt_pos[mxt_pos_idx], (bin) * mxt_pos_idx);
        printf("\nreverse=%d (# %d) (value %d)\n\n", mxt_pos_idx_rev, hist_table_mxt_pos[mxt_pos_idx_rev], (bin) * mxt_pos_idx_rev);


        puts("mxt-");
        for (int i=0; i<range; i++) {

            printf(" %u", hist_table_mxt_neg[i]);

        }
        printf("\nposition=%d (# %d) (value %d)", mxt_neg_idx, hist_table_mxt_neg[mxt_neg_idx], (-bin) * mxt_neg_idx);
        printf("\nreverse=%d (# %d) (value %d)\n\n", mxt_neg_idx_rev, hist_table_mxt_neg[mxt_neg_idx_rev], (bin) * mxt_neg_idx_rev);


        puts("myt+");
        for (int i=0; i<range; i++) {

            printf(" %u", hist_table_myt_pos[i]);

        }
        printf("\nposition=%d (# %d) (value %d)", myt_pos_idx, hist_table_myt_pos[myt_pos_idx], (bin) * myt_pos_idx);
        printf("\nreverse=%d (# %d) (value %d)\n\n", myt_pos_idx_rev, hist_table_myt_pos[myt_pos_idx_rev], (bin) * myt_pos_idx_rev);

        puts("myt-");
        for (int i=0; i<range; i++) {

            printf(" %u", hist_table_myt_neg[i]);

        }
        printf("\nposition=%d (# %d) (value %d)", myt_neg_idx, hist_table_myt_neg[myt_neg_idx], (-bin) * myt_neg_idx);
        printf("\nreverse=%d (# %d) (value %d)\n\n", myt_neg_idx_rev, hist_table_myt_pos[myt_neg_idx_rev], (bin) * myt_neg_idx_rev);
    #endif


    // Find the boundaries of the histogram
    float mxt_boundry_high;
    float mxt_boundry_low;
    float myt_boundry_high;
    float myt_boundry_low;

    // if elements > cut_off exist on both sides of origo
    if (mxt_pos_idx != -1 && mxt_neg_idx != -1) {  // both sides of origo
        mxt_boundry_high = (float) (bin) * mxt_pos_idx;
        mxt_boundry_low  = (float) (-bin) * mxt_neg_idx;

        printf("\nx=<%f,%f>\n", mxt_boundry_low, mxt_boundry_high);
    }
    else if (mxt_pos_idx != -1) { // only positive side of origo
            mxt_boundry_high = (float) (bin) * mxt_pos_idx;
            mxt_boundry_low  = (float) (bin) * mxt_pos_idx_rev;
            printf("\nx=<%f,%f>\n", mxt_boundry_low, mxt_boundry_high);
        }
        else { // mxt_neg_idx != -1, only negative side of origo
            mxt_boundry_high = (float) (-bin) * mxt_neg_idx_rev;
            mxt_boundry_low  = (float) (-bin) * mxt_neg_idx;
            printf("\nx=<%f,%f>\n", mxt_boundry_low, mxt_boundry_high);
        }

    // if elements > cut_off exist on both sides of origo
    if (myt_pos_idx != -1 && myt_neg_idx != -1) {  // both sides of origo
        myt_boundry_high = (float) (bin) * myt_pos_idx;
        myt_boundry_low  = (float) (-bin) * myt_neg_idx;

        printf("\nx=<%f,%f>\n", myt_boundry_low, myt_boundry_high);
    }
    else if (myt_pos_idx != -1) { // only positive side of origo
            myt_boundry_high = (float) (bin) * myt_pos_idx;
            myt_boundry_low  = (float) (bin) * myt_pos_idx_rev;
            printf("\ny=<%f,%f>\n", myt_boundry_low, myt_boundry_high);
        }
        else { // myt_neg_idx != -1, only negative side of origo
            myt_boundry_high = (float) (-bin) * myt_neg_idx_rev;
            myt_boundry_low  = (float) (-bin) * myt_neg_idx;
            printf("\ny=<%f,%f>\n", myt_boundry_low, myt_boundry_high);
        }

    // 1) loop throught mxt,myt and set the outliers
    // 2) Räkna fram seeds
    // 3) set argument (-histogram-raw) for writing mxt,myt raw till stdout, så man kan plotta i R


    free(hist_table_mxt_pos);
    free(hist_table_mxt_neg);
    free(hist_table_myt_pos);
    free(hist_table_myt_neg);

    return 0;
}
