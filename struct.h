#define BLOCK_SIZE 256

typedef struct {
    int seq_id;
    float mfv;
    float rho;
} result_struct;

typedef struct {  // Magnetometer data implement as an array of structs
    //int seq_id;        // seq_id from database
    short mxt;         // CUDA single precision
    short myt;         // CUDA single precision
    bool disable;      // Set outliers to 1 otherwise 0
} mag_record;


// arc_record divides mag_table to chunks at a size suitable for normalization
typedef struct {  // Magnetometer data implement as an array of structs
    int left_mag_idx;        // left index of an arc in mag_record[]; calculated in slice2arc
    int right_mag_idx;       // right index of an arc in mag_record[]; calculated in slice2arc

    //int left_seq_id;         // calculated in kinetics
    //int right_seq_id;        // calculated in kinetics

    // iterative parameters used in CUDA
    float x0;               // seed_x första gången i iterationen
    float y0;               // seed_y första gången i iterationen
    float scale_r;          // seed_scale_r första gången i iterationen
    float scale_y_axis;     // = 1 första gången i iterationen
    float theta;            // = 0 första gången i iterationen

    // these should probably be in a matrix
    float ls;               // least square for an iteration of the elements in arc_table[arc_idx]
    int deepth;             // The total depth of iteration
    int iter_cnt;           // If lest square is not lower for iter_cnt cycles
    int finish;             // then set the thread for this arc to finish

    bool outlier;           // Set outliers to 1 otherwise 0
} arc_record;
