#define BLOCK_SIZE 256

typedef struct {  // Magnetometer data implement as an array of structs
    int seq_id;        // seq_id from database
    float mxt;         // CUDA single precision
    float myt;         // CUDA single precision
    bool outlier;      // Set outliers to 1 otherwise 0

    // results from CUDA iteration
    float normalized_x;
    float normalized_y;
    float quad_error;  // quadratic error

    float mfv;         // magnetic field vector
    float rho;         // baering

} mag_record;


// arc_record divides mag_table to chunks at a size suitable for normalization
typedef struct {  // Magnetometer data implement as an array of structs
    int left_mag_idx;        // left index of an arc in mag_record[]; calculated in slice2arc
    int right_mag_idx;       // right index of an arc in mag_record[]; calculated in slice2arc

    int left_seq_id;         // calculated in kinetics
    int right_seq_id;        // calculated in kinetics

    // iterative parameters used in CUDA
    float x0;               // seed_x första gången i iterationen
    float y0;               // seed_y första gången i iterationen
    float scale_r;          // seed_scale_r första gången i iterationen
    float scale_y_axis;     // = 1 första gången i iterationen
    float theta;            // = 0 första gången i iterationen

    float ls;               // least square for an iteration of the elements in arc_table[arc_idx]
    int deepth;             // The total depth of iteration
    int iter_cnt;           // If lest square is not lower for iter_cnt cycles
    int finish;             // then set the thread for this arc to finish

    bool outlier;           // Set outliers to 1 otherwise 0
} arc_record;
