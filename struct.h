


typedef struct mag_record {  // Magnetometer data implement as an array of structs
    int seq_id;        // seq_id from database
    float mxt;         // CUDA single precision
    float myt;         // CUDA single precision
    bool outlier;      // Set outliers to 1 otherwise 0

    // results from CUDA iteration
    float quad_error;  // quadratic error
    float mfv;            // magnetic field vector
    float rho;            // baering

} helu;



// mag_record[left_mag_idx] - mag_record[right_mag_idx] utgör en arc
typedef struct arc_record {  // Magnetometer data implement as an array of structs
    int left_seq_id;         // calculated in kinetics
    int right_seq_id;        // calculated in kinetics
    int left_mag_idx;        // left index of an arc in mag_record[]; calculated in gps2arc
    int right_mag_idx;       // right index of an arc in mag_record[]; calculated in gps2arc
} helu2;



//  flytta dessa till en meta_table[meta_idx] som täcker N arcs
typedef struct meta_record {
    int left_arc_idx;
    int right_arc_idx;
    // lenght = right_arc_idx - left_arc_idx; needed for CUDA call

    float seed_x;
    float seed_y;
    float seed_scale_r;
    //float seed_scale_y = 1;     These are set in CUDA direct
    //float seed_skew_rad = 0;    These are set in CUDA direct

    // iteration internals
    int deepth;           // The depth of iteration

    // iterative parameters used in CUDA
    float ls;             // least square for an iteration
    float x0;
    float y0;
    float scale_r;
    float scale;
    float rotate;
} helu3;
