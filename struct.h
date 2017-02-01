typedef struct mag_record {  // Magnetometer data implement as an array of structs
    int seq_id;        // seq_id from database
    float mxt;         // CUDA single precision
    float myt;         // CUDA single precision
    bool outlier;      // Set outliers to 1 otherwise 0

    // results from CUDA iteration
    float quad_error;  // quadratic error
    float normalized_x;
    float normalized_y;
    float mfv;         // magnetic field vector
    float rho;         // baering

} helu;


// mag_record[left_mag_idx] - mag_record[right_mag_idx] utgör en arc                  <----------------- ta bort denna
typedef struct arc_record {  // Magnetometer data implement as an array of structs
    int left_seq_id;         // calculated in kinetics
    int right_seq_id;        // calculated in kinetics
    int left_mag_idx;        // left index of an arc in mag_record[]; calculated in gps2arc
    int right_mag_idx;       // right index of an arc in mag_record[]; calculated in gps2arc
} helu2;


//  flytta dessa till en meta_table[meta_idx] som täcker N arcs
//  bör vara multipler av 32 mag_records för att passa CUDA WARPS
//  välj i första hand så att det blir lagom stora datasets
typedef struct meta_record {
    int left_mag_idx;  // tidigare left_arc_idx
    int right_mag_idx; // tidigare right_arc_idx

    // iterative parameters used in CUDA
    float x0;               // seed_x första gången i iterationen
    float y0;               // seed_y första gången i iterationen
    float scale_r;          // seed_scale_r första gången i iterationen
    float scale_y_axis;     // = 1 första gången i iterationen
    float theta;            // = 0 första gången i iterationen

    float ls;               // least square for an iteration
    int deepth;             // The depth of iteration

} helu3;
