

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
