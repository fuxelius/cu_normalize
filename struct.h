#define BLOCK_SIZE 256      // IE CUDA block size
#define CHUNK_SIZE 1024     // HAS TO BE A **MULTIPLE** OF BLOCK_SIZE; how many magvalues to minimize in each step;
#define META_SIZE 10        // default=100 how many chunks to handle each time on the devie (constant memory)

// result_struct has the same length as mag_record = mag_table
// They both have the same index, so indexing into result_struct[i].seq_id
// refers back to the structure of the database. 96 bytes
typedef struct {       // Host only, takes to much space for device (20*10^6 entries (2 years) => 2Gbyte)
    int seq_id;
    float mfv;
    float rho;
} result_record;

// Magnetometer data implemented as an array of structs
// 33 bytes/record => 20*10^6 entries (2 years) => 660 Mbyte of datastructure
typedef struct {       // Host and device
    short mxt;         // 16-bit
    short myt;         // 16-bit
    bool disable;      // Set outliers to 1 otherwise 0
} mag_record;

// chunk_record divides mag_table to chunks at a size suitable for normalization
typedef struct {            // Magnetometer data implement as an array of structs
    int left_mag_idx;       // left index of an chunk in mag_record[]; calculated in slice2chunk
    int right_mag_idx;      // right index of an chunk in mag_record[]; calculated in slice2chunk

    bool outlier;           // Set outliers to 1 otherwise 0

    // iterative parameters used in CUDA, used in CUDA function calls
    float x0;               // seed_x första gången i iterationen
    float y0;               // seed_y första gången i iterationen
    float scale_r;          // seed_scale_r första gången i iterationen
    float scale_y_axis;     // = 1 första gången i iterationen
    float theta;            // = 0 första gången i iterationen

    // these should probably live in a matrix
    float lsq;              // least square for an iteration of the elements in chunk_table[chunk_idx]
    int iter_cnt;           // If lest square is not lower for iter_cnt cycles
    bool finish;             // then set the thread for this chunk to finish
} chunk_record;

// Partitions how many chunks to run in paralell on device:
// 100 is good (if chunk_record is 1024 => 102400 thraeds)
// 102400 threads => error_matrix = 102400 * 33 bytes = 3.38 Mbytes
typedef struct {
    int left_chunk_idx;
    int right_chunk_idx;
} meta_record;
