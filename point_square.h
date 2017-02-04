

void point_square(short mxt, short myt, float x0, float y0, float scale_r, float scale_y_axis, float theta,
                  float *normalized_x, float *normalized_y, float *quad_error);

__global__ void point_square_GPU(chunk_record *arc_table, int arc_len, mag_record *mag_table, int mag_len, int arc_size);
