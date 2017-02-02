

void point_square(float mxt, float myt, float x0, float y0, float scale_r, float scale_y_axis, float theta,
                  float *normalized_x, float *normalized_y, float *quad_error);

__global__ void point_square_GPU(struct arc_record **arc_table, int arc_len, struct mag_record **mag_table, int mag_len, int arc_size);






__global__ void sumMatrixOnGPUMix(float *MatA, float *MatB, float *MatC, int nx, int ny);

void initialData(float *ip, const int size);
