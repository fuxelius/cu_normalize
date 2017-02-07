


__global__ void point_square_GPU(chunk_record *chunk_table, int chunk_len,
                                     mag_record *mag_table, int mag_len, int chunk_size, int meta_idx);

void host_launch(chunk_record *chunk_table, int chunk_len,
                                  mag_record *mag_table, int mag_len,
                                  meta_record *meta_table, int meta_len,
                                                           int chunk_size);
