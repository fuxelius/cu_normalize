CC=nvcc
hellomake: main.cu gps2arc.cu kinetics.cu histogram.cu point_square.cu device_info.cu
	$(CC) -lsqlite3 -o normalize main.cu gps2arc.cu kinetics.cu histogram.cu point_square.cu device_info.cu
