CC=nvcc
hellomake: main.cu gps2arc.cu kinetics.cu histogram.cu
	$(CC) -lsqlite3 -o normalize main.cu gps2arc.cu kinetics.cu histogram.cu
