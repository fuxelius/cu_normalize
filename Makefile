CC=nvcc
hellomake: main.cu gps2arc.cu kinetics.cu
	$(CC) -lsqlite3 -o normalize main.cu gps2arc.cu kinetics.cu
