hellomake: main.cu gps2arc.cu kinetics.cu
	nvcc -o normalize main.cu gps2arc.cu kinetics.cu -lsqlite3
