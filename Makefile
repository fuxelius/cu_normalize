CC=nvcc
hellomake: main.cu slice2chunk.cu kinetics.cu histogram.cu point_square.cu device_info.cu
	$(CC) -arch=sm_35 -rdc=true -lcudadevrt -Wno-deprecated-gpu-targets -lsqlite3 -o normalize main.cu slice2chunk.cu kinetics.cu histogram.cu point_square.cu device_info.cu 
