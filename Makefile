CC=nvcc
hellomake: main.cu slice2chunk.cu kinetics.cu histogram.cu cuda_magix.cu device_info.cu
	$(CC) -arch=sm_60 -rdc=true -lcudadevrt -Wno-deprecated-gpu-targets -lsqlite3 -o normalize main.cu slice2chunk.cu kinetics.cu histogram.cu cuda_magix.cu device_info.cu 
