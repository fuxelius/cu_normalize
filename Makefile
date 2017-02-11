CC=nvcc
hellomake: main.cu slice2chunk.cu kinetics.cu histogram.cu cuda_magix.cu device_info.cu slice2meta_record.cu polar.cu
	$(CC) -arch=sm_60 -rdc=true -lcudadevrt -lcurand -Wno-deprecated-gpu-targets -lsqlite3 -o normalize main.cu slice2chunk.cu kinetics.cu histogram.cu cuda_magix.cu device_info.cu slice2meta_record.cu polar.cu
