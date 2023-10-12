# Normalize skewed magnetometer values with GPU and CUDA-C
Blazingly fast rectangular to polar normalization of massive datasets of projected magnetometer values

## Install CUDA Toolkit

https://developer.nvidia.com


## Update .bashrc
    # Cuda Toolkit
    export PATH="/usr/local/cuda-12.2/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"

## Install SQLite3 development library and compile
    sudo apt-get install libsqlite3-dev
    make

## Usage
    ./normalize ga_uppsala_2.sqlite3
    
## Normalized dataset
Dataset projected on the unity circle (R heatmap)

<img src="analys/ga_uppsala_korrekt.png"  width="400">