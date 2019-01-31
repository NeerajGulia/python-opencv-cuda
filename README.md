# python-opencv-cuda

## Solution
1. Create custom opencv_contrib module
2. Write C++ code to wrap the OpenCV CUDA method
3. Using OpenCV python bindings, expose your custom method
4. Build opencv with opencv_contrib
5. Run python code to test

## Steps to create the build
### Unzip the source: 
1. opencv source code: https://github.com/opencv/opencv/archive/3.4.2.zip
2. opencv_contrib source code: https://github.com/opencv/opencv_contrib/archive/3.4.2.zip

### Create custom module
1.	Copy the folder named "pythoncuda" (inside c++ folder) to: opencv_contrib/modules

### Build opencv using following cmake command
1. create build directory inside the opencv folder, cd to the build directory
2. cmake (I used anaconda3 with environment named as: tensorflow_p36 (with python 3.6))
```
cmake \
-DCMAKE_BUILD_TYPE=RELEASE \
-DWITH_CUDA=ON \
-DCMAKE_INSTALL_PREFIX="/home/$USER/anaconda3/envs/tensorflow_p36" \
-DOPENCV_EXTRA_MODULES_PATH="../../opencv_contrib-3.4.2/modules" \
-DINSTALL_PYTHON_EXAMPLES=OFF \
-DINSTALL_C_EXAMPLES=OFF \
-DBUILD_SHARED_LIBS=OFF \
-DBUILD_DOCS=OFF \
-DBUILD_TESTS=OFF \
-DBUILD_EXAMPLES=OFF \
-DBUILD_PERF_TESTS=OFF \
-DBUILD_opencv_dnn=OFF \
-DTINYDNN_USE_NNPACK=OFF \
-DTINYDNN_USE_TBB=ON \
-DTINYDNN_USE_OMP=ON \
-DENABLE_FAST_MATH=ON \
-DWITH_OPENMP=ON \
-DWITH_TBB=ON \
-DWITH_JPEG=OFF \
-DWITH_IPP=OFF \
-DMKL_WITH_TBB=ON \
-DMKL_WITH_OPENMP=ON \
-DBUILD_opencv_python2=OFF \
-DPYTHON_EXECUTABLE="/home/$USER/anaconda3/envs/tensorflow_p36/bin/python" \
-DPYTHON_LIBRARY="/home/$USER/anaconda3/envs/tensorflow_p36/lib/python3.6" \
-DPYTHON3_LIBRARY="/home/$USER/anaconda3/envs/tensorflow_p36/lib/python3.6" \
-DPYTHON3_EXECUTABLE="/home/$USER/anaconda3/envs/tensorflow_p36/bin/python" \
-DPYTHON3_INCLUDE_DIR="/home/$USER/anaconda3/envs/tensorflow_p36/include/python3.6m" \
-DPYTHON3_INCLUDE_DIR2="/home/$USER/anaconda3/envs/tensorflow_p36/include" \
-DPYTHON3_NUMPY_INCLUDE_DIRS="/home/$USER/anaconda3/envs/tensorflow_p36/lib/python3.6/site-packages/numpy/core/include" \
-DPYTHON3_INCLUDE_PATH="/home/$USER/anaconda3/envs/tensorflow_p36/include/python3.6m" \
-DPYTHON3_LIBRARIES="/home/$USER/anaconda3/envs/tensorflow_p36/lib/libpython3.6m.so" \
..
```
2. ``` make ```
3. ``` sudo make install ```
4. ``` sudo ldconfig ```

### Test the code
1. Activate conda environment
2. Go to folder: python/ and execute the cpu-opt_flow.py and gpu-opt_flow.py python files
``` 
python cpu-opt_flow.py
python gpu-opt_flow.py 
```

### Output at my end:
``` total time in optical flow CPU processing: 74.15 sec, for: 794 frames. FPS: 10.71 ```

``` total time in optical flow GPU processing: 21.98 sec, for: 794 frames. FPS: 36.12 ```

### Harware configuration:
* CPU - i7 7th Gen  
* GPU - [NVIDIA TITAN Xp](https://www.nvidia.com/en-us/titan/titan-xp)
* RAM - 32 GB
