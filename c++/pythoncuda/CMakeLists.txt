set(the_description "access cuda optical flow methods from Python")

set(OPENCV_MODULE_IS_PART_OF_WORLD ON)

# ocv_define_module(infy cudaoptflow WRAP python)
ocv_define_module(pythoncuda WRAP python)

ocv_warnings_disable(CMAKE_CXX_FLAGS -Woverloaded-virtual)
