set(the_description "access opencv cuda methods from Python")

ocv_define_module(pythoncuda opencv_core opencv_imgproc opencv_tracking opencv_cudaoptflow opencv_cudaarithm opencv_cudaimgproc WRAP python)