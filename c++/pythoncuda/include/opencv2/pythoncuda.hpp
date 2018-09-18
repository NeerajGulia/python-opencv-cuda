#ifndef __OPENCV_PYTHONCUDA_HPP__
#define __OPENCV_PYTHONCUDA_HPP__

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

namespace cv 
{ 
    namespace pythoncuda 
    {
        CV_EXPORTS_W int by2(int n);
        CV_EXPORTS_W void cpuOpticalFlowFarneback( InputArray prev, InputArray next, InputOutputArray flow,
                                           double pyr_scale, int levels, int winsize,
                                           int iterations, int poly_n, double poly_sigma,
                                           int flags );
        CV_EXPORTS_W void gpuOpticalFlowFarneback( InputArray prev, InputArray next, InputOutputArray flow,
                                           double pyr_scale, int levels, int winsize,
                                           int iterations, int poly_n, double poly_sigma,
                                           int flags )
                                           
    }
}
#endif /* __OPENCV_PYTHONCUDA_HPP__ */
