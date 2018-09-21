#ifndef __OPENCV_PYTHONCUDA_HPP__
#define __OPENCV_PYTHONCUDA_HPP__

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

namespace cv 
{ 
    namespace pythoncuda 
    {
        CV_EXPORTS_W void cpuOpticalFlowFarneback( InputArray prev, InputArray next, InputOutputArray flow,
                                           double pyr_scale, int levels, int winsize,
                                           int iterations, int poly_n, double poly_sigma,
                                           int flags );
        
        CV_EXPORTS_W void gpuOpticalFlowFarneback( InputArray prev, InputArray next, InputOutputArray flow,
                                           double pyr_scale, int levels, int winsize,
                                           int iterations, int poly_n, double poly_sigma,
                                           int flags );
        
        CV_EXPORTS_W void cpuOpticalFlowPyrLK( InputArray prevImg, InputArray nextImg,
                                            InputArray prevPts, InputOutputArray nextPts,
                                            OutputArray status, OutputArray err,
                                            Size winSize = Size(21,21), int maxLevel = 3,
                                            TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),
                                            int flags = 0, double minEigThreshold = 1e-4 );

        CV_EXPORTS_W void gpuOpticalFlowPyrLK( InputArray prevImg, InputArray nextImg,
                                            InputArray prevPts, InputOutputArray nextPts,
                                            OutputArray status, OutputArray err,
                                            Size winSize = Size(21,21), int maxLevel = 3, int iterations = 30);
    }
}
#endif /* __OPENCV_PYTHONCUDA_HPP__ */
