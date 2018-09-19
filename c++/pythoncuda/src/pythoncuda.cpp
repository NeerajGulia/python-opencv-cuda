#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include <iostream>
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"

namespace cv 
{ 
    namespace pythoncuda 
    {
        CV_EXPORTS_W int by2(int n)
        {
            return n*2;
        }

        CV_EXPORTS_W void cpuOpticalFlowFarneback( InputArray prev, InputArray next, InputOutputArray flow,
                                           double pyr_scale, int levels, int winsize,
                                           int iterations, int poly_n, double poly_sigma,
                                           int flags )
        {
            cv::calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize,
                                        iterations, poly_n, poly_sigma, flags);
        }
        
        CV_EXPORTS_W void gpuOpticalFlowFarneback( InputArray prev, InputArray next, InputOutputArray flow,
                                           double pyr_scale, int levels, int winsize,
                                           int iterations, int poly_n, double poly_sigma,
                                           int flags )
        {
            cv::Ptr<cv::cuda::FarnebackOpticalFlow> farn = cv::cuda::FarnebackOpticalFlow::create();
            farn->setPyrScale(pyr_scale);
            farn->setNumLevels(levels);
            farn->setFastPyramids(false);
            farn->setWinSize(winsize);
            farn->setNumIters(iterations);
            farn->setPolyN(poly_n);
            farn->setPolySigma(poly_sigma);
            farn->setFlags(flags);

            cv::cuda::GpuMat d_flow, d_prev, d_next;
            d_prev.upload(prev);
            d_next.upload(next);
            farn->calc(d_prev, d_next, d_flow);
            d_flow.download(flow);
        }
    }
}
