#include <opencv2/core/ocl.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <QDateTime>

#include <iostream>

#include "Tools.hpp"

#include <opencv2/core/utility.hpp>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudalegacy.hpp>

#include <cuda.h>


cv::Mat                                 KernelFilter2D  = cv::Mat (cv::Size (3, 3), CV_32F, cv::Scalar (-1.0 / (3 * 3 - 1)));
cv::Mat                                 KernelDilate    = cv::Mat (cv::Size (9, 9), CV_32S, cv::Scalar (1));
cv::Mat                                 KernelMultiply;
cv::Mat                                 TemplateFrame;
double                                  Minimum;        //
double                                  Maximum;        //
cv::Point                               PointMinimum;   //
cv::Point                               PointMaximum;   //
std::vector <std::vector <cv::Point>>   Contours;


/*const*/ size_t    CicleCount      = 100/*0*/;


int main(int argc, char *argv[])
{

    cv::CommandLineParser clp (argc, argv,
                               "{ n counter | 150                | iteration count (cicle count) }"
                               "{ p path    | "TEST_IMAGES_PATH" | print help message            }"
                               "{ h help    |                    | print help message            }");

    if (clp.has ("help"))
    {
        std::cout << "Usage : bgfg_segm [options]" << std::endl;
        std::cout << "Available options:" << std::endl;

        clp.printMessage();

        return EXIT_SUCCESS;
    }

    // cudaSetDevice(0);
    // checkCudaErrors(cudaGetLastError());

    CicleCount = clp.get <size_t> ("counter");

    KernelFilter2D.at <float> (3 / 2, 3 / 2) = 1.0;

    std::string     path            = clp.get <std::string> ("path");
    std::string     framePath       = path + std::string ("/New-York-Bridge-HD-Wallpapers-images.jpg");
    std::string     templatePath    = path + std::string ("/New-York-Bridge-Template.jpg");

    /// Print frames pathes
    std::cout << std::endl;
    std::cout << "images   path " << path << std::endl;
    std::cout << "source   path " << framePath << std::endl;
    std::cout << "template path " << templatePath << std::endl;

    TemplateFrame   = cv::imread (templatePath);
    KernelMultiply  = cv::imread (framePath);


    Task::Tools::OpenClInfo();


    std::cout << "CudaEnabledDeviceCount " << cv::cuda::getCudaEnabledDeviceCount () << std::endl;

    for (int number = 0; number < cv::cuda::getCudaEnabledDeviceCount(); ++ number)
    {
        cv::cuda::printCudaDeviceInfo(number);
    }


    /* cuda */ Task::Tools::TestCvCudaGpuMat (framePath.c_str (), CicleCount, "cv::cuda::cvtColor",     [] (const cv::cuda::GpuMat & inputFrame, cv::cuda::GpuMat & outputFrame) { cv::cuda::cvtColor (inputFrame, outputFrame, CV_BGR2RGB); });
    /* cuda */ Task::Tools::TestCvCudaGpuMat (framePath.c_str (), CicleCount, "cv::cuda::threshold",    [] (const cv::cuda::GpuMat & inputFrame, cv::cuda::GpuMat & outputFrame) { cv::cuda::threshold (inputFrame, outputFrame, 45, 255, cv::THRESH_BINARY); });
    /* cuda */ Task::Tools::TestCvCudaGpuMat (framePath.c_str (), CicleCount, "cv::cuda::bitwise_or",   [] (const cv::cuda::GpuMat & inputFrame, cv::cuda::GpuMat & outputFrame) { cv::cuda::bitwise_or (inputFrame, inputFrame, outputFrame); });
    /* cuda */ Task::Tools::TestCvCudaGpuMat (framePath.c_str (), CicleCount, "cv::cuda::minMaxLoc",    [] (const cv::cuda::GpuMat & inputFrame, cv::cuda::GpuMat & /*outputFrame*/) { cv::cuda::minMaxLoc (inputFrame, &Minimum, &Maximum, &PointMinimum, &PointMaximum); }, false);
    // /* cuda */ Task::Tools::TestCvCudaGpuMat (framePath.c_str (), CicleCount, "cv::cuda::findContours", [] (const cv::cuda::GpuMat & /*inputFrame*/, cv::cuda::GpuMat & /*outputFrame*/) { /*std::cout << "cv::cuda::findContours () have not realized yet" << std::endl;*/ });
    /* cuda */ Task::Tools::TestCvCudaGpuMat (framePath.c_str (), CicleCount, "cv::cuda::multiply",     [] (const cv::cuda::GpuMat & inputFrame, cv::cuda::GpuMat & outputFrame) { cv::cuda::multiply (inputFrame, KernelMultiply, outputFrame); });


    Task::Tools::TestCvMatAndCvUMatAndCompareResults (framePath.c_str (),
                                                      CicleCount,
                                                      "cv::cvtColor",
                                                      [] (const cv::UMat & inputFrame, cv::UMat & outputFrame) { cv::cvtColor (inputFrame, outputFrame, CV_BGR2RGB); },
                                                      [] (const cv::Mat  & inputFrame, cv::Mat  & outputFrame) { cv::cvtColor (inputFrame, outputFrame, CV_BGR2RGB); });


    Task::Tools::TestCvMatAndCvUMatAndCompareResults (framePath.c_str (),
                                                      CicleCount,
                                                      "cv::filter2D",
                                                      [] (const cv::UMat & inputFrame, cv::UMat & outputFrame) { cv::filter2D (inputFrame, outputFrame, CV_32F, KernelFilter2D); },
                                                      [] (const cv::Mat  & inputFrame, cv::Mat  & outputFrame) { cv::filter2D (inputFrame, outputFrame, CV_32F, KernelFilter2D); });



    Task::Tools::TestCvMatAndCvUMatAndCompareResults (framePath.c_str (),
                                                      CicleCount,
                                                      "cv::threshold",
                                                      [] (const cv::UMat & inputFrame, cv::UMat & outputFrame) { cv::threshold (inputFrame, outputFrame, 45, 255, cv::THRESH_BINARY); },
                                                      [] (const cv::Mat  & inputFrame, cv::Mat  & outputFrame) { cv::threshold (inputFrame, outputFrame, 45, 255, cv::THRESH_BINARY); });


    Task::Tools::TestCvMatAndCvUMatAndCompareResults (framePath.c_str (),
                                                      CicleCount,
                                                      "cv::dilate",
                                                      [] (const cv::UMat & inputFrame, cv::UMat & outputFrame) { cv::dilate (inputFrame, outputFrame, KernelDilate); },
                                                      [] (const cv::Mat  & inputFrame, cv::Mat  & outputFrame) { cv::dilate (inputFrame, outputFrame, KernelDilate); });



    Task::Tools::TestCvMatAndCvUMatAndCompareResults (framePath.c_str (),
                                                      CicleCount,
                                                      "cv::bitwise_or",
                                                      [] (const cv::UMat & inputFrame, cv::UMat & outputFrame) { cv::bitwise_or (inputFrame, inputFrame, outputFrame); },
                                                      [] (const cv::Mat  & inputFrame, cv::Mat  & outputFrame) { cv::bitwise_or (inputFrame, inputFrame, outputFrame); });


    Task::Tools::TestCvMatAndCvUMatAndCompareResults (framePath.c_str (),
                                                      CicleCount,
                                                      "cv::matchTemplate",
                                                      [] (const cv::UMat & inputFrame, cv::UMat & outputFrame) { cv::matchTemplate (inputFrame, TemplateFrame, outputFrame, CV_TM_CCOEFF_NORMED); },
                                                      [] (const cv::Mat  & inputFrame, cv::Mat  & outputFrame) { cv::matchTemplate (inputFrame, TemplateFrame, outputFrame, CV_TM_CCOEFF_NORMED); });



    Task::Tools::TestCvMatAndCvUMatAndCompareResults (framePath.c_str (),
                                                      CicleCount,
                                                      "cv::minMaxLoc",
                                                      [] (const cv::UMat & inputFrame, cv::UMat & /*outputFrame*/) { cv::minMaxLoc (inputFrame, &Minimum, &Maximum, &PointMinimum, &PointMaximum); },
                                                      [] (const cv::Mat  & inputFrame, cv::Mat  & /*outputFrame*/) { cv::minMaxLoc (inputFrame, &Minimum, &Maximum, &PointMinimum, &PointMaximum); },
                                                      false);


    Task::Tools::TestCvMatAndCvUMatAndCompareResults (framePath.c_str (),
                                                      CicleCount,
                                                      "cv::findContours",
                                                      [] (const cv::UMat & inputFrame, cv::UMat & /*outputFrame*/) { cv::findContours (inputFrame, Contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); },
                                                      [] (const cv::Mat  & inputFrame, cv::Mat  & /*outputFrame*/) { cv::findContours (inputFrame, Contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); },
                                                      false);


    Task::Tools::TestCvMatAndCvUMatAndCompareResults (framePath.c_str (),
                                                      CicleCount,
                                                      "cv::multiply",
                                                      [] (const cv::UMat & inputFrame, cv::UMat & outputFrame) { cv::multiply (inputFrame, KernelMultiply, outputFrame); },
                                                      [] (const cv::Mat  & inputFrame, cv::Mat  & outputFrame) { cv::multiply (inputFrame, KernelMultiply, outputFrame); });



    std::cout << "Test is finished. Press any key ..." << std::endl;

    cv::waitKey();

    return 0;
}




