#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <opencv2/core/ocl.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <functional>

#include <QDateTime>

#include <opencv2/cudaarithm.hpp>

namespace Task
{
class Tools
{
public:
    static void OpenClInfo ()
    {
        std::cout << ""                                         << std::endl;
        std::cout << "Have Open CL  " << cv::ocl::haveOpenCL()  << std::endl;
        std::cout << "Use  Open CL  " << cv::ocl::useOpenCL()   << std::endl;
        std::cout << "Have AMD Blas " << cv::ocl::haveAmdBlas() << std::endl;
        std::cout << "Have AMD FFT  " << cv::ocl::haveAmdFft()  << std::endl;
        std::cout << "Have SVM      " << cv::ocl::haveSVM()     << std::endl;

        /// Display devices
        if (cv::ocl::haveOpenCL ())
        {
            cv::ocl::setUseOpenCL (true);

            cv::ocl::Context context = cv::ocl::Context::getDefault ();

            std::cout << "" << std::endl;
            std::cout << context.ndevices () << " GPU (Open CL) devices are detected." << std::endl;

            for (size_t index = 0; index < context.ndevices (); index++)
            {
                cv::ocl::Device device = context.device (index);

                std::cout << "" << std::endl;
                std::cout << "OpenCL device Name            :" << device.name ().c_str ()<< std::endl;
                std::cout << "OpenCL device Available       :" << device.available ()<< std::endl;
                std::cout << "OpenCL device ImageSupport    :" << device.imageSupport ()<< std::endl;
                std::cout << "OpenCL device OpenCL C Version:" << device.OpenCL_C_Version().c_str ()<< std::endl;
                std::cout << "OpenCL device OpenCL Version  :" << device.OpenCLVersion ().c_str ()<< std::endl;
                std::cout << "OpenCL device Driver Version  :" << device.driverVersion ().c_str ()<< std::endl;
                std::cout << "OpenCL device Version         :" << device.version ().c_str ()<< std::endl;
            }

            std::cout << "" << std::endl;
            std::cout << "Default OpenCL device Name            :" << cv::ocl::Device::getDefault ().name().c_str()<< std::endl;
            std::cout << "Default OpenCL device Available       :" << cv::ocl::Device::getDefault ().available()<< std::endl;
            std::cout << "Default OpenCL device ImageSupport    :" << cv::ocl::Device::getDefault ().imageSupport()<< std::endl;
            std::cout << "Default OpenCL device OpenCL_C_Version:" << cv::ocl::Device::getDefault ().OpenCL_C_Version ().c_str()<< std::endl;
            std::cout << "Default OpenCL device OpenCL Version  :" << cv::ocl::Device::getDefault ().OpenCLVersion ().c_str ()<< std::endl;
            std::cout << "Default OpenCL device Driver Version  :" << cv::ocl::Device::getDefault ().driverVersion ().c_str ()<< std::endl;
            std::cout << "Default OpenCL device Version         :" << cv::ocl::Device::getDefault ().version ().c_str ()<< std::endl;
            std::cout << "" << std::endl;
        }
        else
        {
            std::cout << ""                             << std::endl;
            std::cout << "OpenCL is not available..."   << std::endl;
        }

        std::cout << "" << std::endl;
    }

    static cv::UMat TestCvUMat (const char * framePath, size_t cicleNumber, const char * functionName, const std::function <void (const cv::UMat &, cv::UMat &)>& function, bool frameIsColor = true)
    {
        cv::Mat readFrame = cv::imread (framePath, frameIsColor ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

        cv::UMat frame;
        cv::UMat processedFrame;

        readFrame.copyTo (frame);

        int64 ticks = cv::getTickCount ();

        for (size_t number = 0; number < cicleNumber; ++ number)
        {
            function (frame, processedFrame);
        }

        double time = ((double) cv::getTickCount () - ticks) / cv::getTickFrequency ();

        std::cout << "Test"         << " "
                  << functionName   << " "
                  << "cicle count"  << " "
                  << cicleNumber    << " "
                  << "cv::UMat"     << " "
                  << QTime::fromMSecsSinceStartOfDay ((int64)(time * 1000.0)).toString("hh:mm:ss.zzz").toStdString ()
                  << std::endl;

        return processedFrame;

        // cv::imshow("Processed cv::UMat frame", processedFrame);
    }

    static cv::Mat  TestCvMat  (const char * framePath, size_t cicleNumber, const char * functionName, const std::function <void (const cv::Mat &, cv::Mat &)>&   function, bool frameIsColor = true)
    {
        cv::Mat readFrame = cv::imread (framePath, frameIsColor ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

        cv::Mat processedFrame;

        int64 ticks = cv::getTickCount ();

        for (size_t number = 0; number < cicleNumber; ++ number)
        {
            function (readFrame, processedFrame);
        }

        double time = ((double) cv::getTickCount () - ticks) / cv::getTickFrequency ();

        std::cout << "Test"         << " "
                  << functionName   << " "
                  << "cicle count"  << " "
                  << cicleNumber    << " "
                  << " cv::Mat"     << " "
                  << QTime::fromMSecsSinceStartOfDay ((int64)(time * 1000.0)).toString ("hh:mm:ss.zzz").toStdString ()
                  << std::endl;

        return processedFrame;

        // cv::imshow("Processed  cv::Mat frame", processedFrame);
    }

    static cv::cuda::GpuMat TestCvCudaGpuMat (const char * framePath,
                                              size_t cicleNumber,
                                              const char * functionName,
                                              const std::function <void (const cv::cuda::GpuMat &, cv::cuda::GpuMat &)>& function,
                                              bool frameIsColor = true)
    {
        cv::Mat readedFrame = cv::imread (framePath, frameIsColor ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

        cv::cuda::GpuMat cudaFrame (readedFrame);
        cv::cuda::GpuMat cudaProcessedFrame;

        // readedFrame.copyTo (cudaFrame);

        int64 ticks = cv::getTickCount ();

        for (size_t number = 0; number < cicleNumber; ++ number)
        {
            function (cudaFrame, cudaProcessedFrame);
        }

        double time = ((double) cv::getTickCount () - ticks) / cv::getTickFrequency ();

        std::cout << "Test"             << " "
                  << functionName       << " "
                  << "cicle count"      << " "
                  << cicleNumber        << " "
                  << "cv::cuda::GpuMat" << " "
                  << QTime::fromMSecsSinceStartOfDay ((int64)(time * 1000.0)).toString("hh:mm:ss.zzz").toStdString ()
                  << std::endl;

        return cudaProcessedFrame;

        // cv::imshow("Processed cv::UMat frame", processedFrame);
    }

    static void TestCvMatAndCvUMatAndCompareResults (const char *   framePath,
                                                     size_t         cicleNumber,
                                                     const char *   functionName,
                                                     const std::function <void (const cv::UMat &, cv::UMat &)>& cvUMatFunction,
                                                     const std::function <void (const cv::Mat  &, cv::Mat  &)>&  cvMatFunction,
                                                     bool           frameIsColor = true)
    {
        cv::Mat convertedFrame;

        cv::UMat uMatFrame = TestCvUMat (framePath, cicleNumber, functionName, cvUMatFunction, frameIsColor);
        cv::Mat   matFrame = TestCvMat  (framePath, cicleNumber, functionName,  cvMatFunction, frameIsColor);

        uMatFrame.copyTo (convertedFrame);

        if (convertedFrame.type ()     == matFrame.type () &&
                convertedFrame.channels () == matFrame.channels ())
        {
            // for (int row = 0; row < convertedFrame.rows; ++ row)
            // {
            //     for (int column = 0; column < convertedFrame.cols; ++ column)
            //     {
            //         if (convertedFrame.at <double> (row, column) == convertedFrame.at <double> (row, column))
            //         {
            //         }
            //         else
            //         {
            //             std::cout << "Pixel don`t equal, possition is { row "
            //                       << row
            //                       << ", column " << column << "}"
            //                       << ""
            //                       << std::endl;
            //         }
            //     }
            // }
        }
        else
        {
            std::cout << "Type of cv::UMat doesn`t equal type of cv::Mat." << std::endl;
            std::cout << "Type of cv::UMat " << uMatFrame.type () << std::endl;
            std::cout << "Type of cv::Mat  " <<  matFrame.type () << std::endl;
        }

        std::cout << std::endl;
    }


    static void TestCvUMat (const char *                                    framePath,
                            const char *                                    functionName,
                            size_t                                          cicleNumber,
                            const std::function <void (void)>&              initFunction,
                            const std::function <void (const cv::UMat &)>&  processingFunction,
                            const std::function <void (void)>&              debugFunction,
                            bool                                            frameIsColor = true)
    {
        cv::UMat frame;
        cv::Mat readFrame = cv::imread (framePath, frameIsColor ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

        readFrame.copyTo (frame);

        initFunction ();


        int64 ticks = cv::getTickCount ();

        for (size_t number = 0; number < cicleNumber; ++ number)
        {
            processingFunction (frame/*, processedFrame*/);
        }

        double time = ((double) cv::getTickCount () - ticks) / cv::getTickFrequency ();


        std::cout << "Test"         << " "
                  << functionName   << " "
                  << "cicle count"  << " "
                  << cicleNumber    << " "
                  << "cv::UMat"     << " "
                  << QTime::fromMSecsSinceStartOfDay ((int64)(time * 1000.0)).toString("hh:mm:ss.zzz").toStdString ()
                  << std::endl;

        debugFunction ();

        std::cout << std::endl;
    }

    static void TestCvMat  (const char *                                    framePath,
                            const char *                                    functionName,
                            size_t                                          cicleNumber,
                            const std::function <void (void)>&              initFunction,
                            const std::function <void (const cv::Mat &)>&   processingFunction,
                            const std::function <void (void)>&              debugFunction,
                            bool                                            frameIsColor = true)
    {
        cv::Mat readFrame = cv::imread (framePath, frameIsColor ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

        initFunction ();


        int64 ticks = cv::getTickCount ();

        for (size_t number = 0; number < cicleNumber; ++ number)
        {
            processingFunction (readFrame/*, processedFrame*/);
        }

        double time = ((double) cv::getTickCount () - ticks) / cv::getTickFrequency ();


        std::cout << "Test"         << " "
                  << functionName   << " "
                  << "cicle count"  << " "
                  << cicleNumber    << " "
                  << " cv::Mat"     << " "
                  << QTime::fromMSecsSinceStartOfDay ((int64)(time * 1000.0)).toString ("hh:mm:ss.zzz").toStdString ()
                  << std::endl;

        debugFunction ();

        std::cout << std::endl;
    }
};
}

// using TestCvUMat = Task::Tools::TestCvUMat;
// using TestCvMat  = Task::Tools::TestCvMat;

#endif // TOOLS_HPP
