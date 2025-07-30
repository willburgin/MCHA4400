#include <cstdlib>
#include <cstdio>
#include <string>
#include <sstream>
#include <print>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <nanobench.h>
#include "imagefeatures.h"

int main(int argc, char *argv[])
{
    cv::String keys = 
        // Argument names | defaults | help message
        "{help h usage ?  |          | print this message}"
        "{@input          | <none>   | input can be a path to an image or video (e.g., ../data/lab.jpg)}"
        "{export e        |          | export output file to the ./out/ directory}"
        "{N               | 10       | maximum number of features to find}"
        "{detector d      | fast     | feature detector to use (e.g., harris, shi, aruco, fast)}"
        "{benchmark b     |          | run benchmark for all detectors}"
    ;
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("MCHA4400 Lab 2");

    if (parser.has("help"))
    {
        parser.printMessage();
        return EXIT_SUCCESS;
    }

    // Parse input arguments
    bool doExport = parser.has("export");
    int maxNumFeatures = parser.get<int>("N");
    bool doBenchmark = parser.has("benchmark");
    cv::String detector = parser.get<std::string>("detector");
    std::filesystem::path inputPath = parser.get<std::string>("@input");

    // Check for syntax errors
    if (!parser.check())
    {
        parser.printMessage();
        parser.printErrors();
        return EXIT_FAILURE;
    }

    if (!std::filesystem::exists(inputPath))
    {
        std::println("File: {} does not exist", inputPath.string());
        return EXIT_FAILURE;
    }

    // Prepare output directory
    std::filesystem::path outputDirectory;
    if (doExport)
    {
        std::filesystem::path appPath = parser.getPathToApplication();
        outputDirectory = appPath / ".." / "out";

        // Create output directory if we need to
        if (!std::filesystem::exists(outputDirectory))
        {
            std::println("Creating directory {}", outputDirectory.string());
            std::filesystem::create_directory(outputDirectory);
        }
        std::println("Output directory set to {}", outputDirectory.string());
    }

    // Prepare output file path
    std::filesystem::path outputPath;
    if (doExport)
    {
        std::string outputFilename = inputPath.stem().string()
                                   + "_"
                                   + detector
                                   + inputPath.extension().string();
        outputPath = outputDirectory / outputFilename;
        std::println("Output name: {}", outputPath.string());
    }

    // Check if input is an image or video (or neither)
    bool isVideo = false;
    bool isImage = false;

    std::vector<std::string> image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"};
    std::vector<std::string> video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"};

    std::string ext = inputPath.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);  // make extension lowercase

    isImage = std::find(image_exts.begin(), image_exts.end(), ext) != image_exts.end();
    isVideo = std::find(video_exts.begin(), video_exts.end(), ext) != video_exts.end();


    if (!isImage && !isVideo)
    {
        std::println("Could not read file: {}", inputPath.string());
        return EXIT_FAILURE;
    }

    if (doBenchmark)
    {
        if (!isImage)
        {
            std::println("Benchmark can only be run on images, not videos.");
            return EXIT_FAILURE;
        }

        // Suppress console output during benchmark
        std::FILE* old_stdout = stdout;
        stdout = std::fopen("/dev/null", "w");

        // Create benchmark object
        ankerl::nanobench::Bench bench;
        
        // Capture benchmark output in a separate stringstream
        std::stringstream bench_output;
        bench.output(&bench_output);

        // TODO: Run the benchmarks for the 4 feature detectors IGNORE THIS FOR NOW!!!
        if (doBenchmark)
        {
            cv::Mat data = cv::imread(inputPath.string(), cv::IMREAD_COLOR);
            bench.performanceCounters(false);
            bench.minEpochIterations(200);
            bench.relative(true);
            bench.title("Benchmark results");

            bench.run("Harris", [&]
            {
                cv::Mat out = detectAndDrawHarris(data, maxNumFeatures);
                bench.doNotOptimizeAway(out);
            });

            bench.run("Shi-Tomasi", [&]
            {
                cv::Mat out = detectAndDrawShiAndTomasi(data, maxNumFeatures);
                bench.doNotOptimizeAway(out);
            });

            bench.run("FAST", [&]
            {
                cv::Mat out = detectAndDrawFAST(data, maxNumFeatures);
                bench.doNotOptimizeAway(out);
            });
            bench.run("ArUco", [&]
            {
                cv::Mat out = detectAndDrawArUco(data, maxNumFeatures);
                bench.doNotOptimizeAway(out);
            });
        }

        // Restore console output
        std::fclose(stdout);
        stdout = old_stdout;

        // Print benchmark results
        std::println("\nBenchmark results:");
        std::print("{}", bench_output.str());

        return EXIT_SUCCESS;
    }

    if (isImage)
    {
        cv::Mat inputImage = cv::imread(inputPath.string());
        if (inputImage.empty()) 
        {
            std::println("Error loading image: {}", inputPath.string());
            return EXIT_FAILURE;
        }
        cv::Mat outputImage;

        // TODO: Call one of the detectAndDraw functions from imagefeatures.cpp according to the detector option specified at the command line
        if (detector == "harris")
        {
            outputImage = detectAndDrawHarris(inputImage, maxNumFeatures);
        }
        else if (detector == "shi")
        {
            outputImage = detectAndDrawShiAndTomasi(inputImage, maxNumFeatures);
        }
        else if (detector == "fast")
        {
            outputImage = detectAndDrawFAST(inputImage, maxNumFeatures);
        }
        else if (detector == "aruco")
        {
            outputImage = detectAndDrawArUco(inputImage, maxNumFeatures);
        }
        else 
        {
            std::println("No support for other detectors right now.");
            return EXIT_FAILURE;
        }
        if (doExport)
        {
            // TODO: Write image returned from detectAndDraw to outputPath
            cv::imwrite(outputPath.string(), outputImage);
            std::println("Exported output to {}", outputPath.string());
        }
        else
        {
            // TODO: Display image returned from detectAndDraw on screen and wait for keypress
            cv::imshow("Feature Detection Result", outputImage);
            cv::waitKey(0);
        }
    }

    if (isVideo)
    {
        cv::VideoCapture cap(inputPath.string());
        if (!cap.isOpened()) 
        {
            std::println("Error: Failed to open video file: {}", inputPath.string());
            return EXIT_FAILURE;
        }

        cv::VideoWriter writer;           
        double fps = cap.get(cv::CAP_PROP_FPS);
        if (doExport) 
        {
            // TODO: Open output video for writing using the same fps as the input video
            //       and the codec set to cv::VideoWriter::fourcc('m', 'p', '4', 'v')
            int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
            int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

            writer.open(outputPath.string(),
                        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                        fps,
                        cv::Size(width, height));

            if (!writer.isOpened()) 
            {
                std::println("Error: Could not open output video file: {}", outputPath.string());
                return EXIT_FAILURE;
            }
        }

        while (true)
        {
            // TODO: Get next frame from input video
            cv::Mat frame;
            cap >> frame;

            // TODO: If frame is empty, break out of the while loop
            if (frame.empty()) 
            {
                break;
            }
            
            // TODO: Call one of the detectAndDraw functions from imagefeatures.cpp according to the detector option specified at the command line
            cv::Mat outputFrame;
            if (detector == "harris")
            {
                outputFrame = detectAndDrawHarris(frame, maxNumFeatures);
            }
            else if (detector == "shi")
            {
                outputFrame = detectAndDrawShiAndTomasi(frame, maxNumFeatures);
            } 
            else if (detector == "fast")
            {
                outputFrame = detectAndDrawFAST(frame, maxNumFeatures);
            }
            else if (detector == "aruco")
            {
                outputFrame = detectAndDrawArUco(frame, maxNumFeatures);
            }
            else
            {
                std::println("No support for other detections right now.");
                return EXIT_FAILURE;
            }

            if (doExport)
            {
                // TODO: Write image returned from detectAndDraw to frame of output video
                writer.write(outputFrame);
            }
            else
            {
                // TODO: Display image returned from detectAndDraw on screen and wait for 1000/fps milliseconds
                cv::imshow("Video Feature Detection", outputFrame);
                int delay = static_cast<int>(1000.0 / fps);
                if (cv::waitKey(delay) == 27) 
                {
                    break;
                }
            }
        }

        // TODO: release the input video object
        cap.release();

        if (doExport)
        {
            // TODO: release the output video object
            writer.release();
        }
    }

    return EXIT_SUCCESS;
}



