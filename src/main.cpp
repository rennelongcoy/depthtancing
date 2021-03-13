// FPS calculations
#include <chrono>
#include <iostream>

// Include DepthAI Utility Functions
#include "utility.hpp"

// Includes common necessary includes for development using depthai library
#include "depthai/depthai.hpp"

// Include OpenCV Library
#include <opencv2/opencv.hpp>

// mobilenet-ssd labels
static const std::vector<std::string> labelMap = {"background",  // 0
                                                  "aeroplane",   // 1
                                                  "bicycle",     // 2
                                                  "bird",        // 3
                                                  "boat",        // 4
                                                  "bottle",      // 5
                                                  "bus",         // 6
                                                  "car",         // 7
                                                  "cat",         // 8
                                                  "chair",       // 9
                                                  "cow",         // 10
                                                  "diningtable", // 11
                                                  "dog",         // 12
                                                  "horse",       // 13
                                                  "motorbike",   // 14
                                                  "person",      // 15
                                                  "pottedplant", // 16
                                                  "sheep",       // 17
                                                  "sofa",        // 18
                                                  "train",       // 19
                                                  "tvmonitor"};  // 20

// Sync frame with NN output
static bool syncNN = true;

dai::Pipeline createNNPipeline(std::string nnPath) {
    dai::Pipeline pipeline;

    // Create nodes
    std::shared_ptr<dai::node::ColorCamera> colorCam = pipeline.create<dai::node::ColorCamera>();
    std::shared_ptr<dai::node::MobileNetSpatialDetectionNetwork> spatialDetectionNetwork = pipeline.create<dai::node::MobileNetSpatialDetectionNetwork>();
    std::shared_ptr<dai::node::MonoCamera> monoLeft = pipeline.create<dai::node::MonoCamera>();
    std::shared_ptr<dai::node::MonoCamera> monoRight = pipeline.create<dai::node::MonoCamera>();
    std::shared_ptr<dai::node::StereoDepth> stereo = pipeline.create<dai::node::StereoDepth>();

    // Create XLink connections
    std::shared_ptr<dai::node::XLinkOut> xoutRgb = pipeline.create<dai::node::XLinkOut>();
    std::shared_ptr<dai::node::XLinkOut> xoutNN = pipeline.create<dai::node::XLinkOut>();
    std::shared_ptr<dai::node::XLinkOut> xoutBoundingBoxDepthMapping = pipeline.create<dai::node::XLinkOut>();
    std::shared_ptr<dai::node::XLinkOut> xoutDepth = pipeline.create<dai::node::XLinkOut>();

    // Set names for the XLinkOut nodes
    xoutRgb->setStreamName("preview");
    xoutNN->setStreamName("detections");
    xoutBoundingBoxDepthMapping->setStreamName("boundingBoxDepthMapping");
    xoutDepth->setStreamName("depth");

    // Settings of ColorCamera node
    colorCam->setPreviewSize(300, 300);
    colorCam->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
    colorCam->setInterleaved(false);
    colorCam->setColorOrder(dai::ColorCameraProperties::ColorOrder::BGR);

    // Settings of MonoCamera nodes
    monoLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
    monoLeft->setBoardSocket(dai::CameraBoardSocket::LEFT);
    monoRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
    monoRight->setBoardSocket(dai::CameraBoardSocket::RIGHT);

    // Settings of StereoDepth node
    stereo->setOutputDepth(true);
    stereo->setConfidenceThreshold(255);

    // Settings of MobileNetSpatialDetectionNetwork node
    spatialDetectionNetwork->setBlobPath(nnPath);
    spatialDetectionNetwork->setConfidenceThreshold(0.5f);
    spatialDetectionNetwork->input.setBlocking(false);
    spatialDetectionNetwork->setBoundingBoxScaleFactor(0.5);
    spatialDetectionNetwork->setDepthLowerThreshold(100);
    spatialDetectionNetwork->setDepthUpperThreshold(5000);

    // Link MonoCamera outputs to StereoDepth input
    monoLeft->out.link(stereo->left);
    monoRight->out.link(stereo->right);

    // Link ColorCamera output to MobileNetSpatialDetectionNetwork input
    colorCam->preview.link(spatialDetectionNetwork->input);
    if (syncNN) {
        // If need to sync, allow passthrough of ColorCamera frames
        spatialDetectionNetwork->passthrough.link(xoutRgb->input);
    }
    else {
        // If sync not needed, connect ColorCamera directly to XLinkOut
        colorCam->preview.link(xoutRgb->input);
    }

    // Link MobileNetSpatialDetectionNetwork output to XLinkOut input
    spatialDetectionNetwork->out.link(xoutNN->input);
    spatialDetectionNetwork->boundingBoxMapping.link(xoutBoundingBoxDepthMapping->input);

    // Link SteroDepth output to MobileNetSpatialDetectionNetwork input to allow depth calculations
    stereo->depth.link(spatialDetectionNetwork->inputDepth);
    spatialDetectionNetwork->passthroughDepth.link(xoutDepth->input);

    return pipeline;
}

int main(int argc, char* argv[]) {
    std::string nnPath("/home/eli/apps/depthtancing/mobilenet-ssd/mobilenet-ssd.blob");

    // If path to blob specified, use that
    if (argc > 1) {
        nnPath = std::string(argv[1]);
    }

    // Print which blob is used
    std::cout << "Using blob at path: " << nnPath << std::endl;

    // Create pipeline
    dai::Pipeline pipeline = createNNPipeline(nnPath);

    // Connect to device with above created pipeline
    dai::Device device(pipeline);

    // Start the pipeline
    device.startPipeline();

    // Define output queues based on XLinkOut nodes
    std::shared_ptr<dai::DataOutputQueue> preview = device.getOutputQueue("preview", 4, false);
    std::shared_ptr<dai::DataOutputQueue> detections = device.getOutputQueue("detections", 4, false);
    std::shared_ptr<dai::DataOutputQueue> xoutBoundingBoxDepthMapping = device.getOutputQueue("boundingBoxDepthMapping", 4, false);
    std::shared_ptr<dai::DataOutputQueue> depthQueue = device.getOutputQueue("depth", 4, false);

    // FPS-relatd variables
    std::chrono::_V2::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    int counter = 0;
    float fps = 0;

    // Text Color
    cv::Scalar color(255, 255, 255);

    while (true) {
        // Obtain packets form queues
        std::shared_ptr<dai::ImgFrame> imgFrame = preview->get<dai::ImgFrame>();
        std::shared_ptr<dai::SpatialImgDetections> det = detections->get<dai::SpatialImgDetections>();
        std::shared_ptr<dai::ImgFrame> depth = depthQueue->get<dai::ImgFrame>();

        // Obtain detections
        std::vector<dai::SpatialImgDetection> detections = det->detections;

        // Update FPS measurement
        counter++;
        std::chrono::_V2::steady_clock::time_point currentTime = std::chrono::steady_clock::now();
        std::chrono::duration<float> elapsed = std::chrono::duration_cast<std::chrono::duration<float>>(currentTime - startTime);
        // Count number of processed frames in 1 second
        if(elapsed > std::chrono::seconds(1)) {
            fps = (float)counter / elapsed.count();
            counter = 0;
            startTime = currentTime;
        }

        // Convert data from queue to OpenCV format
        // CxHxW = 3x300x300
        // C = data_packet->dimensions[0] = 3
        // H = data_packet->dimensions[1] = 300
        // W = data_packet->dimensions[2] = 300

        // HxWxC = 300x300x3
        // H = frame.rows       = 300
        // W = frame.cols       = 300
        // C = frame.channels() = 3
        cv::Mat frame;
        if (imgFrame) {
            frame = toMat(imgFrame->getData(), imgFrame->getWidth(), imgFrame->getHeight(), 3, 1);
        }

        for (const auto& detection : detections) {
            int x1 = detection.xmin * frame.cols;
            int y1 = detection.ymin * frame.rows;
            int x2 = detection.xmax * frame.cols;
            int y2 = detection.ymax * frame.rows;

            int labelIndex = detection.label;
            std::string labelStr = std::to_string(labelIndex);
            if(labelIndex < labelMap.size()) {
                labelStr = labelMap[labelIndex];
            }
            cv::putText(frame, labelStr, cv::Point(x1 + 10, y1 + 20), cv::FONT_HERSHEY_TRIPLEX, 0.5, color);
            std::stringstream confStr;
            confStr << std::fixed << std::setprecision(2) << detection.confidence*100;
            cv::putText(frame, confStr.str(), cv::Point(x1 + 10, y1 + 35), cv::FONT_HERSHEY_TRIPLEX, 0.5, color);

            std::stringstream depthX;
            depthX << "X: " << (int)detection.spatialCoordinates.x << " mm";
            cv::putText(frame, depthX.str(), cv::Point(x1 + 10, y1 + 50), cv::FONT_HERSHEY_TRIPLEX, 0.5, color);
            std::stringstream depthY;
            depthY << "Y: " << (int)detection.spatialCoordinates.y << " mm";
            cv::putText(frame, depthY.str(), cv::Point(x1 + 10, y1 + 65), cv::FONT_HERSHEY_TRIPLEX, 0.5, color);
            std::stringstream depthZ;
            depthZ << "Z: " << (int)detection.spatialCoordinates.z << " mm";
            cv::putText(frame, depthZ.str(), cv::Point(x1 + 10, y1 + 80), cv::FONT_HERSHEY_TRIPLEX, 0.5, color);
            
            cv::rectangle(frame, cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)), cv::Scalar(0, 255, 0), cv::FONT_HERSHEY_SIMPLEX);
        }

        // Display FPS measurement
        std::stringstream fpsStr;
        fpsStr << std::fixed << std::setprecision(2) << fps;
        cv::putText(frame, fpsStr.str(), cv::Point(2, imgFrame->getHeight()-4), cv::FONT_HERSHEY_TRIPLEX, 0.4, color);

        // Display frame
        cv::Mat resizedFrame;
        cv::resize(frame, resizedFrame, cv::Size(900, 900));
        cv::imshow("Depthtancing v0.01", resizedFrame);
        int key = cv::waitKey(1);
        if (key == 'q') {
            return 0;
        }
    }

    return 0;
}