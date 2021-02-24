#include <iostream>
#include <chrono>

// Inludes common necessary includes for development using depthai library
#include <depthai/device.hpp>

#include <opencv2/opencv.hpp>

// include depthai-utility
#include "utility.hpp"

int main(int argc, char** argv) {
    std::cout << "-- START --" << std::endl;
    cv::Mat frame;

    // Create DepthAI Device
    Device device("", false);

    // Create Pipeline
    //std::string config_json_str = "{\"streams\": [\"metaout\", \"previewout\"],\"ai\": {\"calc_dist_to_bb\": true,\"blob_file\": \"/home/eli/apps/depthtancing/person-detection-retail-0013/model.blob\",\"blob_file_config\": \"/home/eli/apps/depthtancing/person-detection-retail-0013/config.json\"}}";
    std::string config_json_str = "{\"streams\": [\"metaout\", \"previewout\"],\"ai\": {\"calc_dist_to_bb\": true,\"blob_file\": \"/home/eli/apps/depthtancing/mobilenet-ssd/model.blob\",\"blob_file_config\": \"/home/eli/apps/depthtancing/mobilenet-ssd/config.json\"}}";
    std::shared_ptr<CNNHostPipeline> pipeline = device.create_pipeline(config_json_str);
    while(true) {
        std::tuple<std::list<std::shared_ptr<NNetPacket>>, std::list<std::shared_ptr<HostDataPacket>>> packets = pipeline->getAvailableNNetAndDataPackets();

        std::list<std::shared_ptr<NNetPacket>> nnet_packets = std::get<0>(packets);
        std::list<std::shared_ptr<HostDataPacket>> data_packets = std::get<1>(packets);

        //std::cout << "A" << nnet_packets.size() << std::endl;
        for (std::shared_ptr<NNetPacket> nnet_packet : nnet_packets) {
            std::shared_ptr<dai::Detections> detectedObjects = nnet_packet->getDetectedObjects();
            for (int i = 0; i < detectedObjects->detection_count; ++i) {
                std::cout << "Detection: " << detectedObjects->detections[i].label << "|" << detectedObjects->detections[i].depth_x << "|" << detectedObjects->detections[i].depth_y << "|" << detectedObjects->detections[i].depth_z << std::endl;
            }
        }

        for (std::shared_ptr<HostDataPacket> data_packet : data_packets) {
            if (data_packet->stream_name == "previewout") {
                //const unsigned char* data = data_packet->getData();
                // CxHxW = 3x320x544
                //std::cout << "test " << data_packet->dimensions[0] << "x" << data_packet->dimensions[1] << "x" << data_packet->dimensions[2] << std::endl;
                frame = toMat(*(data_packet->data), data_packet->dimensions[2], data_packet->dimensions[1], 3, 1);
                //frame = toMat(data_packet->getData(), data_packet->dimensions[2], data_packet->dimensions[1], 3, 1);
                //std::cout << frame.rows << "x" << frame.cols << "x" << frame.channels() << std::endl;
                cv::imshow("preview", frame);
                // Wait and check if 'q' pressed
                if (cv::waitKey(1) == 'q') {
                    return 0;
                }
            }
        }
    }

    std::cout << "-- END --" << std::endl;
    return 0;
}