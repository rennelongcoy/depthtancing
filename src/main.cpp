#include <iostream>

// Inludes DepthAI Device Library
#include <depthai/device.hpp>

// include DepthAI Utility Functions
#include "utility.hpp"

// Include OpenCV Library
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    // Create DepthAI Device
    Device device("", false);

    // Create Device Pipeline based on JSON config file
    // person-detection-retail-0013
    // std::string config_json_str = "{\"streams\": [\"metaout\", \"previewout\"],\"ai\": {\"calc_dist_to_bb\": true,\"blob_file\": \"/home/eli/apps/depthtancing/person-detection-retail-0013/model.blob\",\"blob_file_config\": \"/home/eli/apps/depthtancing/person-detection-retail-0013/config.json\"}}";
    // mobilenet-ssd
    std::string config_json_str = "{\"streams\": [\"metaout\", \"previewout\"],\"ai\": {\"calc_dist_to_bb\": true,\"blob_file\": \"/home/eli/apps/depthtancing/mobilenet-ssd/model.blob\",\"blob_file_config\": \"/home/eli/apps/depthtancing/mobilenet-ssd/config.json\"}}";
    std::shared_ptr<CNNHostPipeline> pipeline = device.create_pipeline(config_json_str);

    // Continuously get packets from pipeline
    while (true) {
        // Get tuple (NNetPacket, HostDataPacket) from Pipeline
        std::tuple<std::list<std::shared_ptr<NNetPacket>>, std::list<std::shared_ptr<HostDataPacket>>> packets = pipeline->getAvailableNNetAndDataPackets();

        // Get 1st element of tuple (NNetPacket, HostDataPacket)
        std::list<std::shared_ptr<NNetPacket>> nnet_packets = std::get<0>(packets);

        // Get 2nd element of tuple (NNetPacket, HostDataPacket)
        std::list<std::shared_ptr<HostDataPacket>> data_packets = std::get<1>(packets);

        std::vector<std::unordered_map<std::string, float>> boxes;
        std::vector<std::unordered_map<std::string, float>> depths;

        // For the retrieved nnet_packet,
        for (std::shared_ptr<NNetPacket> &nnet_packet : nnet_packets) { // nnet_packets.size() = 1
            // Get array of detected objects (maximum = 100)
            std::shared_ptr<dai::Detections> network_results = nnet_packet->getDetectedObjects();

            // Get properties of detected objects
                for (int i = 0; i < network_results->detection_count; ++i) {
                    /*boxes[i].insert({"id", i});
                    boxes[i].insert({"label", network_results->detections[i].label});
                    boxes[i].insert({"confidence", network_results->detections[i].confidence});
                    boxes[i].insert({"x_min", network_results->detections[i].x_min});
                    boxes[i].insert({"y_min", network_results->detections[i].y_min});
                    boxes[i].insert({"x_max", network_results->detections[i].x_max});
                    boxes[i].insert({"y_max", network_results->detections[i].y_max});*/
                    std::unordered_map<std::string, float> box;
                    box["id"] = i;
                    box["label"] = network_results->detections[i].label;
                    box["confidence"] = network_results->detections[i].confidence;
                    box["x_min"] = network_results->detections[i].x_min;
                    box["y_min"] = network_results->detections[i].y_min;
                    box["x_max"] = network_results->detections[i].x_max;
                    box["y_max"] = network_results->detections[i].y_max;
                    boxes.push_back(box);

                    /*std::unordered_map<std::string, float> depth;
                    depth["id"] = i;
                    depth["depth_x"] = network_results->detections[i].depth_x;
                    depth["depth_y"] = network_results->detections[i].depth_y;
                    depth["depth_z"] = network_results->detections[i].depth_z;
                    depths.push_back(depth);
                    */
                    std::cout << "[i=" << i << "] "
                    << "Label = " << network_results->detections[i].label
                    << " | confidence = " << network_results->detections[i].confidence
                    << " | x_min = " << network_results->detections[i].x_min
                    << " | y_min = " << network_results->detections[i].y_min
                    << " | x_max = " << network_results->detections[i].x_max
                    << " | y_max = " << network_results->detections[i].y_max
                    << " | depth_x = " << network_results->detections[i].depth_x
                    << " | depth_y = " << network_results->detections[i].depth_y
                    << " | depth_z = " << network_results->detections[i].depth_z
                    << std::endl;
                }
                std::cout << "boxes.size() C = " << boxes.size() << std::endl;
        }
        //std::cout << "boxes.size() D = " << boxes.size() << std::endl;

        // Create cv::Mat object to hold data_packet from Pipeline
        cv::Mat frame;
        for (std::shared_ptr<HostDataPacket> data_packet : data_packets) {
            if (data_packet->stream_name == "previewout") {
                // If no data, continue to next packet
                //if (data_packet->getData() == nullptr) {
                //    continue;
                //}

                // DepthAI HostDataPacket format
                // CxHxW = 3x320x544
                // C = data_packet->dimensions[0] = 3
                // H = data_packet->dimensions[1] = 320
                // W = data_packet->dimensions[2] = 544

                // Convert to OpenCV cv::Mat format
                // HxWxC = 320x544x3
                // H = frame.rows       = 320
                // W = frame.cols       = 544
                // C = frame.channels() = 3
                frame = toMat(*(data_packet->data), data_packet->dimensions[2], data_packet->dimensions[1], 3, 1);

                // Calculate bounding box locations (in pixels)
                /*std::vector<std::unordered_map<std::string, int>> boxes;
                for (int i = 0; i < network_results->detection_count; ++i) {
                    boxes[i]["id"] = i;
                    boxes[i]["x_min"] = static_cast<int>(network_results->detections[i].x_min * frame.cols);
                    boxes[i]["y_min"] = static_cast<int>(network_results->detections[i].y_min * frame.rows);
                    boxes[i]["x_max"] = static_cast<int>(network_results->detections[i].x_max * frame.cols);
                    boxes[i]["y_max"] = static_cast<int>(network_results->detections[i].y_max * frame.rows);
                }

                // Draw bounding box on the frame in-place
                for (int i = 0; i < network_results->detection_count; ++i) {
                    cv2::rectangle(frame, cv::Point(boxes[i]['x_min'], boxes[i]['y_min']), cv::Point(boxes[i]['x_max'], boxes[i]['y_max']), cv::Scalar(0, 255, 0), 2)
                }*/
                //std::cout << boxes.size() << std::endl;
                for (int i = 0; i < boxes.size(); ++i) {
                    std::cout << "TEST" << std::endl;
                    cv::rectangle(frame, cv::Point(boxes[i]["x_min"]*frame.cols, boxes[i]["y_min"]*frame.rows), cv::Point(boxes[i]["x_max"]*frame.cols, boxes[i]["y_max"]*frame.rows), cv::Scalar(0, 255, 0), 2);
                }

                // Display frame
                cv::imshow("preview", frame);

                // Wait and check if 'q' pressed
                if (cv::waitKey(1) == 'q') {
                    return 0;
                }
            }
        }
       // }
    }

    return 0;
}