#include <iostream>
#include <cmath>

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
    // mobilenet-ssd
    std::string config_json_str = "{\"streams\":[\"metaout\",\"previewout\"],\"ai\":{\"calc_dist_to_bb\":true,\"blob_file\":\"/home/eli/apps/depthtancing/mobilenet-ssd/model.blob\",\"blob_file_config\":\"/home/eli/apps/depthtancing/mobilenet-ssd/config.json\"},\"app\":{\"sync_video_meta_streams\":true}}";
    std::shared_ptr<CNNHostPipeline> pipeline = device.create_pipeline(config_json_str);

    // Continuously get packets from pipeline
    std::shared_ptr<dai::Detections> network_results;
    while (true) {
        // Get tuple (NNetPacket, HostDataPacket) from Pipeline
        std::tuple<std::list<std::shared_ptr<NNetPacket>>, std::list<std::shared_ptr<HostDataPacket>>> packets = pipeline->getAvailableNNetAndDataPackets();

        // Get 1st element of tuple (NNetPacket, HostDataPacket)
        std::list<std::shared_ptr<NNetPacket>> nnet_packets = std::get<0>(packets);

        // Get 2nd element of tuple (NNetPacket, HostDataPacket)
        std::list<std::shared_ptr<HostDataPacket>> data_packets = std::get<1>(packets);

        // For the retrieved nnet_packet,
        for (std::shared_ptr<NNetPacket> &nnet_packet : nnet_packets) { // nnet_packets.size() = 1
            // Get array of detected objects (maximum = 100)
            network_results = nnet_packet->getDetectedObjects();
        }

        for (std::shared_ptr<HostDataPacket> &data_packet : data_packets) {
            if (data_packet->stream_name == "previewout") {
                // If no data, continue to next packet
                if (data_packet->getData() == nullptr) {
                    continue;
                }

                // DepthAI HostDataPacket format
                // CxHxW = 3x320x544
                // C = data_packet->dimensions[0] = 3
                // H = data_packet->dimensions[1] = 320
                // W = data_packet->dimensions[2] = 544

                // HxWxC = 320x544x3
                // H = frame.rows       = 320
                // W = frame.cols       = 544
                // C = frame.channels() = 3

                // Convert to OpenCV cv::Mat object to hold data_packet from Pipeline
                cv::Mat frame = toMat(*(data_packet->data), data_packet->dimensions[2], data_packet->dimensions[1], 3, 1);

                // Get properties of detected objects
                std::vector<std::unordered_map<std::string, float>> detections;
                for (int i = 0; i < network_results->detection_count; ++i) {
                    std::unordered_map<std::string, float> detection;
                    detection["id"] = i;
                    detection["label"] = network_results->detections[i].label;
                    detection["confidence"] = network_results->detections[i].confidence;

                    // Calculate bounding box locations (in pixels)
                    detection["x_min"] = network_results->detections[i].x_min * frame.cols;
                    detection["y_min"] = network_results->detections[i].y_min * frame.rows;
                    detection["x_max"] = network_results->detections[i].x_max * frame.cols;
                    detection["y_max"] = network_results->detections[i].y_max * frame.rows;

                    // Store depth information
                    detection["depth_x"] = network_results->detections[i].depth_x;
                    detection["depth_y"] = network_results->detections[i].depth_y;
                    detection["depth_z"] = network_results->detections[i].depth_z;

                    detection["2d_loc_x"] = (detection["x_min"] + detection["x_max"]) / 2;
                    detection["2d_loc_y"] = detection["y_max"];

                    detection["isSafe"] = true;
                    detections.push_back(detection);

                    // Print detection parameters
                    /*std::cout << "[i=" << i << "] "
                    << "Label = " << network_results->detections[i].label
                    << " | confidence = " << network_results->detections[i].confidence
                    << " | x_min = " << network_results->detections[i].x_min
                    << " | y_min = " << network_results->detections[i].y_min
                    << " | x_max = " << network_results->detections[i].x_max
                    << " | y_max = " << network_results->detections[i].y_max
                    << " | depth_x = " << network_results->detections[i].depth_x
                    << " | depth_y = " << network_results->detections[i].depth_y
                    << " | depth_z = " << network_results->detections[i].depth_z
                    << std::endl;*/
                }

                // Overlay metadata on the frame in-place
                cv::Mat overlay = frame.clone();
                for (int i = 0; i < detections.size(); ++i) {
                    // Draw bounding box
                    cv::rectangle(frame,                                                     // image to overlay
                                  cv::Point(detections[i]["x_min"], detections[i]["y_min"]), // Top-Left point
                                  cv::Point(detections[i]["x_max"], detections[i]["y_max"]), // Bottom-Right point
                                  cv::Scalar(0, 255, 0),                                     // Color (BGR)
                                  2);                                                        // Thickness

                    // Write X
                    cv::putText(frame,
                                "x: " + std::to_string(detections[i]["depth_x"]),
                                cv::Point(detections[i]["x_min"], detections[i]["y_min"] + 30),
                                cv::FONT_HERSHEY_TRIPLEX,
                                0.5,
                                255);

                    // Write Y
                    cv::putText(frame,
                                "y: " + std::to_string(detections[i]["depth_y"]),
                                cv::Point(detections[i]["x_min"], detections[i]["y_min"] + 50),
                                cv::FONT_HERSHEY_TRIPLEX,
                                0.5,
                                255);

                    // Write Z
                    cv::putText(frame,
                                "z: " + std::to_string(detections[i]["depth_z"]),
                                cv::Point(detections[i]["x_min"], detections[i]["y_min"] + 70),
                                cv::FONT_HERSHEY_TRIPLEX,
                                0.5,
                                255);

                    // Write confidence level
                    cv::putText(frame,
                                "conf: " + std::to_string(detections[i]["confidence"]),
                                cv::Point(detections[i]["x_min"], detections[i]["y_min"] + 90),
                                cv::FONT_HERSHEY_TRIPLEX,
                                0.5,
                                255);

                    // Draw connecting lines
                    if (i + 1 < detections.size()) {
                        for (int j = i + 1; j < detections.size(); ++j) {
                            double dx = detections[i]["depth_x"] - detections[j]["depth_x"];
                            double dy = detections[i]["depth_y"] - detections[j]["depth_y"];
                            double dz = detections[i]["depth_z"] - detections[j]["depth_z"];
                            double distance = sqrt(pow(dx,2) + pow(dy, 2) + pow(dz,2));
                            if (distance < 1) {
                                detections[i]["isSafe"] = false;
                                detections[j]["isSafe"] = false;
                            }

                            cv::line(overlay,
                                     cv::Point(detections[i]["2d_loc_x"], detections[i]["2d_loc_y"]),
                                     cv::Point(detections[j]["2d_loc_x"], detections[j]["2d_loc_y"]),
                                     distance >= 1 ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255),
                                     2);

                            // Write distance in meters
                            cv::putText(frame,
                                        std::to_string(distance) + "m",
                                        cv::Point((detections[i]["2d_loc_x"] + detections[j]["2d_loc_x"])/2,
                                                  (detections[i]["2d_loc_y"] + detections[j]["2d_loc_y"])/2),
                                        cv::FONT_HERSHEY_TRIPLEX,
                                        0.5,
                                        cv::Scalar(255, 255, 255),
                                        1);
                        }
                    }

                    // Draw shadow effect
                    cv::ellipse(overlay,
                                cv::Point(detections[i]["2d_loc_x"], detections[i]["2d_loc_y"]),
                                cv::Size((detections[i]["x_max"] - detections[i]["x_min"]) / 2, 10),
                                0,
                                0,
                                360,
                                detections[i]["isSafe"] ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255),
                                cv::FILLED);
                }

                // Display frame
                cv::Mat blendedFrame;
                cv::addWeighted(overlay, 0.4, frame, 0.6, 0.0, blendedFrame);
                cv::Mat resizedFrame;
                cv::resize(blendedFrame, resizedFrame, cv::Size(900, 900));
                cv::imshow("Depthtancing", resizedFrame);

                // Wait and check if 'q' pressed
                if (cv::waitKey(1) == 'q') {
                    return 0;
                }
            }
        }
    }

    return 0;
}