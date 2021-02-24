#include <iostream>
#include <chrono>

// Inludes common necessary includes for development using depthai library
#include <depthai/device.hpp>

#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    std::cout << "-- START --" << std::endl;

    // Create DepthAI Device
    Device device("", false);

    // Create Pipeline
    std::string config_json_str = "{\"streams\": [\"metaout\", \"previewout\"],\"ai\": {\"calc_dist_to_bb\": true,\"blob_file\": \"/home/eli/apps/depthtancing/person-detection-retail-0013/model.blob\",\"blob_file_config\": \"/home/eli/apps/depthtancing/person-detection-retail-0013/config.json\"}}";
    std::shared_ptr<CNNHostPipeline> pipeline = device.create_pipeline(config_json_str);
    while(true) {
        std::tuple<std::list<std::shared_ptr<NNetPacket>>, std::list<std::shared_ptr<HostDataPacket>>> packets = pipeline->getAvailableNNetAndDataPackets();

        std::list<std::shared_ptr<NNetPacket>> nnet_packets = std::get<0>(packets);
        std::list<std::shared_ptr<HostDataPacket>> data_packets = std::get<1>(packets);

        for (std::shared_ptr<HostDataPacket> data_packet : data_packets) {
            if (data_packet->stream_name == "previewout") {
                const unsigned char* data = data_packet->getData();
                // 3x320x544
                std::cout << "test " << data_packet->dimensions[0] << "x" << data_packet->dimensions[1] << "x" << data_packet->dimensions[2] << std::endl;
            }
        }
    }

    std::cout << "-- END --" << std::endl;
    return 0;
}