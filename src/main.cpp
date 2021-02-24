#include <iostream>
#include <chrono>

// Inludes common necessary includes for development using depthai library
#include <depthai/device.hpp>

int main(int argc, char** argv) {
    std::cout << "-- START --" << std::endl;
    Device d("", false);
    std::cout << "-- END --" << std::endl;
    return 0;
}