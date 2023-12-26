#include "vc/texturing/LayerTexture.hpp"
#include <opencv2/core.hpp>

using namespace volcart;
using namespace volcart::texturing;

using Texture = LayerTexture::Texture;

// Texture LayerTexture::compute()
// {
//     // Setup
//     result_.clear();
//     auto height = static_cast<int>(ppm_->height());
//     auto width = static_cast<int>(ppm_->width());

//     // Setup output images
//     for (size_t i = 0; i < gen_->extents()[0]; i++) {
//         result_.emplace_back(cv::Mat::zeros(height, width, CV_16UC1));
//     }

//     // Get the mappings
//     auto mappings = ppm_->getMappings();

//     // Sort the mappings by Z-value
//     std::sort(
//         mappings.begin(), mappings.end(), [](const auto& lhs, const auto& rhs) {
//             return lhs.pos[2] < rhs.pos[2];
//         });

//     // Iterate through the mappings
//     for (const auto& pixel : mappings) {
//         // Generate the neighborhood
//         auto neighborhood = gen_->compute(vol_, pixel.pos, {pixel.normal});

//         // Assign to the output images
//         size_t it = 0;
//         for (const auto& v : neighborhood) {
//             result_.at(it++).at<uint16_t>(
//                 static_cast<int>(pixel.y), static_cast<int>(pixel.x)) = v;
//         }
//     }

//     return result_;
// }

Texture LayerTexture::compute()
{
    // Setup
    result_.clear();
    auto height = static_cast<int>(ppm_->height());
    auto width = static_cast<int>(ppm_->width());
    std::cout << "Generating " << gen_->extents()[0] << " layers" << std::endl;
    // Setup output images
    for (size_t i = 0; i < gen_->extents()[0]; i++) {
        result_.emplace_back(cv::Mat::zeros(height, width, CV_16UC1));
    }
    std::cout << "Sorting mappings" << std::endl;
    // Get the sorted mappings
    auto sortedMappings = ppm_->getSortedMappings();
    std::cout << "Starting texture generation" << std::endl;
    // Iterate through the sorted mappings
    for (const auto& mappedPixel : sortedMappings) {
        const size_t x = mappedPixel.x;
        const size_t y = mappedPixel.y;
        const cv::Vec6d& pixelData = *(mappedPixel.mapping);

        // Generate the neighborhood
        auto neighborhood = gen_->compute(vol_, {pixelData[0], pixelData[1], pixelData[2]}, {{pixelData[3], pixelData[4], pixelData[5]}});

        // check if pixel data normal norm is close to 1
        if (std::abs(cv::norm(cv::Vec3d(pixelData[3], pixelData[4], pixelData[5])) - 1) > 0.01) {
            std::cout << "Pixel data normal norm is not close to 1" << std::endl;
        }

        // Assign to the output images
        size_t it = 0;
        for (const auto& v : neighborhood) {
            result_.at(it++).at<uint16_t>(static_cast<int>(y), static_cast<int>(x)) = v;
        }
    }

    return result_;
}