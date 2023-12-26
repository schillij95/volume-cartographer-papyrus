#include "vc/core/types/VolumeGrids.hpp"

#include <iomanip>
#include <sstream>

#include <opencv2/imgcodecs.hpp>

#include "vc/core/io/TIFFIO.hpp"

namespace fs = volcart::filesystem;
namespace tio = volcart::tiffio;

using namespace volcart;

// Load a VolumeGrids from disk
VolumeGrids::VolumeGrids(fs::path path) : DiskBasedObjectBaseClass(std::move(path))
{
    if (metadata_.get<std::string>("type") != "vol") {
        throw std::runtime_error("File not of type: vol");
    }

    width_ = metadata_.get<int>("width");
    height_ = metadata_.get<int>("height");
    slices_ = metadata_.get<int>("slices");
    numSliceCharacters_ = std::to_string(slices_).size();

    std::vector<std::mutex> init_mutexes(slices_);

    slice_mutexes_.swap(init_mutexes);
}

// Setup a VolumeGrids from a folder of slices
VolumeGrids::VolumeGrids(fs::path path, std::string uuid, std::string name)
    : DiskBasedObjectBaseClass(
          std::move(path), std::move(uuid), std::move(name)),
          slice_mutexes_(slices_)
{
    metadata_.set("type", "vol");
    metadata_.set("width", width_);
    metadata_.set("height", height_);
    metadata_.set("slices", slices_);
    metadata_.set("voxelsize", double{});
    metadata_.set("min", double{});
    metadata_.set("max", double{});
}

// Load a VolumeGrids from disk, return a pointer
VolumeGrids::Pointer VolumeGrids::New(fs::path path)
{
    return std::make_shared<VolumeGrids>(path);
}

// Set a VolumeGrids from a folder of slices, return a pointer
VolumeGrids::Pointer VolumeGrids::New(fs::path path, std::string uuid, std::string name)
{
    return std::make_shared<VolumeGrids>(path, uuid, name);
}

int VolumeGrids::sliceWidth() const { return width_; }
int VolumeGrids::sliceHeight() const { return height_; }
int VolumeGrids::numSlices() const { return slices_; }
double VolumeGrids::voxelSize() const { return metadata_.get<double>("voxelsize"); }
double VolumeGrids::min() const { return metadata_.get<double>("min"); }
double VolumeGrids::max() const { return metadata_.get<double>("max"); }

void VolumeGrids::setSliceWidth(int w)
{
    width_ = w;
    metadata_.set("width", w);
}

void VolumeGrids::setSliceHeight(int h)
{
    height_ = h;
    metadata_.set("height", h);
}

void VolumeGrids::setNumberOfSlices(size_t numSlices)
{
    slices_ = numSlices;
    numSliceCharacters_ = std::to_string(numSlices).size();
    metadata_.set("slices", numSlices);
}

void VolumeGrids::setVoxelSize(double s) { metadata_.set("voxelsize", s); }
void VolumeGrids::setMin(double m) { metadata_.set("min", m); }
void VolumeGrids::setMax(double m) { metadata_.set("max", m); }

VolumeGrids::Bounds VolumeGrids::bounds() const
{
    return {
        {0, 0, 0},
        {static_cast<double>(width_), static_cast<double>(height_),
         static_cast<double>(slices_)}};
}

bool VolumeGrids::isInBounds(double x, double y, double z) const
{
    return x >= 0 && x < width_ && y >= 0 && y < height_ && z >= 0 &&
           z < slices_;
}

bool VolumeGrids::isInBounds(const cv::Vec3d& v) const
{
    return isInBounds(v(0), v(1), v(2));
}

fs::path VolumeGrids::getSlicePath(int index) const
{
    std::stringstream ss;
    ss << std::setw(numSliceCharacters_) << std::setfill('0') << index
       << ".tif";
    return path_ / ss.str();
}

cv::Mat VolumeGrids::getSliceData(int index) const
{
    if (cacheSlices_) {
        return cache_slice_(index);
    } else {
        return load_slice_(index);
    }
}

cv::Mat VolumeGrids::getSliceDataCopy(int index) const
{
    return getSliceData(index).clone();
}

cv::Mat VolumeGrids::getSliceDataRect(int index, cv::Rect rect) const
{
    auto whole_img = getSliceData(index);
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return whole_img(rect);
}

cv::Mat VolumeGrids::getSliceDataRectCopy(int index, cv::Rect rect) const
{
    auto whole_img = getSliceData(index);
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return whole_img(rect).clone();
}

void VolumeGrids::setSliceData(int index, const cv::Mat& slice, bool compress)
{
    auto slicePath = getSlicePath(index);
    tio::WriteTIFF(
        slicePath.string(), slice,
        (compress) ? tiffio::Compression::LZW : tiffio::Compression::NONE);
}

uint16_t VolumeGrids::intensityAt(int x, int y, int z) const
{
    // clang-format off
    if (x < 0 || x >= sliceWidth() ||
        y < 0 || y >= sliceHeight() ||
        z < 0 || z >= numSlices()) {
        return 0;
    }
    // clang-format on
    return getSliceData(z).at<uint16_t>(y, x);
}

// Trilinear Interpolation
// From: https://en.wikipedia.org/wiki/Trilinear_interpolation
uint16_t VolumeGrids::interpolateAt(double x, double y, double z) const
{
    // insert safety net
    if (!isInBounds(x, y, z)) {
        return 0;
    }

    double intPart;
    double dx = std::modf(x, &intPart);
    auto x0 = static_cast<int>(intPart);
    int x1 = x0 + 1;
    double dy = std::modf(y, &intPart);
    auto y0 = static_cast<int>(intPart);
    int y1 = y0 + 1;
    double dz = std::modf(z, &intPart);
    auto z0 = static_cast<int>(intPart);
    int z1 = z0 + 1;

    auto c00 =
        intensityAt(x0, y0, z0) * (1 - dx) + intensityAt(x1, y0, z0) * dx;
    auto c10 =
        intensityAt(x0, y1, z0) * (1 - dx) + intensityAt(x1, y0, z0) * dx;
    auto c01 =
        intensityAt(x0, y0, z1) * (1 - dx) + intensityAt(x1, y0, z1) * dx;
    auto c11 =
        intensityAt(x0, y1, z1) * (1 - dx) + intensityAt(x1, y1, z1) * dx;

    auto c0 = c00 * (1 - dy) + c10 * dy;
    auto c1 = c01 * (1 - dy) + c11 * dy;

    auto c = c0 * (1 - dz) + c1 * dz;
    return static_cast<uint16_t>(cvRound(c));
}

Reslice VolumeGrids::reslice(
    const cv::Vec3d& center,
    const cv::Vec3d& xvec,
    const cv::Vec3d& yvec,
    int width,
    int height) const
{
    auto xnorm = cv::normalize(xvec);
    auto ynorm = cv::normalize(yvec);
    auto origin = center - ((width / 2) * xnorm + (height / 2) * ynorm);

    cv::Mat m(height, width, CV_16UC1);
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            m.at<uint16_t>(h, w) =
                interpolateAt(origin + (h * ynorm) + (w * xnorm));
        }
    }

    return Reslice(m, origin, xnorm, ynorm);
}

// cv::Mat VolumeGrids::load_slice_(int index) const
// {
//     {
//         std::unique_lock<std::shared_mutex> lock(print_mutex_);
//         std::cout << "Requested to load slice " << index << std::endl;
//     }
//     auto slicePath = getSlicePath(index);
//     return cv::imread(slicePath.string(), -1);
// }

cv::Mat VolumeGrids::load_slice_(int index) const
{
    auto slicePath = getSlicePath(index);
    auto slicePathSSD = slicePath;
    std::string pathStr = slicePath.string();
    std::string pathStr2 = slicePath.string();
    std::string pathStr3 = slicePath.string();
    // Replace SSD4TB with HDD8TB if the path does not exist
    // string of the disk name to be replaced
    char* diskName = "SSD4TB2";
    char* diskName2 = "HDD8TB";
    // check if diskname in path, if not: set diskname to diskname2
    auto pos2 = pathStr2.find(diskName);
    if (pos2 == std::string::npos) {
        diskName = diskName2;
    }

    size_t pos = pathStr.find(diskName);
    if (pos != std::string::npos) {
        pathStr.replace(pos, std::strlen(diskName), "SSD4TB");
        slicePathSSD = std::filesystem::path(pathStr);
    }
    // Check if the slice exists
    if (std::filesystem::exists(slicePathSSD)) {
        slicePath = slicePathSSD;
    }
    else {
        pos = pathStr2.find(diskName);
        if (pos != std::string::npos) {
            pathStr2.replace(pos, std::strlen(diskName), "SSD120GB");
            slicePathSSD = std::filesystem::path(pathStr2);
        }
        if (std::filesystem::exists(slicePathSSD)) {
            slicePath = slicePathSSD;
        }
        else {
            pos = pathStr3.find(diskName);
            if (pos != std::string::npos) {
                pathStr3.replace(pos, std::strlen(diskName), "FastSSD");
                slicePathSSD = std::filesystem::path(pathStr3);
            }
            if (std::filesystem::exists(slicePathSSD)) {
                slicePath = slicePathSSD;
            }
        }

    }
    

    {
        std::unique_lock<std::shared_mutex> lock(print_mutex_);
        std::cout << "Requested to load slice " << index << " from " << slicePath << std::endl;
    }
    // Attempt to load the slice
    return cv::imread(slicePath.string(), -1);
}

cv::Mat VolumeGrids::cache_slice_(int index) const
{
    // Check if the slice is in the cache.
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        if (cache_->contains(index)) {
            return cache_->get(index);
        }
    }

    {
        // Get the lock for this slice.
        auto& mutex = slice_mutexes_[index];

        // If the slice is not in the cache, get exclusive access to this slice's mutex.
        std::unique_lock<std::mutex> lock(mutex);
        // Check again to ensure the slice has not been added to the cache while waiting for the lock.
        {
            std::shared_lock<std::shared_mutex> lock(cache_mutex_);
            if (cache_->contains(index)) {
                return cache_->get(index);
            }
        }
        // Load the slice and add it to the cache.
        {
            auto slice = load_slice_(index);
            std::unique_lock<std::shared_mutex> lock(cache_mutex_);
            cache_->put(index, slice);
            return slice;
        }
    }

}


void VolumeGrids::cachePurge() const 
{
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    cache_->purge();
}

