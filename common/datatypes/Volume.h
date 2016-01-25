#pragma once

#ifndef _VOLCART_VOLUME_H_
#define _VOLCART_VOLUME_H_

#include <string>
#include <array>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include "LRUCache.h"
#include "Tensor3D.h"
#include "Slice.h"

using Voxel = cv::Vec3d;
using EigenValues = cv::Vec3d;
using EigenValue = double;
using EigenVectors = cv::Matx33d;
using EigenVector = cv::Vec3d;
using StructureTensor = cv::Matx33d;

namespace volcart
{
class ZeroStructureTensorException : public std::exception
{
public:
    virtual const char* what() const throw()
    {
        return "Structure tensor was zero";
    }
};

class Volume
{
public:
    Volume() = default;

    Volume(boost::filesystem::path slicePath,
           boost::filesystem::path normalPath, int32_t nslices,
           int32_t sliceWidth, int32_t sliceHeight)
        : slicePath_(slicePath),
          normalPath_(normalPath),
          numSlices_(nslices),
          sliceWidth_(sliceWidth),
          sliceHeight_(sliceHeight)
    {
        numSliceCharacters_ = std::to_string(nslices).size();
    }

    cv::Mat getSliceData(const int32_t index) const;

    bool setSliceData(const int32_t index, const cv::Mat& slice);

    boost::filesystem::path getSlicePath(const int32_t index) const;

    boost::filesystem::path getNormalPathAtIndex(const int32_t index) const;

    uint16_t getInterpolatedIntensity(const Voxel nonGridPoint) const
    {
        return interpolateAt(nonGridPoint);
    }

    uint16_t getIntensityAtCoord(const cv::Vec3d v) const
    {
        return getIntensityAtCoord(int32_t(v(0)), int32_t(v(1)), int32_t(v(2)));
    }

    uint16_t getIntensityAtCoord(const int32_t x, const int32_t y,
                                 const int32_t z) const
    {
        return getSliceData(z).at<uint16_t>(y, x);
    }

    void setCacheSize(const size_t newCacheSize)
    {
        cache_.setSize(newCacheSize);
    }

    size_t getCacheSize() const { return cache_.size(); };
    void setCacheMemoryInBytes(const size_t nbytes)
    {
        setCacheSize(nbytes / (sliceWidth_ * sliceHeight_));
    }

    Slice reslice(const Voxel center, const cv::Vec3d xvec,
                  const cv::Vec3d yvec, const int32_t width = 64,
                  const int32_t height = 64) const;

    StructureTensor structureTensorAtIndex(const int32_t x, const int32_t y,
                                           const int32_t z,
                                           const int32_t voxelRadius = 1) const;

    StructureTensor structureTensorAtIndex(const cv::Vec3i index,
                                           const int32_t voxelRadius = 1) const
    {
        return structureTensorAtIndex(index(0), index(1), index(2),
                                      voxelRadius);
    }

    std::array<std::pair<EigenValue, EigenVector>, 3> eigenPairsAtIndex(
        const int32_t x, const int32_t y, const int32_t z,
        const int32_t voxelRadius = 1) const;

    std::array<std::pair<EigenValue, EigenVector>, 3> eigenPairsAtIndex(
        const cv::Vec3i index, const int32_t voxelRadius = 1) const
    {
        return eigenPairsAtIndex(index(0), index(1), index(2), voxelRadius);
    }

    template <typename DType>
    Tensor3D<DType> getVoxelNeighbors(const cv::Point3i center,
                                      const int32_t dx, const int32_t dy,
                                      const int32_t dz) const;

    template <typename DType>
    Tensor3D<DType> getVoxelNeighborsCubic(const cv::Point3i origin,
                                           const int32_t diameter) const;

private:
    boost::filesystem::path slicePath_;
    boost::filesystem::path normalPath_;
    int32_t numSlices_;
    int32_t sliceWidth_;
    int32_t sliceHeight_;
    int32_t numSliceCharacters_;
    mutable volcart::LRUCache<int32_t, cv::Mat> cache_;

    uint16_t interpolateAt(const Voxel point) const;
};
}

#endif
