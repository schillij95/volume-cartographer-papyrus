#pragma once

#include <opencv2/core.hpp>
#include "common/types/PerPixelMap.h"
#include "common/types/UVMap.h"
#include "common/vc_defines.h"

namespace volcart
{
namespace texturing
{

class PPMGenerator
{
public:
    // Constructors/Destructors
    PPMGenerator() : _width(0), _height(0){};

    // Set/Get Parameters
    void setMesh(ITKMesh::Pointer m) { _inputMesh = m; };
    void setUVMap(const UVMap& u) { _uvMap = u; };
    void setDimensions(uint8_t w, uint8_t h);

    // Run
    void compute();

    // Output
    const PerPixelMap& getPPM() const { return _ppm; };
    PerPixelMap& getPPM() { return _ppm; };
private:
    struct CellInfo {
        std::vector<cv::Vec3d> Pts2D;
        std::vector<cv::Vec3d> Pts3D;
        cv::Vec3d Normal;
    };

    // Helpers
    void _generateCentroidMesh();
    void _generatePPM();
    cv::Vec3d _BarycentricCoord(
        const cv::Vec3d& nXYZ,
        const cv::Vec3d& nA,
        const cv::Vec3d& nB,
        const cv::Vec3d& nC);
    cv::Vec3d _CartesianCoord(
        const cv::Vec3d& nUVW,
        const cv::Vec3d& nA,
        const cv::Vec3d& nB,
        const cv::Vec3d& nC);

    // Data members
    ITKMesh::Pointer _inputMesh;
    ITKMesh::Pointer _centroidMesh;
    std::vector<CellInfo> _cellInformation;
    UVMap _uvMap;
    PerPixelMap _ppm;

    uint8_t _width;
    uint8_t _height;

    double _progress;
};
}
}  // namespace volcart