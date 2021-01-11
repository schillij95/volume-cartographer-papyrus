#pragma once

/** @file */

#include <cmath>

#include <opencv2/core.hpp>

#include "ShapePrimitive.hpp"

namespace volcart
{
namespace shapes
{
/**
 * @author Melissa Shankle
 * @date 12/02/15
 *
 * @brief Half-circle arch shape
 *
 * This class generates a half-circle arch shape, like a half-pipe. The width
 * parameter determines the number of points on the circumference of one arch of
 * the half-pipe. The height determines the number of arches, which can be
 * thought of as the the length of the half-pipe. Each arch is separated by
 * 1 unit.
 *
 * The shape will have a number of points equal to width x height. The points
 * generated by this class are ordered.
 *
 * @ingroup Shapes
 */
class Arch : public ShapePrimitive
{
public:
    Arch(int width = 10, int height = 10)
    {

        float rad = 5;

        std::vector<cv::Vec3d> curve;
        std::vector<cv::Vec3d> points;

        // rad will always be the same
        // theta (t) will be between 0 and pi
        // z will be between 0 and width
        cv::Vec3d c_point;
        for (int w = 0; w < width; ++w) {
            double t = w * M_PI / width;
            c_point[0] = rad * cos(t);
            c_point[1] = rad * sin(t);
            c_point[2] = 0;
            curve.push_back(c_point);
        }

        for (float z = 0; z < height; z += 1) {
            for (auto p_id = curve.begin(); p_id != curve.end(); ++p_id) {
                addVertex_(p_id->val[0], p_id->val[1], z);
            }
        }

        // generate the cells
        for (int i = 1; i < height; ++i) {
            for (int j = 1; j < width; ++j) {
                int v1, v2, v3, v4;
                v1 = i * width + j;
                v2 = v1 - 1;
                v3 = v2 - width;
                v4 = v1 - width;
                addCell_(v1, v2, v3);
                addCell_(v1, v3, v4);
            }
        }

        // Set this as an ordered mesh
        ordered_ = true;
        orderedWidth_ = width;
        orderedHeight_ = height;
    }
};
}  // namespace shapes
}  // namespace volcart
