//
// Created by Ryan Taber on 11/12/15.
//


/* PURPOSE:
 *   Create a point cloud with points of the XYZRGB variety
 *   Save the resulting pcd using the volcart::meshing::orderedPCDMesher()
 *   This outfile is loaded by orderedPCDMesherTest for later comparison
 */

#include "shapes.h"
#include "orderedPCDMesher.h"
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>

int main(/*int argc, char** argv*/) {

    volcart::shapes::Plane mesh;
    //Create point cloud from mesh
    pcl::PointCloud <pcl::PointXYZRGB> pCloud = mesh.pointCloudXYZRGB();

    //convert pCloud to Ptr for orderedPCD() call
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud <pcl::PointXYZRGB>);
    *cloud = pCloud;

//
//    std::string outfile;
//
//    if (argc == 1){  //no outfile filename provided
//        outfile = "orderedPCDExample.pcd";
//    }
//    else if (argc == 2){
//        outfile = argv[1];
//    }
//    else{
//        return 1;
//    }

    //call orderedPCDMesher()
    //Will need to convert the ply data to pcd when reading in via orderedPCDMesherTest
    //Issues with ply reader:
        //http://answers.ros.org/question/136916/conversion-from-sensor_msgspointcloud2-to-pclpointcloudt/

    volcart::meshing::orderedPCDMesher(cloud, "orderedPCDExample.ply");

    std::cerr << "File written as orderedPCDExample.ply" << std::endl;


    return 0;
}
