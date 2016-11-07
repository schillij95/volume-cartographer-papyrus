//
// Created by Hannah Hatch on 8/12/16.
//
#ifdef VC_USE_VCGLIB

#include "meshing/QuadricEdgeCollapseDecimation.h"

using namespace volcart::meshing;

///// Constructors /////
QuadricEdgeCollapseDecimation::QuadricEdgeCollapseDecimation()
{
    itkInput_ = nullptr;
    setDefaultParams();
}

QuadricEdgeCollapseDecimation::QuadricEdgeCollapseDecimation(
    volcart::ITKMesh::Pointer mesh)
{
    itkInput_ = mesh;
    setDefaultParams();
}

///// Set inputs & params /////
void QuadricEdgeCollapseDecimation::setMesh(volcart::ITKMesh::Pointer mesh)
{
    itkInput_ = mesh;
}

void QuadricEdgeCollapseDecimation::setDefaultParams()
{
    collapseParams_.SetDefaultParams();
    collapseParams_.PreserveBoundary = true;
    collapseParams_.PreserveTopology = true;
}

///// Processing /////
void QuadricEdgeCollapseDecimation::compute()
{
    convertMeshtoVCG_();
    vcg::LocalOptimization<VcgMesh> deciSession(vcgInput_, &collapseParams_);

    // Sets the target number of faces
    deciSession.SetTargetSimplices(desiredFaces_);
    deciSession.Init<VcgTriEdgeCollapse>();

    // Loop for large meshes that it can't decimate in one iteration
    while (vcgInput_.fn > static_cast<int>(desiredFaces_) &&
           deciSession.DoOptimization()) {
        std::cerr << "volcart::meshing::QECD: No. of faces: " << vcgInput_.fn
                  << std::endl;
    }
    // Cleans up pointers generated by the edge collapse
    deciSession.Finalize<VcgTriEdgeCollapse>();
}

void QuadricEdgeCollapseDecimation::compute(size_t desiredFaces)
{
    desiredFaces_ = desiredFaces;
    compute();
}

///// Get Output /////
volcart::ITKMesh::Pointer QuadricEdgeCollapseDecimation::getMesh()
{
    // New ITK Mesh
    outputMesh_ = volcart::ITKMesh::New();

    // Used to give the vertices an id that can be used to create faces
    volcart::ITKPoint point;
    unsigned long j = 0;
    VcgMesh::VertexPointer vp;
    vcg::SimpleTempData<VcgMesh::VertContainer, unsigned long> indices(
        vcgInput_.vert);

    // Takes vcg vertices and stores their coordinates into an itk point and
    // adds it to itk mesh
    for (auto vi = vcgInput_.vert.begin(); vi != vcgInput_.vert.end(); vi++) {
        vp = &(*vi);
        indices[vp] = j;
        if (!vi->IsD()) {
            point[0] = vi->P()[0];
            point[1] = vi->P()[1];
            point[2] = vi->P()[2];

            outputMesh_->SetPoint(j, point);
            j++;
        }
    }

    // Takes vcg cells and store their vertices into and itk cell and adds it to
    // itk mesh
    unsigned long cellID = 0;
    volcart::ITKCell::CellAutoPointer newCell;
    for (auto fi = vcgInput_.face.begin(); fi != vcgInput_.face.end(); fi++) {
        if (!fi->IsD()) {
            newCell.TakeOwnership(new ITKTriangle);

            auto point1 = indices[fi->V(0)];
            auto point2 = indices[fi->V(1)];
            auto point3 = indices[fi->V(2)];

            newCell->SetPointId(0, point1);
            newCell->SetPointId(1, point2);
            newCell->SetPointId(2, point3);

            outputMesh_->SetCell(cellID, newCell);
            cellID++;
        }
    }
    return outputMesh_;
}

///// Utilities /////
void QuadricEdgeCollapseDecimation::convertMeshtoVCG_()
{

    // Make a clean, empty mesh
    vcgInput_.Clear();

    // Takes itk points, gets their coordinates and adds a point to the vcg mesh
    for (auto p_It = itkInput_->GetPoints()->Begin();
         p_It != itkInput_->GetPoints()->End(); p_It++) {
        vcg::tri::Allocator<VcgMesh>::AddVertex(
            vcgInput_, VcgMesh::CoordType(
                           p_It.Value()[0], p_It.Value()[1], p_It.Value()[2]));
    }

    // Iterates over the cells in the itk mesh and the points within those cells
    // to create cells for the vcg mesh
    VcgMesh::FaceIterator fi = vcg::tri::Allocator<VcgMesh>::AddFaces(
        vcgInput_, itkInput_.GetPointer()->GetNumberOfCells());
    for (auto c_It = itkInput_->GetCells()->Begin();
         c_It != itkInput_->GetCells()->End(); c_It++) {
        int i = 0;
        VcgMesh::VertexPointer ivp[3];
        for (auto p_id = c_It.Value()->PointIdsBegin();
             p_id != c_It.Value()->PointIdsEnd(); p_id++, i++) {
            ivp[i] = &(vcgInput_.vert[*p_id]);
        }
        fi->V(0) = ivp[0];
        fi->V(1) = ivp[1];
        fi->V(2) = ivp[2];

        fi++;
    }
}
#endif
