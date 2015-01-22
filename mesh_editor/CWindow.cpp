// CWindow.cpp
// Chao Du 2014 Dec
#include "CWindow.h"

#include "HBase.h"

#include "C3DView.h"
#include "C2DView.h"
#include "CXCurve.h"
#include "CMeshGL.h"
#include "CLinkedList.h"

#include <QtCore>
#include <QtGui>

#include "UDataManipulateUtils.h"

#include "volumepkg.h"

using namespace ChaoVis;


#define VOLPKG_SLICE_MIN_INDEX 2


// Constructor
CWindow::CWindow( void ) :
    f3DView( NULL ),
    f2DView( NULL ),
    fVpkg( NULL ),
	fCurrentSliceIndex( 0 )
{
    // inititialize member variables
    f3DView = new C3DView( this );
    f2DView = new C2DView( this );

	connect( f2DView, SIGNAL( SendSignalOnNextClicked() ), this, SLOT( OnLoadNextSlice() ) );
	connect( f2DView, SIGNAL( SendSignalOnPrevClicked() ), this, SLOT( OnLoadPrevSlice() ) );
    connect( f2DView, SIGNAL( SendSignalMeshChanged() ), f3DView, SLOT( updateGL() ) );

    // create 3D view
    QVBoxLayout *a3DViewLayout = new QVBoxLayout;
    a3DViewLayout->addWidget( ( QWidget* )f3DView );

    // create 2D view
    QVBoxLayout *a2DViewLayout = new QVBoxLayout;
    a2DViewLayout->addWidget( f2DView );

    // compose layouts to the window
#ifdef _CWINDOW_IS_DERIVED_FROM_QWIGET_
    // REVISIT - how to set up layout for QMainWindow
    // http://stackoverflow.com/questions/17597960/qt-cant-set-qvboxlayout-as-layout-in-qmainwindow
    QHBoxLayout *aMainLayout = new QHBoxLayout;
    aMainLayout->addLayout( a3DViewLayout );
    aMainLayout->addLayout( a2DViewLayout );
    setLayout( aMainLayout );
#else // _CWINDOW_IS_DERIVED_FROM_QMAINWINDOW_
    QHBoxLayout *aMainLayout = new QHBoxLayout;

	QWidget* aLeftView = new QWidget;
	aLeftView->setLayout( a3DViewLayout );
	aMainLayout->addWidget( aLeftView );
	
	QWidget* aRightView = new QWidget;
	aRightView->setLayout( a2DViewLayout );
    aMainLayout->addWidget( aRightView );
    
//	QWidget *aWidget = new QWidget();
	QSplitter *aSplitter = new QSplitter();
//    aWidget->setLayout( aMainLayout );
	aSplitter->addWidget( aLeftView );
	aSplitter->addWidget( aRightView );
	aSplitter->setStretchFactor( 0, 1 );
	aSplitter->setStretchFactor( 1, 1 );
//    setCentralWidget( aWidget );
	setCentralWidget( aSplitter );
#endif // _CWINDOW_IS_DERIVED_FROM_QWIDGET_

    // set window title
    setWindowTitle( tr( "Mesh Editor" ) );

    // create actions
    CreateActions();

    // create menus
    CreateMenus();
}

// Destructor
CWindow::~CWindow( void )
{
	deleteNULL( f3DView );
	deleteNULL( f2DView );
	deleteNULL( fVpkg );
	for ( size_t i = 0; i < fIntersections.size(); ++i ) {
		deleteNULL( fIntersections[ i ] );
	}
}

// Handle loading next slice
void CWindow::OnLoadNextSlice( void )
{
	// REVISIT - FILL ME HERE
	// REVISIT - get signal from 2D view that new slice should be loaded
	// TODO: (1) load new slice 
	//       (2) update intersection curve 
    //       (3) notify 3D view to update the location of the slice plane proxy
    fCurrentSliceIndex++; // REVISIT - FILL ME HERE - safe net needed, overflow
    f2DView->SetIntersection( fIntersections[ fCurrentSliceIndex ] );
    f2DView->SetSliceIndex( fCurrentSliceIndex );

    OpenSlice( fCurrentSliceIndex );
    f2DView->SetImage( fSliceImage );
    update();
}

// Handle loading previous slice
void CWindow::OnLoadPrevSlice( void )
{
	// REVISIT - FILL ME HERE
	// REVISIT - get signal from 2D view that new slice should be loaded
	// TODO: (1) load new slice 
	//       (2) update intersection curve 
    //       (3) notify 3D view to update the location of the slice plane proxy
    fCurrentSliceIndex--; // REVISIT - FILL ME HERE - safe net needed, overflow
    f2DView->SetIntersection( fIntersections[ fCurrentSliceIndex ] );
    f2DView->SetSliceIndex( fCurrentSliceIndex );

    OpenSlice( fCurrentSliceIndex );
    f2DView->SetImage( fSliceImage );
    update();
}

// Handle key press event
void CWindow::keyPressEvent( QKeyEvent *event )
{
	if ( event->key() == Qt::Key_Escape ) {
		// REVISIT - should prompt warning before exit
		close();
	} else {
		// REVISIT - dispatch key press event
		f3DView->ProcessKeyEvent( event );
	}
}

// Create menus
void CWindow::CreateMenus( void )
{
    fFileMenu = new QMenu( tr( "&File" ), this );
    fFileMenu->addAction( fOpenMeshAct );
    fFileMenu->addAction( fOpenVolAct );
    fFileMenu->addSeparator();
    fFileMenu->addAction( fExitAct );

	fEditMenu = new QMenu( tr( "&Edit" ), this );
	fEditMenu->addAction( fGetIntersectionAct );

    menuBar()->addMenu( fFileMenu );
	menuBar()->addMenu( fEditMenu );
}

// Create actions
void CWindow::CreateActions( void )
{
    fOpenMeshAct = new QAction( tr( "&Open mesh..." ), this );
    fOpenMeshAct->setShortcut( tr( "Ctrl+O" ) );
    connect( fOpenMeshAct, SIGNAL( triggered() ), this, SLOT( OpenMesh() ) );

    fOpenVolAct = new QAction( tr( "Open &volume..." ), this );
    fOpenVolAct->setShortcut( tr( "Ctrl+V" ) );
    connect( fOpenVolAct, SIGNAL( triggered() ), this, SLOT( OpenVol() ) );

    fExitAct = new QAction( tr( "E&xit" ), this );
    fExitAct->setShortcut( tr( "Ctrl+Q" ) );
    connect( fExitAct, SIGNAL( triggered() ), this, SLOT( Close() ) );

	fGetIntersectionAct = new QAction( tr( "&Calculate intersections" ), this );
	connect( fGetIntersectionAct, SIGNAL( triggered() ), this , SLOT( GetIntersection() ) );
}

// Initialize volume package
bool CWindow::InitializeVolumePkg( const std::string &nVpkgPath )
{
	deleteNULL( fVpkg );
	fVpkg = new VolumePkg( nVpkgPath );

	if ( fVpkg == NULL ) {
		return false;
	}

    return true;
}

void CWindow::OpenSlice( int nSliceIndex )
{
    // get image as cv::Mat
	cv::Mat aImgMat;
	fVpkg->getSliceAtIndex( nSliceIndex ).copyTo( aImgMat );

	// REVISIT - we need to convert 16-bit Mat to 8-bit 3-channel Mat
	//           for display purpose
	aImgMat.convertTo( aImgMat, CV_8U, 1.0 / 256.0 );
	cvtColor( aImgMat, aImgMat, CV_GRAY2BGR );

	// convert cv::Mat to Qt::QImage
	fSliceImage = Mat2QImage( aImgMat );
}

void CWindow::Initialize2DView( void )
{
    f2DView->SetImage( fSliceImage );
}

// Calculate the intersection of the mesh and slices
void CWindow::CalcIntersection( std::vector< CXCurve* > &nIntersections,
								VolumePkg *nVpkg,
								const CMeshGL *nMeshModel )
{
	// REVISIT - this code is copied from volume-cartographer/texture/sliceProjection.cpp
	// REVISIT - the slice on each end of the mesh may or may not have triangle on it
	// REVISIT - manually set minimum and maximum
	int aMinSliceIndex = VOLPKG_SLICE_MIN_INDEX;
	int aMaxSliceIndex = fVpkg->getNumberOfSlices() - 1 - VOLPKG_SLICE_MIN_INDEX;
	int aNumSlices = aMaxSliceIndex - aMinSliceIndex + 1; // does not equal to # of image slice
	// because it does not contain the beginning 2 and ending 2 slices

	std::vector< std::vector< Vec2< float > > > aIntrsctPos( aNumSlices ); // actual points
	std::vector< std::vector< int > > aIntrsctIndex( aNumSlices ); // point indices

	// iterate through all the edges
	std::set< Vec2< int >, ChaoVis::EdgeCompareLess >::const_iterator aIter;
    int aIndex1, aIndex2;
    PointXYZRGBNormal aV1, aV2;
	for ( aIter = nMeshModel->fEdges.begin(); aIter != nMeshModel->fEdges.end(); ++aIter ) {

        aV1 = nMeshModel->fPoints[ ( *aIter )[ 0 ] ];
        aV2 = nMeshModel->fPoints[ ( *aIter )[ 1 ] ];
        aIndex1 = ( *aIter )[ 0 ];
        aIndex2 = ( *aIter )[ 1 ];

        if ( aV2.x < aV1.x ) { // make sure the points are ordered
            swap< int >( aIndex1, aIndex2 );
            swap< PointXYZRGBNormal >( aV1, aV2 );
        }

		int aStartIndx = ( int )ceil( aV1.x );
        int aEndIndx = ( int )floor( aV2.x );

		// safe net
		if ( aStartIndx < aMinSliceIndex || aEndIndx > aMaxSliceIndex - 1 ) {
			continue;
		}

		// interpolate all the intersection points
        for ( int i = aStartIndx; i <= aEndIndx; ++i ) {

			int aRow, aCol;
			if ( fabs( aV2.x - aV1.x ) < 1e-6 ) {
				if ( fabs( aV2.x - i ) < 1e-6 ) {
					// point 1
					aRow = round( aV2.y );
					aCol = round( aV2.z );

                    assert( aRow > 0 && aCol > 0 );

					Vec2< float > aPt( aRow, aCol );
					aIntrsctPos[ i - aMinSliceIndex ].push_back( aPt );
                    aIntrsctIndex[ i - aMinSliceIndex ].push_back( aIndex2 );

					// point 2
					aRow = round( aV1.y );
					aCol = round( aV1.z );

                    assert( aRow > 0 && aCol > 0 );

					aPt = Vec2< float >( aRow, aCol );
					aIntrsctPos[ i - aMinSliceIndex ].push_back( aPt );
                    aIntrsctIndex[ i - aMinSliceIndex ].push_back( aIndex1 );

				} // if
				continue;
			} // if

			double d = ( aV2.x - i ) / ( aV2.x - aV1.x );

			aRow = round( d * aV1.y + ( 1.0 - d ) * aV2.y );
			aCol = round( d * aV1.z + ( 1.0 - d ) * aV2.z );

            assert( aRow > 0 && aCol > 0 );

			Vec2< float > aPt( aRow, aCol );
			aIntrsctPos[ i - aMinSliceIndex ].push_back( aPt );
            aIntrsctIndex[ i - aMinSliceIndex ].push_back( d < 0.5 ? aIndex1 : aIndex2 );

		} // for i

	} // for aIter

	// after getting all the intersection points, we need to get connection between the points
	// solutions: (1) sort the points by y (or z) value, which may not work for curved lines
	//            (2) "region grow" like scheme, find nearest point as connected
	// REVISIT - this is BUGGY, because we didn't utilize the connectivity information
	//           from the edges
	for ( size_t i = 0; i < aIntrsctPos.size(); ++i ) {
#ifdef _DEBUG
        // REVISIT - DEBUG
        qDebug() << "Before sorting:\n";
        for ( size_t j = 0; j < aIntrsctPos[ i ].size(); ++j ) {
            qDebug() << "(" << aIntrsctPos[ i ][ j ][ 0 ] << "," << aIntrsctPos[ i ][ j ][ 1 ] << ") ";
        }
        qDebug() << "\n";
#endif // _DEBUG

		// sort
		MakeChain( aIntrsctPos[ i ], aIntrsctIndex[ i ] );

#ifdef _DEBUG
        // REVISIT - DEBUG
        qDebug() << "After sorting:\n";
        for ( size_t j = 0; j < aIntrsctPos[ i ].size(); ++j ) {
            qDebug() << "(" << aIntrsctPos[ i ][ j ][ 0 ] << "," << aIntrsctPos[ i ][ j ][ 1 ] << ") ";
        }
        qDebug() << "\n";
#endif // _DEBUG

		// store
		CXCurve *aCurve = new CXCurve();
		for ( size_t j = 0; j < aIntrsctPos[ i ].size(); ++j ) {
			aCurve->InsertPoint( aIntrsctPos[ i ][ j ] );
			aCurve->Insert3DIndex( aIntrsctIndex[ i ][ j ] );
		} // for j
        // REVISIT - FILL ME HERE - set curve slice index
		nIntersections.push_back( aCurve );

	} // for i

}

// Make a list of vertices a chain
void CWindow::MakeChain( std::vector< Vec2< float > > &nPts,
						 std::vector< int > &nIndices )
{
	// safe net
	if ( nPts.size() == 0 || nIndices.size() == 0 ) {
		return;
	}

	// use linked list
	CLinkedList< Vec2< float > > aPtsList;
	CLinkedList< int > aIndexList;

	// a list of flags for which points have already been processed
	unsigned char *aProcessed;
	aProcessed = new unsigned char[ nPts.size() ];
	memset( aProcessed, 0, sizeof( unsigned char ) * nPts.size() );
	int aNumPointsToProcess = nPts.size();

	// threshold for detecting the end of chain, this value is three times the average of edge lengths
	float aDistThreshold = 1e6; // REVISIT - big enough
	float aDistThrasholdFactor = 3.0;

	// insert initial point
	aPtsList.Append( &nPts[ 0 ] );
	aIndexList.Append( &nIndices[ 0 ] );
	aProcessed[ 0 ] = 1;
	aNumPointsToProcess--;

	Vec2< float > aCurPt = nPts[ 0 ];
	int aNearPtIndex;

	// insert along one direction
	while ( aNumPointsToProcess != 0 ) {

		aNearPtIndex = FindNearestPoint( nPts, aProcessed, aCurPt );

		if ( Norm< float >( nPts[ aNearPtIndex ] - aCurPt ) < aDistThreshold * aDistThrasholdFactor ) {
			aPtsList.Append( &nPts[ aNearPtIndex ] );
			aIndexList.Append( &nIndices[ aNearPtIndex ] );

			aProcessed[ aNearPtIndex ] = 1;
			aNumPointsToProcess--;

            if ( aPtsList.GetSize() == 1 ) {
                aDistThreshold = Norm< float >( nPts[ aNearPtIndex ] - aCurPt );
            } else {
                // threshold = ( threshold * (n - 1) + newDist ) / n
                aDistThreshold += ( Norm< float >( nPts[ aNearPtIndex ] - aCurPt ) - aDistThreshold ) / aPtsList.GetSize();
                aCurPt = nPts[ aNearPtIndex ];
            }
		}
	}

	aCurPt = nPts[ 0 ];

	// insert along the other direction, if there are any left
	while ( aNumPointsToProcess != 0 ) {

		aNearPtIndex = FindNearestPoint( nPts, aProcessed, aCurPt );

		if ( Norm< float >( nPts[ aNearPtIndex ] - aCurPt ) < aDistThreshold * aDistThrasholdFactor ) {
			aPtsList.RevAppend( &nPts[ 0 ] );
			aIndexList.RevAppend( &nIndices[ 0 ] );

			aProcessed[ aNearPtIndex ] = 1;
			aNumPointsToProcess--;

			// threshold = ( threshold * (n - 1) + newDist ) / n
			aDistThreshold += ( Norm< float >( nPts[ aNearPtIndex ] - aCurPt ) - aDistThreshold ) / aPtsList.GetSize();
			aCurPt = nPts[ aNearPtIndex ];
		}
	}

	// finally, store back to points and indices
	// we don't care about the order (forward or backward) here, but we may somewhere else
	for ( size_t i = 0; i < nPts.size(); ++i ) {
		nPts[ i ] = aPtsList.GetData( i );
		nIndices[ i ] = aIndexList.GetData( i );
	}

	// clean up
	delete []aProcessed;
}

int CWindow::FindNearestPoint( std::vector< Vec2< float > > &nPts,
							   const unsigned char *nPtFlag,
							   const Vec2< float > &nCurPt )
{
	float aMinDist = 1e6; // REVISIT - big enough
	int aNearPtIndex = -1;

	for ( size_t i = 0; i < nPts.size(); ++i ) {
		if ( nPtFlag[ i ] != 1 && Norm< float >( nPts[ i ] - nCurPt ) < aMinDist ) {
			aNearPtIndex = i;
			aMinDist = Norm< float >( nPts[ i ] - nCurPt );
		}
	}

	return aNearPtIndex;
}

// Open mesh
void CWindow::OpenMesh( void )
{
    QString aMeshName = QFileDialog::getOpenFileName( this,
                                                      tr( "Open Mesh File" ),
                                                      QDir::currentPath(),
                                                      tr( "Mesh (*.ply *.obj)" ) );

    // open mesh model
	if ( aMeshName.length() != 0 ) {
        if ( f3DView->InitializeMeshModel( aMeshName.toStdString() ) ) {
            if ( f2DView != NULL ) {
                f2DView->SetMeshModel( f3DView->GetMeshModel() );
            }
        }
	}
}

// Open volume package
void CWindow::OpenVol( void )
{
    QString aVpkgPath = QString( "" );
    aVpkgPath = QFileDialog::getExistingDirectory( this,
                                                   tr( "Open Directory" ),
                                                   QDir::currentPath(),
                                                   QFileDialog::ShowDirsOnly |
                                                   QFileDialog::DontResolveSymlinks );
    if ( aVpkgPath.length() == 0 ) { // cancelled
        return;
    }

    if ( !InitializeVolumePkg( aVpkgPath.toStdString() + "/" ) ) {
        printf( "ERROR: cannot open the volume package at the specified location.\n" );
        return;
    }

    // get slice index
    bool aIsIndexOk = false;
	fCurrentSliceIndex = QInputDialog::getInt( this,
                                              tr( "Input" ),
                                              tr( "Please enter the index of the slice you want to overlay: " ),
											  VOLPKG_SLICE_MIN_INDEX, // default value
											  VOLPKG_SLICE_MIN_INDEX, // minimum value
											  fVpkg->getNumberOfSlices() - 1 - VOLPKG_SLICE_MIN_INDEX, // maximum value
                                              1, // step
                                              &aIsIndexOk );
    if ( !aIsIndexOk ) {
        printf( "WARNING: nothing was loaded because no slice index was specified.\n" );
        return;
    }

    // setup cross section plane for 3D rendering
	// REVISIT
    f3DView->InitializeXsectionPlane();

    // read slice
	OpenSlice( fCurrentSliceIndex );

    // set up view
    Initialize2DView();
    f2DView->SetSliceIndex( fCurrentSliceIndex );
}

// Close application
void CWindow::Close( void )
{
    close();
}

// Get intersection
void CWindow::GetIntersection( void )
{
    if ( fVpkg == NULL || f3DView->GetMeshModelConst() == NULL ) {
		// not all the data necessary is loaded
		char aMsg[ 256 ];
        if ( fVpkg == NULL && f3DView->GetMeshModelConst() == NULL ) {
			sprintf( aMsg, "Data %s is not loaded.", "volume package and mesh" );
		} else if ( fVpkg == NULL ) {
			sprintf( aMsg, "Data %s is not loaded.", "volume package" );
        } else if ( f3DView->GetMeshModelConst() == NULL ) {
			sprintf( aMsg, "Data %s is not loaded.", "mesh" );
		}
		QMessageBox::warning( this, tr( "Warning" ), tr( aMsg ) );
		return;
	}

	// things to do: 
	//   (1) calculate intersections -> fIntersecitons
	//   (2) draw "current" intersection on f2DView (that means, we have to have a "current" slice)
	//   (3) we need to find the connection (correspondence) of 2D intersection with 3D mesh
	// correspondences of intersection curve and mesh model were set up when finding them
    CalcIntersection( fIntersections, fVpkg, f3DView->GetMeshModelConst() );

    f2DView->SetIntersection( fIntersections[ fCurrentSliceIndex - VOLPKG_SLICE_MIN_INDEX ] );
    f2DView->SetSliceIndex( fCurrentSliceIndex );

    update();
}
