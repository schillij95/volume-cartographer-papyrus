#pragma once

#include <QCloseEvent>
#include <QComboBox>
#include <QMessageBox>
#include <QObject>
#include <QRect>
#include <QShortcut>
#include <QSpinBox>
#include <QThread>
#include <QTimer>
#include <QtWidgets>

#include "BlockingDialog.hpp"
#include "CBSpline.hpp"
#include "CXCurve.hpp"
#include "MathUtils.hpp"
#include "ui_VCMain.h"
#include "SegmentationStruct.hpp"

#include "vc/core/types/VolumePkg.hpp"
#include "vc/segmentation/ChainSegmentationAlgorithm.hpp"

#include <thread>
#include <condition_variable>
#include <atomic>
#include <SDL2/SDL.h>
#include <cmath>
#include <queue>
#include <unordered_map>


// Volpkg version required by this app
static constexpr int VOLPKG_SUPPORTED_VERSION = 6;
static constexpr int VOLPKG_SLICE_MIN_INDEX = 0;

namespace ChaoVis
{

class CVolumeViewerWithCurve;

class CWindow : public QMainWindow
{

    Q_OBJECT

public:
    enum EWindowState {
        WindowStateSegment,       // under segmentation state
        WindowStateRefine,        // under mesh refinemen state
        WindowStateDrawPath,      // draw new path
        WindowStateSegmentation,  // segmentation mode
        WindowStateIdle
    };  // idle
    enum SaveResponse : bool { Cancelled, Continue };

    // Structure for segmentation parameters
    // Declare parameters for new algos here and update SetUpSegParams()
    using SSegParams = struct SSegParams_tag {
        int fNumIters;
        double fAlpha;
        double fBeta;
        double fDelta;
        double fK1;
        double fK2;
        int fPeakDistanceWeight;
        int fWindowWidth{5};
        bool fIncludeMiddle;
        int targetIndex;
        // Optical Flow Segmentation Parameters
        bool purge_cache;
        int cache_slices;
        int smoothen_by_brightness;
        int outside_threshold;
        int optical_flow_pixel_threshold;
        int optical_flow_displacement_threshold;
        bool enable_smoothen_outlier;
        bool enable_edge;
        int edge_jump_distance;
        int edge_bounce_distance;
        int backwards_smoothnes_interpolation_window;
        int backwards_length;
    };

    using Segmenter = volcart::segmentation::ChainSegmentationAlgorithm;

signals:
    void submitSegmentation(Segmenter::Pointer s);

public slots:
    void onSegmentationFinished(Segmenter::PointSet ps);
    void onSegmentationFailed(std::string s);

public:
    CWindow();
    ~CWindow(void);

protected:
    void mousePressEvent(QMouseEvent* nEvent);
    void mouseReleaseEvent(QMouseEvent* nEvent);
    void keyPressEvent(QKeyEvent* event);

private:
    void CreateWidgets(void);
    void CreateMenus(void);
    void CreateActions(void);
    void CreateBackend();

    void closeEvent(QCloseEvent* closing);

    void setWidgetsEnabled(bool state);

    bool InitializeVolumePkg(const std::string& nVpkgPath);
    void setDefaultWindowWidth(volcart::Volume::Pointer volume);
    SaveResponse SaveDialog(void);

    void UpdateView(void);
    void ChangePathItem(std::string segID);
    void RemovePathItem(std::string segID);

    void SplitCloud(void);
    void DoSegmentation(void);
    void CleanupSegmentation(void);
    bool SetUpSegParams(void);

    void SetUpCurves(void);
    void SetCurrentCurve(int nCurrentSliceIndex);

    void prefetchSlices(void);
    void startPrefetching(int index);
    void OpenSlice(void);

    void InitPathList(void);

    void SetPathPointCloud(void);

    void queueSegmentation(std::string segmentationId, Segmenter::Pointer s);
    void executeNextSegmentation();

    void OpenVolume(void);
    void CloseVolume(void);

    void ResetPointCloud(void);
    static void audio_callback(void *user_data, Uint8 *raw_buffer, int bytes);
    void playPing();

private slots:
    void Open(void);
    void Close(void);
    void Keybindings(void);
    void About(void);
    void SavePointCloud();

    void OnNewPathClicked(void);
    void OnPathItemClicked(QTreeWidgetItem* item, int column);

    void PreviousSelectedId();
    void NextSelectedId();

    void ActivatePenTool();
    void ActivateSegmentationTool();
    void TogglePenTool(void);
    void ToggleSegmentationTool(void);

    void OnChangeSegAlgo(int index);

    void OnEdtAlphaValChange();
    void OnEdtBetaValChange();
    void OnEdtDeltaValChange();
    void OnEdtK1ValChange();
    void OnEdtK2ValChange();
    void OnEdtDistanceWeightChange();
    void OnEdtWindowWidthChange(int);
    void OnOptIncludeMiddleClicked(bool clicked);

    // void OnEdtSampleDistValChange( QString nText );
    void OnEdtStartingSliceValChange(QString nText);
    void OnEdtEndingSliceValChange();

    void OnBtnStartSegClicked(void);

    void OnEdtImpactRange(int nImpactRange);

    void OnLoadAnySlice(int nSliceIndex);
    void OnLoadNextSlice(void);
    void OnLoadPrevSlice(void);
    void OnLoadNextSliceShift(int shift);
    void OnLoadPrevSliceShift(int shift);

    void OnPathChanged(void);

    void UpdateSegmentCheckboxes(std::string aSegID);
    void toggleDisplayAll(bool checked);
    void toggleComputeAll(bool checked);

private:
    // data model
    EWindowState fWindowState;

    volcart::VolumePkg::Pointer fVpkg;
    QString fVpkgPath;
    std::string fVpkgName;
    bool fVpkgChanged;

    std::string fSegmentationId;
    volcart::Segmentation::Pointer fSegmentation;
    volcart::Volume::Pointer currentVolume;

    static const int AMPLITUDE = 28000;
    static const int FREQUENCY = 44100;

    SegmentationStruct fSegStruct;
    std::unordered_map<std::string, SegmentationStruct> fSegStructMap;
    int fMinSegIndex;
    int fMaxSegIndex;
    int fPathOnSliceIndex;
    int fEndTargetOffset{5};

    // for drawing mode
    CBSpline fSplineCurve;  // the curve at current slice
    // for editing mode
    CXCurve fIntersectionCurve;
    std::vector<CXCurve> fIntersections;  // curves of all the slices
    //    std::vector< CXCurve > fCurvesLower; // neighboring curves, { -1, -2,
    //    ... }
    //    std::vector< CXCurve > fCurvesUpper; // neighboring curves, { +1, +2,
    //    ... }

    SSegParams fSegParams;
    std::queue<std::pair<std::string, Segmenter::Pointer>> segmentationQueue;
    std::string submittedSegmentationId;

    volcart::OrderedPointSet<cv::Vec3d> fMasterCloud;
    volcart::OrderedPointSet<cv::Vec3d> fUpperPart;
    std::vector<cv::Vec3d> fStartingPath;

    // window components
    QMenu* fFileMenu;
    QMenu* fHelpMenu;

    QAction* fOpenVolAct;
    QAction* fSavePointCloudAct;
    QAction* fExitAct;
    QAction* fKeybinds;
    QAction* fAboutAct;

    CVolumeViewerWithCurve* fVolumeViewerWidget;
    QCheckBox* fchkDisplayAll;
    QCheckBox* fchkComputeAll;
    QTreeWidget* fPathListWidget;
    QPushButton* fPenTool;  // REVISIT - change me to QToolButton
    QPushButton* fSegTool;
    QComboBox* volSelect;
    QPushButton* assignVol;

    QSpinBox* fEdtWindowWidth;
    QLineEdit* fEdtDistanceWeight;
    QLineEdit* fEdtAlpha;
    QLineEdit* fEdtBeta;
    QLineEdit* fEdtDelta;
    QLineEdit* fEdtK1;
    QLineEdit* fEdtK2;
    QCheckBox* fOptIncludeMiddle;

    QLineEdit* fEdtStartIndex;
    QLineEdit* fEdtEndIndex;

    QSlider* fEdtImpactRange;
    QLabel* fLabImpactRange;

    // keyboard shortcuts
    QShortcut* slicePrev;
    QShortcut* sliceNext;
    QShortcut* sliceZoomIn;
    QShortcut* sliceZoomOut;
    QShortcut* displayCurves;
    QShortcut* displayCurves_C;
    QShortcut* impactUp;
    QShortcut* impactDwn;
    QShortcut* impactUp_old;
    QShortcut* impactDwn_old;
    QShortcut* segmentationToolShortcut;
    QShortcut* penToolShortcut;
    QShortcut* next1;
    QShortcut* prev1;
    QShortcut* next10;
    QShortcut* prev10;
    QShortcut* next100;
    QShortcut* prev100;
    QShortcut* prevSelectedId;
    QShortcut* nextSelectedId;

    Ui_VCMainWindow ui;

    QStatusBar* statusBar;

    bool can_change_volume_();

    QThread worker_thread_;
    BlockingDialog worker_progress_;
    QTimer worker_progress_updater_;
    size_t progress_{0};
    QLabel* progressLabel_;
    QProgressBar* progressBar_;

    // Prefetching worker
    std::thread prefetchWorker;
    std::condition_variable cv;
    std::mutex cv_m;
    std::atomic<bool> stopPrefetching;
    std::atomic<int> prefetchSliceIndex;
};  // class CWindow

class VolPkgBackend : public QObject
{
    Q_OBJECT
public:
    using Segmenter = volcart::segmentation::ChainSegmentationAlgorithm;

    explicit VolPkgBackend(QObject* parent = nullptr) : QObject(parent) {}

signals:
    void segmentationStarted(size_t);
    void segmentationFinished(Segmenter::PointSet ps);
    void segmentationFailed(std::string);
    void progressUpdated(size_t);

public slots:
    void startSegmentation(Segmenter::Pointer segmenter)
    {
        segmenter->progressUpdated.connect(
            [=](size_t p) { progressUpdated(p); });
        segmentationStarted(segmenter->progressIterations());
        try {
            auto result = segmenter->compute();
            segmentationFinished(result);
        } catch (const std::exception& e) {
            segmentationFailed(e.what());
        }
    }
};

}  // namespace ChaoVis
