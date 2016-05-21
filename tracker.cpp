#include "tracker.hpp"
using namespace std;
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>

#define drawCross( center, color, d )                                 \
line( out, Point( center.x - d, center.y - d ), Point( center.x + d, center.y + d ), color, 2, CV_AA, 0); \
line( out, Point( center.x + d, center.y - d ), Point( center.x - d, center.y + d ), color, 2, CV_AA, 0 )

Tracker::Tracker():
  m_showControlsGUI(false),
  m_initialized(false)

{}

Tracker::InitParams::InitParams()
  : histDims(16),
    vMin(32),
    vMax(256),
    sMin(50),
    sBox(8),
    showBackproject(false),
    showControlsGUI(true),
    showHistogram(true),
    histRanges()
{
  histRanges[0] = 0;
  histRanges[1] = 180.0f;
}

void Tracker::Init(const cv::Size &frameSize,
                   const InitParams &initParams)
{
  // Get frame size
  m_frameSize = frameSize;

  // Load init params.
  m_hSize = 16;
  m_histDims = initParams.histDims;
  m_vMin = initParams.vMin;
  m_vMax = initParams.vMax;
  m_sMin = initParams.sMin;
  m_sBox = initParams.sBox;
  m_histRanges[0] = initParams.histRanges[0];
  m_histRanges[1] = initParams.histRanges[1];

  m_KF = cv::KalmanFilter(4, 2, 0);
  m_state = cv::Mat(2, 1, CV_32F); /* (phi, delta_phi) */
  m_measured = cv::Mat::zeros(2, 1, CV_32F);

  m_KF.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0,   0, 1, 0, 1,  0, 0, 1, 0,  0, 0, 0, 1);
  cv::setIdentity(m_KF.measurementMatrix);
  cv::setIdentity(m_KF.processNoiseCov, cv::Scalar::all(1e-2));
  cv::setIdentity(m_KF.measurementNoiseCov, cv::Scalar::all(10));
  cv::setIdentity(m_KF.errorCovPost, cv::Scalar::all(1));

  m_past.clear();
  m_kalmanv.clear();

  m_histImg = Mat::zeros(200, 320, CV_8UC3);

  // Show controlsGUI, m_imgBackproject, histogram
  m_showControlsGUI = false;
  ShowControlsGUI();
  // Set initialized flag
  m_initialized = true;

  // Static variables
  // g_selectObject = false;
  g_initTracking = false;
  // g_selId = -1;
  // g_selRect;
  // g_selOrigin;
  // bool Tracker::g_selectObject = false;
  // int Tracker::g_initTracking = 0;
  // int Tracker::g_selId = -1;
  // cv::Rect Tracker::g_selRect;
  // cv::Point Tracker::g_selOrigin;


}

void Tracker::ShowControlsGUI()
{
  // cvNamedWindow(m_controlsGUIWndName.c_str(), 1);
  cv::namedWindow("CamShift", 0);
  cv::namedWindow("Histogram", 0);
  cv::namedWindow("Trackbars", 0);
  if (!m_showControlsGUI) {

    cv::createTrackbar("Vmin", "Trackbars", &m_vMin, 256, 0);
    cv::createTrackbar("Vmax", "Trackbars", &m_vMax, 256, 0);
    cv::createTrackbar("Smin", "Trackbars", &m_sMin, 256, 0);
    cv::setMouseCallback("CamShift", Tracker::OnMouse, 0);
  }
  m_showControlsGUI = true;

}

Tracker::~Tracker()
{
  // Safe free buffers.
  // Deinit();
  // Destroy windows.
  cv::destroyAllWindows();
}

void Tracker::OnMouse(int event, int x, int y, int /*flags*/, void *param)
{

  if (g_selectObject) {
    g_selRect.x = MIN(x, g_selOrigin.x);
    g_selRect.y = MIN(y, g_selOrigin.y);
    g_selRect.width = std::abs(x - g_selOrigin.x);
    g_selRect.height = std::abs(y - g_selOrigin.y);

  }

  switch (event) {

  case EVENT_LBUTTONDOWN:
    g_selOrigin = cv::Point(x, y);
    g_selRect = cv::Rect(x, y, 0, 0);
    g_selectObject = true;
    break;
  case EVENT_LBUTTONUP:
    g_selectObject = false;
    if (g_selRect.width > 0 && g_selRect.height > 0)
      g_initTracking = -1;
    break;
  }


}
void Tracker::InitTrackWindow(const cv::Mat &img, const cv::Rect &selRect)
{

  // if (g_initTracking < 0) {

  cv::Mat roi(m_imgHue, g_selRect), maskroi(m_imgMask, g_selRect);

  // Create histograms
  const float *phranges = m_histRanges;

  cv::calcHist(&roi, 1, 0, maskroi, m_hist, 1, &m_hSize, &phranges);
  cv::normalize(m_hist, m_hist, 0, 255, NORM_MINMAX);

  m_trackWindow = g_selRect;
  g_initTracking = 1;

  //Init state
  m_state.at<float>(0) = g_selRect.x + g_selRect.width / 2;
  m_state.at<float>(1) = g_selRect.y + g_selRect.height / 2;

  m_histImg = Scalar::all(0);
  int binW = m_histImg.cols / m_hSize;
  Mat buf(1, m_hSize, CV_8UC3);
  for (int i = 0; i < m_hSize; i++)
    buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i * 180. / m_hSize), 255, 255);
  cvtColor(buf, buf, COLOR_HSV2BGR);

  for (int i = 0; i < m_hSize; i++) {
    int val = saturate_cast<int>(m_hist.at<float>(i) * m_histImg.rows / 255);
    rectangle(m_histImg, Point(i * binW, m_histImg.rows),
              Point((i + 1)*binW, m_histImg.rows - val),
              Scalar(buf.at<Vec3b>(i)), -1, 8);
  }

  // }

}

void Tracker::ProcessFrame(const cv::Mat &img, cv::Mat &out)
{

  img.copyTo(out);

  // Draw selection box
  if (g_selectObject) {
    if ((g_selRect.width > 0) && (g_selRect.height > 0)) {
      Mat roi(out, g_selRect);
      bitwise_not(roi, roi);

    }
  }

  int ch[] = {0, 0};
  cv::cvtColor(img, m_imgHSV, COLOR_BGR2HSV);
  cv::inRange(m_imgHSV, cv::Scalar(0, m_sMin, MIN(m_vMin, m_vMax)),
              cv::Scalar(180, 256, MAX(m_vMin, m_vMax)), m_imgMask);
  m_imgHue.create(m_imgHSV.size(), m_imgHSV.depth());
  cv::mixChannels(&m_imgHSV, 1, &m_imgHue, 1, ch, 1);

  // Check if time to init
  if (g_initTracking < 0) {
    InitTrackWindow(img, g_selRect);
  }

  if (g_initTracking > 0) {

    Mat prediction = m_KF.predict();

    const float *phranges = m_histRanges;
    cv::calcBackProject(&m_imgHue, 1, 0, m_hist, m_imgBackproject, &phranges);
    m_imgBackproject &= m_imgMask;

    // double bhatt = compareHist(hist, back_hist, CV_COMP_BHATTACHARYYA);
    // cout << "B error " << bhatt << endl;

    m_trackBox = cv::CamShift(m_imgBackproject, m_trackWindow,
                              TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));
    // if (m_trackWindow.area() <= 1) {
    //   int cols = m_imgBackproject.cols, rows = m_imgBackproject.rows, r = (MIN(cols, rows) + 5) / 6;
    //   m_trackWindow = Rect(m_trackWindow.x - r, m_trackWindow.y - r,
    //                        m_trackWindow.x + r, m_trackWindow.y + r) &
    //                   Rect(0, 0, cols, rows);
    // }
    // Rect pRect(prediction.at<float>(0), prediction.at<float>(1),m_trackWindow.width,m_trackWindow.height);
    // rectangle(out, pRect, Scalar(0, 0, 255));

    Tracker::PredictPos();
    Tracker::DrawStuff(out);
  }

}

void Tracker::PredictPos()
{

  m_measured.at<float>(0) = m_trackBox.center.x;
  m_measured.at<float>(1) = m_trackBox.center.y;

  m_estimated = m_KF.correct(m_measured);
  cv::Point statePt(m_estimated.at<float>(0), m_estimated.at<float>(1));

  m_past.push_back(m_trackBox.center);
  m_kalmanv.push_back(statePt);

  // m_trackWindow = Rect(statePt.x - m_trackWindow.width / 2, statePt.y - m_trackWindow.height / 2,\
  //  m_trackWindow.width, m_trackWindow.height);

}


void Tracker::DrawStuff(cv::Mat &out)
{

  if (m_showBackproject)
    cvtColor(m_imgBackproject, out, COLOR_GRAY2BGR);

  Rect brect = m_trackBox.boundingRect();

  // ellipse(image, m_trackBox, Scalar(0, 255, 255), 3, LINE_AA);
  Rect btrack(m_trackBox.center.x - brect.width / 2, m_trackBox.center.y - brect.height / 2, brect.width, brect.height);

  // rectangle(image, btrack, Scalar(0, 0, 255));
  
  rectangle(out, m_trackWindow, Scalar(255, 0, 0));

  for (int i = 0; i < m_kalmanv.size() - 1; i++)
    line(out, m_kalmanv[i], m_kalmanv[i + 1], Scalar(0, 0, 255), 3);

  Point statePt(m_estimated.at<float>(0), m_estimated.at<float>(1));
  drawCross(statePt, Scalar(255, 0, 255), 5);
// drawCross(measPt, Scalar(0, 180, 255), 5);
}

void Tracker::HideControlsGUI()
{
  cvDestroyWindow("Trackbars");
  m_showControlsGUI = false;
}

bool Tracker::ToggleShowBackproject()
{
  
  m_showBackproject = !m_showBackproject;
}

// void Tracker::ShowBackproject()
// {
//   cvNamedWindow("CamShift", 1);
//   if (!m_showBackproject)
//   {
//     cvMoveWindow(m_backprojectWndName.c_str(), (2 * m_frameSize.width) + 20, 0);
//   }
//   cvShowImage(m_backprojectWndName.c_str(), m_imgBackproject);
//   m_showBackproject = true;
// }

// void Tracker::HideBackproject()
// {
//   cvDestroyWindow(m_backprojectWndName.c_str());
//   m_showBackproject = false;
// }

// void Tracker::ShowHistogram()
// {
//   cvNamedWindow(m_histogramWndName.c_str(), 1);
//   if (!m_showHistogram)
//   {
//     cvMoveWindow(m_histogramWndName.c_str(), (2 * m_frameSize.width) + 20, m_frameSize.height + 55);
//   }
//   cvShowImage(m_histogramWndName.c_str(), m_histImg);
//   m_showHistogram = true;
// }

// void Tracker::HideHistogram()
// {
//   cvDestroyWindow(m_histogramWndName.c_str());
//   m_showHistogram = false;
// }

