#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <iostream>
#include <ctype.h>
#include "functions.hpp"
#include "registration.hpp"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

Mat image;

bool backprojMode = false;
bool selectObject = false;
int trackObject = 0;
bool showHist = true;
Point origin;
Rect selection;
int vmin = 20, vmax = 256, smin = 50;

extern string hot_keys;
extern const char *keys;

int main(int argc, const char **argv)
{
    VideoCapture cap;
    Rect trackWindow;
    int hsize = 16;
    float hranges[] = {0, 180};
    const float *phranges = hranges;

    cap.open(argv[1]);
    if (!cap.isOpened()) {
        cout << "***Could not access file...***\n";
        return -1;
    }

    namedWindow("CamShift", 0);
    setMouseCallback("CamShift", onMouse, 0);

    Mat frame, hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj,back_hist;
    bool paused = false;
    bool start  = false;

    cap >> frame;

    //SURF
    int minHessian = 500;
    Ptr<SURF> detector = SURF::create(minHessian, 2, 3);
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;

    KalmanFilter KF(4, 2, 0);
    Mat state(2, 1, CV_32F); /* (phi, delta_phi) */
    Mat processNoise(2, 1, CV_32F);
    Mat measurement = Mat::zeros(2, 1, CV_32F);

    KF.transitionMatrix = (Mat_<float>(4, 4) << 1, 0, 1, 0,   0, 1, 0, 1,  0, 0, 1, 0,  0, 0, 0, 1);
    KF.statePre = (Mat_<int>(2, 2) << origin.x, origin.y, 0, 0);

    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(1));
    setIdentity(KF.measurementNoiseCov, Scalar::all(5));
    setIdentity(KF.errorCovPost, Scalar::all(1));

    vector<Point> mousev, kalmanv;
    mousev.clear();
    kalmanv.clear();

    // int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));
    // VideoWriter outputVideo;
    // Size S = Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH),    //Acquire input size
    //               (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    // outputVideo.open("output.mp4" , ex, cap.get(CV_CAP_PROP_FPS), S, true);

    while (1) {

        if (!paused && trackObject) {
            cap >> frame;
            if (frame.empty())
                break;
        }

        frame.copyTo(image);

        if (!paused) {
            cvtColor(image, hsv, COLOR_BGR2HSV);

            if (trackObject) {
                //Init state
                int _vmin = vmin, _vmax = vmax;

                inRange(hsv, Scalar(0, smin, MIN(_vmin, _vmax)),
                        Scalar(180, 256, MAX(_vmin, _vmax)), mask);
                int ch[] = {0, 0};
                hue.create(hsv.size(), hsv.depth());
                mixChannels(&hsv, 1, &hue, 1, ch, 1);

                if (trackObject < 0) {
                    Mat roi(hue, selection), maskroi(mask, selection);
                    calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
                    normalize(hist, hist, 0, 255, NORM_MINMAX);

                    trackWindow = selection;
                    trackObject = 1;

                    state.at<float>(0) = selection.x + selection.width / 2;
                    state.at<float>(1) = selection.y + selection.height / 2;

                }


                Mat prediction = KF.predict();
                Point predictPt(prediction.at<float>(0), prediction.at<float>(1));

                Mat objroi(hue, trackWindow);
                detector->detectAndCompute(objroi, Mat(), keypoints_1, descriptors_1);

                calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
                backproj &= mask;

                calcHist(&backproj, 1, 0, Mat(), back_hist, 1, &hsize, &phranges);
                normalize(back_hist, back_hist, 0, 255, NORM_MINMAX);

                double bhatt = compareHist(hist, back_hist, CV_COMP_BHATTACHARYYA);

                RotatedRect trackBox = CamShift(backproj, trackWindow,
                                                TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));


                if (trackWindow.area() <= 1) {
                    int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5) / 6;
                    trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
                                       trackWindow.x + r, trackWindow.y + r) &
                                  Rect(0, 0, cols, rows);
                }
                Rect brect = trackBox.boundingRect();

                Mat targetroi(hue, brect);
                detector->detectAndCompute(targetroi, Mat(), keypoints_2, descriptors_2);

                vector <DMatch> good_matches;

                measurement.at<float>(0) = trackBox.center.x;
                measurement.at<float>(1) = trackBox.center.y;
                Point measPt = trackBox.center;

                Mat estimated = KF.correct(measurement);
                Point statePt(estimated.at<float>(0), estimated.at<float>(1));
                kalmanv.push_back(statePt);

                // trackWindow = Rect(statePt.x - trackWindow.width / 2, statePt.y - trackWindow.height / 2, trackWindow.width, trackWindow.height);

                if (backprojMode)
                    cvtColor(backproj, image, COLOR_GRAY2BGR);

                Rect btrack(trackBox.center.x - brect.width / 2, trackBox.center.y - brect.height / 2, brect.width, brect.height);

                // Rect btrack(trackBox.center.x - brect.width / 2, trackBox.center.y - brect.height / 2, brect.width, brect.height);

                // rectangle(image, btrack, Scalar(0, 0, 255));

                rectangle(image, trackWindow, Scalar(255, 0, 0));

                for (int i = 0; i < kalmanv.size() - 1; i++)
                    line(image, kalmanv[i], kalmanv[i + 1], Scalar(0, 0, 255), 2);

                drawCross(statePt, Scalar(255, 0, 255), 5);

            }
        }
        else if (trackObject < 0)
            paused = false;

        if (selectObject && selection.width > 0 && selection.height > 0) {
            Mat roi(image, selection);
            bitwise_not(roi, roi);
        }

        imshow("CamShift", image);

        // outputVideo << image;

        char c = (char)waitKey(10);
        if (c == 27 || c == 'q')
            break;
        switch (c) {

        case 'b':
            backprojMode = !backprojMode;
            break;
        case'r':
            cap.set(CV_CAP_PROP_POS_AVI_RATIO, 0);
            // outputVideo.set(CV_CAP_PROP_POS_AVI_RATIO, 0);
            trackObject = 0;
            kalmanv.clear();
            cap >> frame;
            break;
        default:
            ;
        }
    }

    cap.release();
    // outputVideo.release();

    return 0;
}
