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

#include "tracker.hpp"
#include "functions.hpp"
#include "registration.hpp"
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;
// using namespace tracker;

// Static variables
bool Tracker::g_selectObject = false;
int Tracker::g_initTracking = 0;
int Tracker::g_selId = -1;
Rect Tracker::g_selRect;
Point Tracker::g_selOrigin;

extern string hot_keys;
extern const char *keys;

int main(int argc, const char **argv)
{

    VideoCapture cap;
    Tracker objTracker;

    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) {
        help();
        return 0;
    }

    cap.open(argv[1]);
    if (!cap.isOpened()) {
        help();
        cout << "***Could not access file...***\n";
        return -1;
    }
    Size S = Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH),    //Acquire input size
                  (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT));

    cout << hot_keys;
    bool paused = false;

    Mat frame;
    cap >> frame;

    objTracker.Init(S, Tracker::InitParams());

    int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));
    VideoWriter outputVideo;
    // outputVideo.open("output.mp4" , ex, cap.get(CV_CAP_PROP_FPS), S, true);

    Mat out;
    try {

        while (1) {

            if (!paused && Tracker::g_initTracking) {
                cap >> frame;
                if (frame.empty())
                    break;
            }

            if (!paused) {


                objTracker.ProcessFrame(frame, out);

            }
            imshow("CamShift", out);
            // outputVideo << out;

            char c = (char)waitKey(10);
            if (c == 27)
                break;
            switch (c) {
            case 'b':
                objTracker.ToggleShowBackproject();
                break;
            case 'c':
                // trackObject = 0;
                // histimg = Scalar::all(0);
                break;
            case 'h':
                objTracker.HideControlsGUI();
            //     showHist = !showHist;    
            //     if (!showHist)
            //         destroyWindow("Histogram");
            //     else
            //         namedWindow("Histogram", 1);
            //     break;
            case 'p':
                paused = !paused;
                break;
            case 'r':
                cap.set(CV_CAP_PROP_POS_AVI_RATIO, 0);
                // outputVideo.set(CV_CAP_PROP_POS_AVI_RATIO, 0);
                cap >> frame;
                objTracker.Init(S, Tracker::InitParams());

                break;
            default:
                ;
            }
        }
    }

    catch (const cv::Exception &e) {
        std::cerr << e.what();
        cap.release();
        outputVideo.release();

        return 1;
    }
    cap.release();
    outputVideo.release();

    return 0;
}
