#include <iostream>
#include <ctype.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>


using namespace cv;
using namespace std;

extern Mat image;
extern bool selectObject;
extern int trackObject;
extern Rect selection;
extern Point origin;

#define drawCross( center, color, d )                                 \
line( image, Point( center.x - d, center.y - d ), Point( center.x + d, center.y + d ), color, 2, CV_AA, 0); \
line( image, Point( center.x + d, center.y - d ), Point( center.x - d, center.y + d ), color, 2, CV_AA, 0 )


void onMouse( int event, int x, int y, int, void* )
{
    if( selectObject )
    {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);

        selection &= Rect(0, 0, image.cols, image.rows);
    }

    switch( event )
    {
    case EVENT_LBUTTONDOWN:
        origin = Point(x,y);
        selection = Rect(x,y,0,0);
        selectObject = true;
        break;
    case EVENT_LBUTTONUP:
        selectObject = false;
        if( selection.width > 0 && selection.height > 0 )
            trackObject = -1;
        break;
    }
}

// int optimize(Point center, Mat &measurement, vector<Point> &values){

//     measurement.at<float>(0) = center.x;
//     measurement.at<float>(1) = center.y;
//     // Mat estimated = KF.correct(measurement);

//     Point statePt(estimated.at<float>(0), estimated.at<float>(1));
//     Point measPt(measurement.at<float>(0), measurement.at<float>(1));

//     trackWindow = Rect(statePt.x - trackWindow.width / 2, statePt.y - trackWindow.height / 2, trackWindow.width, trackWindow.height);



// }