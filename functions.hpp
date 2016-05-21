#ifndef __FUNC_H__
#define __FUNC_H__

#include <iostream>
#include <ctype.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>


using namespace cv;
using namespace std;

extern Mat out;
// extern bool selectObject;
// extern int trackObject;
// extern Rect selection;
// extern Point origin;



// void onMouse( int event, int x, int y, int, void* )
// {
//     if( selectObject )
//     {
//         selection.x = MIN(x, origin.x);
//         selection.y = MIN(y, origin.y);
//         selection.width = std::abs(x - origin.x);
//         selection.height = std::abs(y - origin.y);

//         selection &= Rect(0, 0, image.cols, image.rows);
//     }

//     switch( event )
//     {
//     case EVENT_LBUTTONDOWN:
//         origin = Point(x,y);
//         selection = Rect(x,y,0,0);
//         selectObject = true;
//         break;
//     case EVENT_LBUTTONUP:
//         selectObject = false;
//         if( selection.width > 0 && selection.height > 0 )
//             trackObject = -1;
//         break;
//     }
// }

string hot_keys =
    "\n\nHot keys: \n"
    "\tESC - quit the program\n"
    "\tc - stop the tracking\n"
    "\tb - switch to/from backprojection view\n"
    "\th - show/hide object histogram\n"
    "\tp - pause video\n"
    "To initialize tracking, select the object with mouse\n";

static void help()
{
    cout << "\nMean-shift based tracker\n"
            "Select an object to track using mouse \n"
            "Input as user file path\n"
            "Usage: \n"
            "   ./tracking [path to video]\n";
    cout << hot_keys;
}

const char* keys =
{
    "{help h | | show help message}{@camera_number| 0 | camera number}"
};

#endif