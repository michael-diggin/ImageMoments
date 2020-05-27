#include <iostream>
#include <vector>
#include <numeric>
#include <Windows.h>
#include <opencv2\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\opencv.hpp>
#include "moments.hpp"

using namespace std;
using namespace cv;

int main() {

    Mat image;
	image = imread("images\\grayscale.jpg", IMREAD_GRAYSCALE);
	if (!image.data)
    {
        cout << "Could not find image" << endl;
        return -1;
    }

//    double t0 = (double)getTickCount();
//   Moments m0 = moments(image, false);
//   t0 = ((double)getTickCount()-t0)/getTickFrequency();

    if(!SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS))
        cout << "crap";

    if(!SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL))
        cout << "alos crap";
    //waitKey(15000);

    Moments m1, m2;

    double t2 = (double)getTickCount();
    for(int i = 0; i < 100; i++)
        m2 = opencv_moments(image);
    t2 = ((double)getTickCount()-t2)/getTickFrequency();

    double t1 = (double)getTickCount();
    for(int i = 0; i < 100; i++)
        m1 = drt_moments(image);
    t1 = ((double)getTickCount()-t1)/getTickFrequency();

    cout << "DRT Total: " << t1 << endl;
    cout << "OCV Total: " << t2 << endl;

    cout << m1.m00 << "] m00 [" << m2.m00 << endl;
    cout << m1.m10 << "] m10 [" << m2.m10 << endl;
    cout << m1.m01 << "] m01 [" << m2.m01 << endl;
    cout << m1.m20 << "] m20 [" << m2.m20 << endl;
    cout << m1.m02 << "] m02 [" << m2.m02 << endl;
    cout << m1.m11 << "] m11 [" << m2.m11 << endl;
    cout << m1.m30 << "] m30 [" << m2.m30 << endl;
    cout << m1.m03 << "] m03 [" << m2.m03 << endl;
    cout << m1.m21 << "] m21 [" << m2.m21 << endl;
    cout << m1.m12 << "] m12 [" << m2.m12 << endl;

	return 0;
}
