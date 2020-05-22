#include <iostream>
#include <vector>
#include <numeric>
#include <opencv2\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\opencv.hpp>
#include "moments.hpp"

using namespace std;
using namespace cv;

int main() {

    Mat image;
	image = imread("grayscale.jpg", IMREAD_GRAYSCALE);
	if (!image.data)
    {
        cout << "Could not find image" << endl;
        return -1;
    }

//    double t0 = (double)getTickCount();
//   Moments m0 = moments(image, false);
//   t0 = ((double)getTickCount()-t0)/getTickFrequency();



    double t2 = (double)getTickCount();
    Moments m2 = opencv_moments(image);
    t2 = ((double)getTickCount()-t2)/getTickFrequency();

        double t1 = (double)getTickCount();
    Moments m1 = drt_moments(image);
    t1 = ((double)getTickCount()-t1)/getTickFrequency();



    cout << "DRT Total: " << t1 << endl;
    cout << "OCV Total: " << t2 << endl;

	return 0;
}
