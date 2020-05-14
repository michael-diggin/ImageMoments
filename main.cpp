#include <iostream>
#include <vector>
#include <numeric>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\opencv.hpp>
#include "moments.h"

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

    vector<double> t0s(100);
    vector<double> t1s(100);

    for (int i=0; i<100; i++){
        double t1 = (double)getTickCount();
        Moments r = get_moments(image);
        t1 = ((double)getTickCount()-t1)/getTickFrequency();
        t1s[i] = t1;
    }


    for (int i=0; i<100; i++){
        double t0 = (double)getTickCount();
        Moments m = moments(image, false);
        t0 = ((double)getTickCount()-t0)/getTickFrequency();
        t0s[i] = t0;
    }


    double t0_avg = accumulate(t0s.begin(), t0s.end(), 0.0)/100.0;
    double t1_avg = accumulate(t1s.begin(), t1s.end(), 0.0)/100.0;

    cout << t0_avg << endl;
    cout << t1_avg << endl;
    cout << t1_avg/t0_avg << endl;
	return 0;
}
