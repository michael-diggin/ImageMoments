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

int iround(double d)
{
  return static_cast<int>(floor(d + 0.5));
}

Point rotatePointAbout(Point pt, double x, double y, double theta) {
    Point res = Point(0, 0);
    double ptx, pty;
    ptx = pt.x - x;
    pty = pt.y - y;

    res.x = iround( ptx * cos(theta) - pty * sin(theta) + x );
    res.y = iround( ptx * sin(theta) + pty * cos(theta) + y );

    return res;
}

int main() {

    Moments m1, m2, m3;
    double t1, t2, t3;
    Mat image, rotated;

	image = imread("images\\brain.jpg", IMREAD_GRAYSCALE);
	if (!image.data)
    {
        cout << "Could not find image" << endl;
        return -1;
    }
    
    if(!SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS))
        cout << "Could not set process priority";

    if(!SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL))
        cout << "Could not set thread priority";

    pre_compute_power_arrays(image.size());

    int w0 = image.size().width;
    int h0 = image.size().height;
    int w2 = w0/2;
    int h2 = h0/2;
    int w4 = w0/4;
    int h4 = h0/4;

    Mat display;
    Mat v, h, a, d;
    int r;

    for(r = 0; r < 180; r++) {
        Mat rot = getRotationMatrix2D(Point(w2, h2), r, 1 - static_cast<double>(r) / 400.0);
        Size sz = Size(image.size());

        warpAffine(image, rotated, rot, sz);
        
        cvtColor(rotated, display, COLOR_GRAY2RGB);

        t3 = (double)getTickCount();
        for(int i = 0; i < 10; i++)
            m3 = naive_moments(rotated);
        t3 = ((double)getTickCount()-t3)*10.0/getTickFrequency();

        t2 = (double)getTickCount();
        for(int i = 0; i < 100; i++)
            m2 = opencv_moments(rotated);
        t2 = ((double)getTickCount()-t2)/getTickFrequency();

        t1 = (double)getTickCount();
        for(int i = 0; i < 100; i++)
            m1 = drt_moments(rotated);
        t1 = ((double)getTickCount()-t1)/getTickFrequency();
        
        double x = m1.m10 / m1.m00;
        double y = m1.m01 / m1.m00;

        double u20 = m1.m20/m1.m00 - x*x;
        double u02 = m1.m02/m1.m00 - y*y;
        double u11 = m1.m11/m1.m00 - x*y;
        double theta = 0.5*atan(2*u11/(u20-u02));

        Point startx = rotatePointAbout(Point(x-w4, y), x, y, theta);
        Point endx = rotatePointAbout(Point(x+w4, y), x, y, theta);

        Point starty = rotatePointAbout(Point(x, y-h4), x, y, theta);
        Point endy = rotatePointAbout(Point(x, y+h4), x, y, theta);

        line(display, startx, endx, Scalar(255, 0, 0), 2);
        line(display, starty, endy, Scalar(255, 0, 0), 2);

        drt_images(rotated, v, h, d, a);

        cv::namedWindow("Image");
        cv::imshow("Image", display);

        cv::namedWindow("v");
        cv::imshow("v", v);

        cv::namedWindow("h");
        cv::imshow("h", h);

        cv::namedWindow("d");
        cv::imshow("d", d);

        cv::namedWindow("a");
        cv::imshow("a", a);

        waitKey(1);
    }


    cout << "DRT Total: " << t1 << endl;
    cout << "OCV Total: " << t2 << endl;
    cout << "MOM Total: " << t3 << endl;

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
