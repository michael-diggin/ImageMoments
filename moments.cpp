#include <iostream>
#include <vector>
#include <numeric>
#include <functional>
#include <algorithm>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\opencv.hpp>


using namespace std;
using namespace cv;

Moments drt_moments(const Mat& image)
{

    Size s = image.size();
    int width = s.width;
    int height = s.height;


    //projection arrays
    vector<int> vert(width, 0);
    vector<int> hor(height, 0);
    vector<int> diag(width+height, 0);
    vector<int> anti(width+height, 0);



    //power arrays
    vector<int> d1(width+height, 0);
    vector<int> d2(width+height, 0);
    vector<double> d3(width+height, 0);
    vector<double> a3(width+height, 0);

    for (int k=0; k<height+width; ++k)
    {
        d1[k] = k;
        int k2 = k*k;
        d2[k] = k2;
        d3[k] = k2*k;
        a3[k] = pow((k - width + 1), 3);
    }

    // loop through image by pixel values


    double t = (double)getTickCount();

    int* hptr = &hor[0];
    int* initvptr = &vert[0];
    int* vptr;
    int* dptr;
    int* aptr;
    int _p;


    for (int i=0; i< height; ++i)
    {
        vptr = initvptr;
        dptr = &diag[i];
        aptr = &anti[width-1+i];
        const uchar* p = image.ptr<uchar>(i);
        const uchar* p_end = p + width;

        for (; p<p_end; ++p)
        {
            _p = *p;
            *vptr += _p;
            *hptr += _p;
            *dptr += _p;
            *aptr += _p;

            ++vptr;
            ++dptr;
            --aptr;
        }

        ++hptr;

    }

    t = ((double)getTickCount() -t)/getTickFrequency();
    cout << "DRT Loop Time: " << t << endl;

    double m00, m01, m10, m11, m20, m02, m30, m12, m21, m03;

    m00 = accumulate(begin(vert), end(vert), 0.0);
    m10 = inner_product(begin(hor), end(hor), begin(d1), 0.0);
    m01 = inner_product(begin(vert), end(vert), begin(d1), 0.0);
    m20 = inner_product(begin(hor), end(hor), begin(d2), 0.0);
    m02 = inner_product(begin(vert), end(vert), begin(d2), 0.0);
    m30 = inner_product(begin(hor), end(hor), begin(d3), 0.0);
    m03 = inner_product(begin(vert), end(vert), begin(d3), 0.0);
    m11 = (inner_product(begin(diag), end(diag), begin(d2), 0.0) - m02 - m20)/2.0;
    double temp_1 = inner_product(begin(diag), end(diag), begin(d3), 0.0)/6.0;
    double temp_2 = inner_product(begin(anti), end(anti), begin(a3), 0.0)/6.0;
    m12 = temp_1 + temp_2 - m30/3.0;
    m21 = temp_1 - temp_2 - m03/3.0;

    Moments m(m00, m10, m01, m20, m11, m02, m30, m21, m12, m03);
    return m;
}


Moments opencv_moments(const Mat& image)
{

    Size s = image.size();


    double m00, m01, m10, m11, m20, m02, m30, m12, m21, m03;

    double t = (double)getTickCount();

    for(int y = 0; y < s.height; y++ )
    {
        const uchar* ptr = image.ptr<uchar>(y);
        int x0 = 0, x1 = 0, x2 = 0;
        int x3 = 0;

        for(int x=0; x < s.width; x++ )
        {
            int p = ptr[x];
            int xp = x * p, xxp;

            x0 += p;
            x1 += xp;
            xxp = xp * x;
            x2 += xxp;
            x3 += xxp * x;
        }

        int py = y * x0, sy = y*y;

        m03 += py * sy;  // m03
        m12 += x1 * sy;  // m12
        m21 += x2 * y;  // m21
        m30 += x3;             // m30
        m02 += x0 * sy;        // m02
        m11 += x1 * y;         // m11
        m20 += x2;             // m20
        m01 += py;             // m01
        m10 += x1;             // m10
        m00 += x0;             // m00
    }

    t = ((double)getTickCount() -t)/getTickFrequency();
    cout << "OCV Loop Time: " << t << endl;

    Moments m(m00, m10, m01, m20, m11, m02, m30, m21, m12, m03);
    return m;

}


Moments old_moments(const Mat& image)
{
    Size s = image.size();

    double m00, m01, m10, m11, m20, m02, m30, m12, m21, m03;

    double t = (double)getTickCount();

    for(int y = 0; y < s.height; y++ )
    {
        const uchar* ptr = image.ptr<uchar>(y);

        for(int x=0; x < s.width; x++ )
        {
            int p = ptr[x];
            int xp = x * p, xxp;
            int yp = y*p, yy;

            m00 += p;
            m10 += xp;
            m01 += yp;
            m11 += xp*y;
            xxp = xp * x;
            yy = y*y;
            m20 += xxp;
            m02 += yy*p;
            m30 += xxp * x;
            m21 += xxp*y;
            m12 += xp*yy;
            m03 += yy*yp;
        }
    }

    t = ((double)getTickCount() -t)/getTickFrequency();
    cout << "OLD Loop Time: " << t << endl;

    Moments m(m00, m10, m01, m20, m11, m02, m30, m21, m12, m03);
    return m;
}
