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

//power arrays
double *d1, *d2, *d3, *a3;

double product(const vector<long> &mat, double power[], int many)
{
    double sum = 0.0;
    for(int i = 0; i < many; i++)
        sum += static_cast<double>(mat[i]) * power[i];

    return sum;
}

void pre_compute_power_arrays(const Size s) {
    const int width = s.width;
    const int height = s.height;

    //power arrays
    d1 = new double [width+height];
    d2 = new double [width+height];
    d3 = new double [width+height];
    a3 = new double [width+height];

    for (int k=0; k<height+width; ++k)
    {
        d1[k] = k;
        double k2 = static_cast<double>(k) * static_cast<double>(k);
        d2[k] = k2;
        d3[k] = k2 * static_cast<double>(k);
        a3[k] = pow(static_cast<double>(k - width + 1), 3);
    }
}


Moments drt_moments(const Mat& image)
{
    Size s = image.size();
    const int width = s.width;
    const int height = s.height;

    double m00, m01, m10, m11, m20, m02, m30, m12, m21, m03;

    // projection arrays
    vector<long> vert(width, 0);
    vector<long> hor(height, 0);
    vector<long> diag(width+height, 0);
    vector<long> anti(width+height, 0);

    long* hptr = &hor[0];
    long* vptr = &vert[0];
    long* dptr = &diag[0];
    long* aptr = &anti[height - 1];

    for (int i = 0; i < height; i++)
    {
        const uchar* p = image.ptr<uchar>(i);

        for(int j = 0; j < width; j++)
        {
            vptr[j] += p[j];
            hptr[i] += p[j];
            dptr[j] += p[j];
            aptr[j] += p[j];
        }

        dptr++;
        aptr--;
    }

    m00 = accumulate(begin(vert), end(vert), 0.0);
    m10 = product(vert, d1, width);
    m01 = product(hor, d1, height);
    m20 = product(vert, d2, width);
    m02 = product(hor, d2, height);
    m30 = product(vert, d3, width);
    m03 = product(hor, d3, height);
    m11 = (product(diag, d2, width+height) - m02 - m20) / 2.0;
    double temp_1 = product(diag, d3, width+height) / 6.0;
    double temp_2 = product(anti, a3, width+height) / 6.0;
    m12 = temp_1 + temp_2 - m30/3.0;
    m21 = temp_1 - temp_2 - m03/3.0;

    Moments m(m00, m10, m01, m20, m11, m02, m30, m21, m12, m03);

    return m;
}

template <typename T> 
Moments opencv_moments_template(const Mat& image)
{
    Size s = image.size();

    double m00 = 0.0, m01 = 0.0, m10 = 0.0, m11 = 0.0, m20 = 0.0, m02 = 0.0;
    double m30 = 0.0, m12 = 0.0, m21 = 0.0, m03 = 0.0;

    for(int y = 0; y < s.height; y++ )
    {
        const uchar* p = image.ptr<uchar>(y);
        T x0 = 0;
        T x1 = 0.0, x2 = 0.0, x3 = 0.0;

        for(int x = 0; x < s.width; x++ ) {
            T xp = x * p[x], xxp = x * xp;

            x0 += p[x];
            x1 += xp;
            x2 += xxp;
            x3 += xxp * x;
        }

        T py = y * x0, sy = y*y;

        m03 += py * sy;  
        m12 += x1 * sy;  
        m21 += x2 * y;  
        m30 += x3;
        m02 += x0 * sy;
        m11 += x1 * y;
        m20 += x2;
        m01 += py;
        m10 += x1;
        m00 += x0;
    }

    Moments m(m00, m10, m01, m20, m11, m02, m30, m21, m12, m03);
    return m;
}

Moments opencv_moments(const Mat& image) {
    Size s = image.size();
    if(s.height > 250 || s.width > 250)
        return opencv_moments_template<double>(image);
    else
        return opencv_moments_template<unsigned long>(image);
}

Moments naive_moments(const Mat& image)
{
    Size s = image.size();

    double m00 = 0.0, m01 = 0.0, m10 = 0.0, m11 = 0.0, m20 = 0.0, m02 = 0.0;
    double m30 = 0.0, m12 = 0.0, m21 = 0.0, m03 = 0.0;

    for(int y = 0; y < s.height; y++ )
    {
        const uchar* p = image.ptr<uchar>(y);

        for(int x = 0; x < s.width; x++ )
        {
            double xp = x * p[x], xxp = xp * x;
            double yp = y * p[x], yy = y * y;

            m00 += p[x];
            m10 += xp;
            m01 += yp;
            m11 += xp * y;
            m20 += xxp;
            m02 += yy * p[x];
            m30 += xxp * x;
            m21 += xxp * y;
            m12 += xp * yy;
            m03 += yy * yp;
        }
    }


    Moments m(m00, m10, m01, m20, m11, m02, m30, m21, m12, m03);
    return m;
}
