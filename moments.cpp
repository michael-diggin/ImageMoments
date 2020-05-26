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

double product(const vector<long> &mat, double power[], int many)
{
    double sum = 0.0;
    for(int i = 0; i < many; i++)
        sum += static_cast<double>(mat[i]) * power[i];

    return sum;
}

Moments drt_moments(const Mat& image)
{

    Size s = image.size();
    const int width = s.width;
    const int height = s.height;

    double m00, m01, m10, m11, m20, m02, m30, m12, m21, m03;

    //power arrays
    double *d1 = new double [width+height];
    double *d2 = new double [width+height];
    double *d3 = new double [width+height];
    double *a3 = new double [width+height];

    for (int k=0; k<height+width; ++k)
    {
        d1[k] = k;
        int k2 = k*k;
        d2[k] = k2;
        d3[k] = k2*k;
        a3[k] = pow((k - width + 1), 3);
    }

    // loop through image by pixel values
    //projection arrays
    vector<long> vert(width, 0);
    vector<long> hor(height, 0);
    vector<long> diag(width+height, 0);
    vector<long> anti(width+height, 0);

    long* hptr = &hor[0];
    long* initvptr = &vert[0];
    long* vptr;
    long* dptr;
    long* aptr;

    for (int i=0; i< height; ++i)
    {
        vptr = initvptr;
        dptr = &diag[i];
        aptr = &anti[height-1-i];
        const uchar* p = image.ptr<uchar>(i);

        long h = 0;

        for(int j = 0; j < width; ++j)
        {
            vptr[j] += p[j];
            h += p[j];
            dptr[j] += p[j];
            aptr[j] += p[j];
        }
        hptr[i] = h;

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

Moments opencv_moments(const Mat& image)
{

    Size s = image.size();


    double m00 = 0.0, m01 = 0.0, m10 = 0.0, m11 = 0.0, m20 = 0.0, m02 = 0.0;
    double m30 = 0.0, m12 = 0.0, m21 = 0.0, m03 = 0.0;

    //double t = (double)getTickCount();

    for(int y = 0; y < s.height; y++ )
    {
        const uchar* ptr = image.ptr<uchar>(y);
        long x0 = 0;
        double x1 = 0.0, x2 = 0.0, x3 = 0.0;

        for(int x=0; x < s.width; x++ )
        {
            int p = *(ptr++);
            double xp = x * p, xxp;

            x0 += p;
            x1 += xp;
            xxp = xp * x;
            x2 += xxp;
            x3 += xxp * x;
        }

        double py = y * x0, sy = y*y;

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

    //t = ((double)getTickCount() -t)/getTickFrequency();
    //cout << "OCV Loop Time: " << t << endl;

    Moments m(m00, m10, m01, m20, m11, m02, m30, m21, m12, m03);
    return m;

}


Moments old_moments(const Mat& image)
{
    Size s = image.size();

    double m00 = 0.0, m01 = 0.0, m10 = 0.0, m11 = 0.0, m20 = 0.0, m02 = 0.0;
    double m30 = 0.0, m12 = 0.0, m21 = 0.0, m03 = 0.0;

    // double t = (double)getTickCount();

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

    // t = ((double)getTickCount() -t)/getTickFrequency();
    // cout << "OLD Loop Time: " << t << endl;

    Moments m(m00, m10, m01, m20, m11, m02, m30, m21, m12, m03);
    return m;
}
