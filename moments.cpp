#include <iostream>
#include <vector>
#include <numeric>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

Moments get_moments(const Mat& image)
{

    Size s = image.size();
    if ( s.width <=0 || s.height <=0 ){
        Moments zero_m;
        return zero_m;
    }

    //projection arrays
    vector<int> vert(s.width, 0);
    vector<int> hor(s.height, 0);
    vector<int> diag(s.width+s.height, 0);
    vector<int> anti(s.width+s.height, 0);



    //power arrays
    vector<int> d1(s.width+s.height, 0);
    vector<int> d2(s.width+s.height, 0);
    vector<int> d3(s.width+s.height, 0);
    vector<int> a3(s.width+s.height, 0);


    for (int k=0; k<s.height+s.width; k++)
    {
        d1[k] = k;
        d2[k] = pow(k, 2);
        d3[k] = d2[k]*k;
        a3[k] = pow((k - s.width + 1), 3);
    }

    // loop through image by pixel values
    double m00, m01, m10, m11, m20, m02, m30, m12, m21, m03;

    for (int i=0; i< s.width; i++)
    {
        const uchar* p = image.ptr<uchar>(i);
        for (int j=0; j<s.height; j++)
        {
            vert[i] += p[j];
            hor[j] += p[j];
            diag[i+j] += p[j];
            int k = s.width -1 + j - i;
            anti[k] += p[j];
        }
    }

    m00 = accumulate(begin(vert), end(vert), 0.0);
    m10 = inner_product(begin(hor), end(hor), begin(d1), 0.0);
    m01 = inner_product(begin(vert), end(vert), begin(d1), 0.0);
    m20 = inner_product(begin(hor), end(hor), begin(d2), 0.0);
    m02 = inner_product(begin(vert), end(vert), begin(d2), 0.0);
    m30 = inner_product(begin(hor), end(hor), begin(d3), 0.0);
    m03 = inner_product(begin(vert), end(vert), begin(d3), 0.0);
    m11 = inner_product(begin(diag), end(diag), begin(d2), 0.0)/2.0 - m02/2.0 - m20/2.0;
    double temp_1 = inner_product(begin(diag), end(diag), begin(d3), 0.0)/6.0;
    double temp_2 = inner_product(begin(anti), end(anti), begin(a3), 0.0)/6.0;
    m12 = temp_1 + temp_2 - m30/3.0;
    m21 = temp_1 - temp_2 - m03/3.0;

    Moments m(m00, m10, m01, m20, m11, m02, m30, m21, m12, m03);
    return m;
}
