#ifndef MOMEMTS
#define MOMENTS

void pre_compute_power_arrays(const cv::Size s);
void drt_images(const cv::Mat& image, cv::Mat& v, cv::Mat& h, cv::Mat& d, cv::Mat& a); 
cv::Moments drt_moments(const cv::Mat& image);
            
cv::Moments opencv_moments(const cv::Mat& image);
cv::Moments naive_moments(const cv::Mat& image);
#endif // MOMENTS
