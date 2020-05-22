#ifndef MOMEMTS
#define MOMENTS

cv::Moments drt_moments(const cv::Mat& image);
cv::Moments opencv_moments(const cv::Mat& image);
cv::Moments old_moments(const cv::Mat& image);
#endif // MOMENTS
