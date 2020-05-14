#define CATCH_CONFIG_MAIN
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\opencv.hpp>
#include "catch.hpp"
#include "moments.hpp"

using namespace std;
using namespace cv;

TEST_CASE( "Raw Moments should equal OpenCV raw moments", "[moments]" ){
     Mat image;
	 image = imread("grayscale.jpg", IMREAD_GRAYSCALE);

	 Moments open_cv = moments(image, false);
	 Moments drt = get_moments(image);

	 REQUIRE( open_cv.m00 == drt.m00 );
	 REQUIRE( open_cv.m10 == drt.m10 );
	 REQUIRE( open_cv.m01 == drt.m01 );
	 REQUIRE( open_cv.m20 == drt.m20 );
	 REQUIRE( open_cv.m11 == drt.m11 );
	 REQUIRE( open_cv.m02 == drt.m02 );
	 REQUIRE( open_cv.m30 == drt.m30 );
	 REQUIRE( open_cv.m21 == drt.m21 );
     REQUIRE( open_cv.m12 == drt.m12 );
	 REQUIRE( open_cv.m03 == drt.m03 );
}
