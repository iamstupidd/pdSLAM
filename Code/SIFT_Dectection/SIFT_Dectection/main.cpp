#include "opencv2/opencv.hpp"  
#include"opencv2/xfeatures2d.hpp"


using namespace cv;
using namespace std;

int main() {
	Mat img = imread("C:/Users/13948/Desktop/slam/Code/SIFT_Dectection/x64/Debug/box.png");//图片路径，可以是相对路径，也可以是绝对路径
	cvtColor(img, img, COLOR_BGR2GRAY);
	Ptr<Feature2D> sift = xfeatures2d::SIFT::create();
	vector<KeyPoint> keypoints;
	Mat descriptors;

	sift->detectAndCompute(img, noArray(), keypoints, descriptors);
	drawKeypoints(img, keypoints, descriptors, Scalar(0, 255, 255));
	imshow("Result", descriptors);
	waitKey(0);

	return 0;
}
