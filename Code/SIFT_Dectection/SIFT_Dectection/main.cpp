//#include <opencv2\highgui.hpp>
//#include <opencv2/core.hpp>
//#ifdef HAVE_OPENCV_XFEATURES2D
//#include"opencv2/xfeatures2d.hpp"
//#include <opencv2/features2d.hpp>
//
//using namespace cv;
//using namespace cv::xfeatures2d;
//using namespace std;
//
//int main() {
//	Mat img1 = imread("C:/Users/13948/Desktop/slam/Code/SIFT_Dectection/SIFT_Dectection/1_optimized.jpg", IMREAD_GRAYSCALE);//ͼƬ·�������������·����Ҳ�����Ǿ���·��
//	Mat img2 = imread("C:/Users/13948/Desktop/slam/Code/SIFT_Dectection/SIFT_Dectection/2_optimized.jpg", IMREAD_GRAYSCALE);
//	int minHessian = 400;
//	Ptr<SURF> detector = SURF::create(minHessian);
//	std::vector<KeyPoint> keypoints1, keypoints2;
//	Mat descriptors1, descriptors2;
//	detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
//	detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);
//	
//	drawKeypoints(img1, keypoints1, descriptors1, Scalar(0, 255, 255));
//	drawKeypoints(img2, keypoints2, descriptors2, Scalar(255, 0, 255));
//
//	imshow("Result1", descriptors1);
//	imshow("Result2", descriptors2);
//	if (descriptors1.empty()) {
//		return 0;
//	}
//	if (descriptors2.empty()) {
//		return 0;
//	}
//	if (descriptors1.type() != CV_16F) {
//		descriptors1.convertTo(descriptors1, CV_16F);
//	}
//
//	if (descriptors2.type() != CV_16F) {
//		descriptors2.convertTo(descriptors2, CV_16F);
//	}
//	//ƥ��
//	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
//	std::vector< std::vector<DMatch> > knn_matches;
//	matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
//
//	const float ratio_thresh = 0.7f;
//	std::vector<DMatch> good_matches;
//	for (size_t i = 0; i < knn_matches.size(); i++)
//	{
//		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
//		{
//			good_matches.push_back(knn_matches[i][0]);
//		}
//	}
//	Mat img_matches;
//	drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
//		Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//	imshow("Good Matches", img_matches);
//
//	waitKey(0);
//
//	return 0;
//}
//#endif

#include <iostream>
#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;
const char* keys =
"{ help h |                          | Print help message. }"
"{ input1 | C:/Users/13948/Desktop/slam/Code/SIFT_Dectection/SIFT_Dectection/1_optimized.jpg          | Path to input image 1. }"
"{ input2 | C:/Users/13948/Desktop/slam/Code/SIFT_Dectection/SIFT_Dectection/2_optimized.jpg | Path to input image 2. }";
int main(int argc, char* argv[])
{
	CommandLineParser parser(argc, argv, keys);
	Mat img1 = imread(parser.get<String>("input1"), IMREAD_GRAYSCALE);
	Mat img2 = imread(parser.get<String>("input2"), IMREAD_GRAYSCALE);
	if (img1.empty() || img2.empty())
	{
		cout << "Could not open or find the image!\n" << endl;
		parser.printMessage();
		return -1;
	}
	//-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
	int minHessian = 400;
	Ptr<SURF> detector = SURF::create(minHessian);
	std::vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2,d2,d1;
	detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
	detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);
	drawKeypoints(img1, keypoints1, d1, Scalar(0, 255, 255));	
	drawKeypoints(img2, keypoints2, d2, Scalar(0, 255, 255));
	imshow("Result1", d1);
	imshow("Result2", d2);

	//Step 2: Matching descriptor vectors with a FLANN based matcher
	// Since SURF is a floating-point descriptor NORM_L2 is used
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	std::vector< std::vector<DMatch> > knn_matches;
	matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
	//-- Filter matches using the Lowe's ratio test
	const float ratio_thresh = 0.7f;
	std::vector<DMatch> good_matches;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		{
			good_matches.push_back(knn_matches[i][0]);
		}
	}
	//-- Draw matches
	Mat img_matches;
	drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
		Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	//-- Show detected matches
	imshow("Good Matches", img_matches);
	waitKey();
	return 0;
}
#else
int main()
{
	std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
	return 0;
}
#endif