////#include <opencv2\highgui.hpp>
////#include <opencv2/core.hpp>
////#ifdef HAVE_OPENCV_XFEATURES2D
////#include"opencv2/xfeatures2d.hpp"
////#include <opencv2/features2d.hpp>
////
////using namespace cv;
////using namespace cv::xfeatures2d;
////using namespace std;
////
////int main() {
////	Mat img1 = imread("C:/Users/13948/Desktop/slam/Code/SIFT_Dectection/SIFT_Dectection/1_optimized.jpg", IMREAD_GRAYSCALE);//图片路径，可以是相对路径，也可以是绝对路径
////	Mat img2 = imread("C:/Users/13948/Desktop/slam/Code/SIFT_Dectection/SIFT_Dectection/2_optimized.jpg", IMREAD_GRAYSCALE);
////	int minHessian = 400;
////	Ptr<SURF> detector = SURF::create(minHessian);
////	std::vector<KeyPoint> keypoints1, keypoints2;
////	Mat descriptors1, descriptors2;
////	detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
////	detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);
////	
////	drawKeypoints(img1, keypoints1, descriptors1, Scalar(0, 255, 255));
////	drawKeypoints(img2, keypoints2, descriptors2, Scalar(255, 0, 255));
////
////	imshow("Result1", descriptors1);
////	imshow("Result2", descriptors2);
////	if (descriptors1.empty()) {
////		return 0;
////	}
////	if (descriptors2.empty()) {
////		return 0;
////	}
////	if (descriptors1.type() != CV_16F) {
////		descriptors1.convertTo(descriptors1, CV_16F);
////	}
////
////	if (descriptors2.type() != CV_16F) {
////		descriptors2.convertTo(descriptors2, CV_16F);
////	}
////	//匹配
////	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
////	std::vector< std::vector<DMatch> > knn_matches;
////	matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
////
////	const float ratio_thresh = 0.7f;
////	std::vector<DMatch> good_matches;
////	for (size_t i = 0; i < knn_matches.size(); i++)
////	{
////		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
////		{
////			good_matches.push_back(knn_matches[i][0]);
////		}
////	}
////	Mat img_matches;
////	drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
////		Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
////	imshow("Good Matches", img_matches);
////
////	waitKey(0);
////
////	return 0;
////}
////#endif
//
//#include <iostream>
//#include "opencv2/core.hpp"
//#ifdef HAVE_OPENCV_XFEATURES2D
//#include "opencv2/highgui.hpp"
//#include "opencv2/features2d.hpp"
//#include "opencv2/xfeatures2d.hpp"
//using namespace cv;
//using namespace cv::xfeatures2d;
//using std::cout;
//using std::endl;
//const char* keys =
//"{ help h |                          | Print help message. }"
//"{ input1 | C:/Users/13948/Desktop/slam/Code/SIFT_Dectection/SIFT_Dectection/1_optimized.jpg          | Path to input image 1. }"
//"{ input2 | C:/Users/13948/Desktop/slam/Code/SIFT_Dectection/SIFT_Dectection/2_optimized.jpg | Path to input image 2. }";
//int main(int argc, char* argv[])
//{
//	CommandLineParser parser(argc, argv, keys);
//	Mat img1 = imread(parser.get<String>("input1"), IMREAD_GRAYSCALE);
//	Mat img2 = imread(parser.get<String>("input2"), IMREAD_GRAYSCALE);
//	if (img1.empty() || img2.empty())
//	{
//		cout << "Could not open or find the image!\n" << endl;
//		parser.printMessage();
//		return -1;
//	}
//	//-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
//	int minHessian = 400;
//	Ptr<SURF> detector = SURF::create(minHessian);
//	std::vector<KeyPoint> keypoints1, keypoints2;
//	Mat descriptors1, descriptors2,d2,d1;
//	detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
//	detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);
//	drawKeypoints(img1, keypoints1, d1, Scalar(0, 255, 255));	
//	drawKeypoints(img2, keypoints2, d2, Scalar(0, 255, 255));
//	imshow("Result1", d1);
//	imshow("Result2", d2);
//
//	//Step 2: Matching descriptor vectors with a FLANN based matcher
//	// Since SURF is a floating-point descriptor NORM_L2 is used
//	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
//	std::vector< std::vector<DMatch> > knn_matches;
//	matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
//	//-- Filter matches using the Lowe's ratio test
//	const float ratio_thresh = 0.7f;
//	std::vector<DMatch> good_matches;
//	for (size_t i = 0; i < knn_matches.size(); i++)
//	{
//		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
//		{
//			good_matches.push_back(knn_matches[i][0]);
//		}
//	}
//	//-- Draw matches
//	Mat img_matches;
//	drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
//		Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//	//-- Show detected matches
//	imshow("Good Matches", img_matches);
//	waitKey();
//	return 0;
//}
//#else
//int main()
//{
//	std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
//	return 0;
//}
//#endif

/* opencv */
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\features2d.hpp>
#include <opencv2\xfeatures2d.hpp>
#include <opencv2\calib3d.hpp>
/* c++ */
#include <iostream>
#include <vector>


using namespace std;
using namespace cv::xfeatures2d;
using namespace cv;

int main() {
	Mat ori = imread("C:/Users/13948/Desktop/slam/Code/SIFT_Dectection/SIFT_Dectection/book_image/test/ori.jpg", IMREAD_GRAYSCALE);
	Mat show_ori = ori.clone();
	int hessian = 400;
	Ptr<SURF> detector = SURF::create(hessian);
	vector<KeyPoint> oriKeyPoints;
	Mat ori_descripters;
	detector->detectAndCompute(ori, noArray(), oriKeyPoints, ori_descripters);
	drawKeypoints(show_ori, oriKeyPoints, show_ori, Scalar(255, 0, 0));
	vector<string> books = { "操作系统教程.jpg","操作系统黑书.jpg","离散数学.jpg","概率论.jpg","数据库.jpg" ,"线性代数.jpg"};
	Mat res_book, res_match;
	int maxlen = 0;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	//Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
	vector< vector<DMatch> > knn_matches;

	for (auto element : books) {
		knn_matches.clear();
		string cur = "C:/Users/13948/Desktop/slam/Code/SIFT_Dectection/SIFT_Dectection/book_image/" + element;
		Mat curImg = imread(cur, IMREAD_GRAYSCALE);
		vector<KeyPoint> curKeyPoints;
		Mat cur_descripters;
		detector->detectAndCompute(curImg, noArray(), curKeyPoints, cur_descripters);
		matcher->knnMatch(ori_descripters, cur_descripters, knn_matches, 2);
		const float ratio_thresh = 0.8f;
		std::vector<DMatch> good_matches;
		for (size_t i = 0; i < knn_matches.size(); i++)
		{
			if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
			{
				good_matches.push_back(knn_matches[i][0]);
			}
		}
		cout << good_matches.size() << endl;
		//ransac
		if (good_matches.size() >= 5) {
				int ptCount = (int)good_matches.size();
				Mat p1(good_matches.size(), 2, CV_32F);
				Mat p2(good_matches.size(), 2, CV_32F);
				Point2f pt;
				for (int i = 0; i<ptCount; i++)
				{
					pt = oriKeyPoints[good_matches[i].queryIdx].pt;
					p1.at<float>(i, 0) = pt.x;
					p1.at<float>(i, 1) = pt.y;
					pt = curKeyPoints[good_matches[i].trainIdx].pt;
					p2.at<float>(i, 0) = pt.x;
					p2.at<float>(i, 1) = pt.y;
				}
		
				Mat m_Fundamental;
				vector<uchar> m_RANSACStatus;
				m_Fundamental = findFundamentalMat(p1, p2, m_RANSACStatus,RANSAC);
				int OutlinerCount = 0;
				for (int i = 0; i<ptCount; i++)
				{
					if (m_RANSACStatus[i] == 0)    // 状态为0表示外点
					{
						OutlinerCount++;
					}
				}

				int InlinerCount = ptCount - OutlinerCount;   // 计算内点   
				cout << "内点数为：" << InlinerCount << endl;
				cout << "外点数为：" << OutlinerCount << endl;
				vector<Point2f> m_LeftInlier;
				vector<Point2f> m_RightInlier;
				vector<DMatch> m_InlierMatches;
				m_InlierMatches.resize(InlinerCount);
				m_LeftInlier.resize(InlinerCount);
				m_RightInlier.resize(InlinerCount);
				InlinerCount = 0;
				float inlier_minRx = ori.cols;
				for (int i = 0; i<ptCount; i++)
				{
					if (m_RANSACStatus[i] != 0)
					{
						m_LeftInlier[InlinerCount].x = p1.at<float>(i, 0);
						m_LeftInlier[InlinerCount].y = p1.at<float>(i, 1);
						m_RightInlier[InlinerCount].x = p2.at<float>(i, 0);
						m_RightInlier[InlinerCount].y = p2.at<float>(i, 1);
						m_InlierMatches[InlinerCount].queryIdx = InlinerCount;
						m_InlierMatches[InlinerCount].trainIdx = InlinerCount;

						if (m_RightInlier[InlinerCount].x<inlier_minRx) inlier_minRx = m_RightInlier[InlinerCount].x;   //存储内点中右图最小横坐标
						InlinerCount++;
					}
				}
				vector<KeyPoint> key1(InlinerCount);
				vector<KeyPoint> key2(InlinerCount);
				KeyPoint::convert(m_LeftInlier, key1);
				KeyPoint::convert(m_RightInlier, key2);
				Mat OutImage;
				drawMatches(ori, key1, curImg, key2, m_InlierMatches, OutImage, Scalar(255, 0, 255), Scalar(0, 255, 0));
				if (m_InlierMatches.size() > maxlen) {
					maxlen = good_matches.size();
					res_book = curImg;
					drawMatches(ori, key1, curImg, key2, m_InlierMatches, res_match, Scalar::all(-1),
						Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
				}
		
				imshow("ransac", OutImage);
				waitKey(1000);
		}
		
		else {
			Mat tmp_match;
			drawMatches(ori, oriKeyPoints, curImg, curKeyPoints, good_matches, tmp_match, Scalar::all(-1),
				Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
			imshow("temp", tmp_match);
			waitKey(1000);
			if (good_matches.size() > maxlen) {
				maxlen = good_matches.size();
				res_book = curImg;
				drawMatches(ori, oriKeyPoints, curImg, curKeyPoints, good_matches, res_match, Scalar::all(-1),
					Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
			}
		}
		//if (good_matches.size() > maxlen) {
		//	maxlen = good_matches.size();
		//	res_book = curImg;
		//	drawMatches(ori, oriKeyPoints, curImg, curKeyPoints, good_matches, res_match, Scalar::all(-1),
		//		Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		//}
		//Mat tmp_match;
		//if (good_matches.size() > 0) {
		//	drawMatches(ori, oriKeyPoints, curImg, curKeyPoints, good_matches, tmp_match, Scalar::all(-1),
		//		Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		//	imshow("temp", tmp_match);
		//	waitKey(1000);
		//}

	}
	assert(maxlen != 0);
	imshow("原图特征点", show_ori);
	imshow("匹配结果", res_match);
	imshow("最匹配书籍", res_book);
	waitKey(0);
	return 0;
}