#include <iostream>
#include<opencv2/opencv.hpp>
#include<vector>
#include<Camera.h>
using namespace std;
#define imageWidth 640
#define imageHight 480

int main() {
//	Camera cap;
//	cap.run();
	//cap.Getpic();
//	cv::namedWindow("test",CV_WINDOW_AUTOSIZE);
//	cv::Mat images = cv::imread("/home/carllee/Downloads/images.jpeg");
//
//	if(images.empty())
//	{
//	    cout<<"image not loaded";
//	    return 0;
//	}
//	cv::imshow("test",images);
//	cv::waitKey(0);

	/* ==================================================================================*/
	cv::Mat  img_1 = cv::imread("/home/carllee/workspace/ConfrimFlat/Pic/image_1.jpg");
	cv::Mat img_2;
	cv::imshow("原始图",img_1);
	cv::VideoCapture camera;
	camera.open(1);   //left camera
	camera.set(CV_CAP_PROP_FRAME_WIDTH,imageWidth);
	camera.set(CV_CAP_PROP_FRAME_HEIGHT,imageHight);
	if(!camera.isOpened()) { return 0;}
	while(cv::waitKey(1)!=27)
	{
		camera.read(img_2);
		std::vector<cv::KeyPoint> keyPoints_1, keyPoints_2;
		cv::Mat descriptors_1,descriptors_2;
		cv::imshow("show",img_2);

		cv::Ptr<cv::ORB> orb = cv::ORB::create(500, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);

		orb->detect(img_1,keyPoints_1);
		orb->detect(img_2,keyPoints_2);

		orb->compute(img_1,keyPoints_1,descriptors_1);
		orb->compute(img_2,keyPoints_2,descriptors_2);

		cv::Mat outimg1;
		cv::drawKeypoints(img_1,keyPoints_1,outimg1,cv::Scalar::all(-1),cv::DrawMatchesFlags::DEFAULT);
		cv::imshow("ORB特征点",outimg1);


		std::vector<cv::DMatch> matches;
		cv::BFMatcher matcher (cv::NORM_HAMMING);
		matcher.match(descriptors_1,descriptors_2,matches);

		double min_dist = 1000,max_dist = 0;

		for(int i =0;i<descriptors_1.rows;i++)
		{

			double dist = matches[i].distance;
			if(dist < min_dist) min_dist = dist;
			if(dist > max_dist) max_dist = dist;
		}

		std::vector<cv::DMatch> good_matches;
		for(int i=0;i<descriptors_1.rows;i++)
		{

			if(matches[i].distance <= max(2*min_dist,30.0)) good_matches.push_back (matches[i]);
		}
		cv::Mat img_goodmatch;
		cv::drawMatches(img_1,keyPoints_1,img_2,keyPoints_2,good_matches,img_goodmatch);
		cv::imshow("Good Matches",img_goodmatch);
	}


//	cv::Mat grayImage;
//	cv::cvtColor(srcImage,grayImage,CV_BGR2GRAY);
//
//	cv::OrbFeatureDetector featureDetector;
//
//
//	featureDetector.detect(grayImage,keyPoints);
//
//	cv::OrbDescriptorExtractor featureExtractor;
//	featureExtractor.compute(grayImage,keyPoints,descriptors);
//
//	cv::flann::Index flannIndex(descriptors,cv::flann::LshIndexParams(12,20,2),cv::flann::FLANN_DIST_HAMMING);
	return 0;
}
