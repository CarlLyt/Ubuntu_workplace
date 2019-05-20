#include<string.h>
#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<sstream>
#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

class Camera
{
public:
	const std::string imageDirs = "/home/carllee/workspace/ConfrimFlat/Pic/image_";
	const int imageWidth = 640;
	const int imageHight = 480;
	const double PI = 3.141592653579;
	double xpostion=0,ypostion=0,zpostion=0;
	int xMax=100,xMin=10,yMax=100,yMin=10;
	cv::Mat binary;
	cv::Mat cameraMatrix , discoffe;    //Camera parameter
	cv::VideoCapture camera;
	std::vector<cv::Point2f> corner,corner2;
	cv::Mat R,t;
	//std::vector<cv::Point> rectshow;
	//cv::Mat corner;
//	cv::Mat corner2;//corner
	int paitingPoint[480 * 2] = { 0 };
	int therehold = 0;
	int pointsums;
	int LinePoint[960][2];
	void findcorner(cv::Mat &srcImage,int cornernums, int circleratio,std::vector<cv::Point2f> &corner);
	void Getpic();
	void run();
	void start();
	void trackPeople();
	~Camera();
	Camera();
	void LKfollow(cv::Mat undisImage,cv::Mat undisImage2);
	void foundORB(cv::Mat undisImage,cv::Mat undisImage2);
	int foundFeatures(cv::Mat srcImage1,cv::Mat srcImage2);
	void prestart();
private:
	void drawPoint(std::vector<cv::Point2f> corner,cv::Mat &srcImage,int circleratio);
	void FindPoint(cv::Mat src);
	void paiting(cv::Mat src);
};
