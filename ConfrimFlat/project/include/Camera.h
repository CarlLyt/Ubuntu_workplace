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
	cv::Mat cameraMatrix , discoffe;    //Camera parameter
	cv::VideoCapture camera;
	void ORB_match();
	void Getpic();
	void loadMatrix();
	void run();
	~Camera();
	Camera();

};
