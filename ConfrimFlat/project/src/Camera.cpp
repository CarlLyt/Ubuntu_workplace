#include<Camera.h>

Camera::Camera()
{
	std::cout<<"Camera init start~~"<<std::endl;
	camera.open(1);   //left camera
	camera.set(CV_CAP_PROP_FRAME_WIDTH,imageWidth);
	camera.set(CV_CAP_PROP_FRAME_HEIGHT,imageHight);
}
Camera::~Camera()
{
	std::cout<<"Camera exit~~"<<std::endl;

}
void Camera::loadMatrix()
{
	cameraMatrix.create(3,3,CV_64FC1);
	discoffe.create(5,1,CV_64FC1);
//	602.2252  	331.0514  	231.0949
//  0  			604.1488    0
//  0			0			1.0000
	cameraMatrix.at<double>(0,0)=602.2252;cameraMatrix.at<double>(0,1)=0;cameraMatrix.at<double>(0,2)=331.0514;
	cameraMatrix.at<double>(1,0)=0;cameraMatrix.at<double>(1,1)=604.1488;cameraMatrix.at<double>(1,2)=231.0949;
	cameraMatrix.at<double>(2,0)=0;cameraMatrix.at<double>(2,1)=0;cameraMatrix.at<double>(2,2)=1;

//	-0.4476    0.2181	0	0	0
	discoffe.at<double>(0,0) = -0.4476;
	discoffe.at<double>(1,0) = 0.2181;
	discoffe.at<double>(3,0) = 0;
	discoffe.at<double>(4,0) = 0;
	discoffe.at<double>(5,0) = 0;
}
void Camera::Getpic()
{
	cv::Mat image;
	int num = 0;
	std::stringstream buf;
	while(1)
	{
		if(!camera.isOpened()) { return ;}
		camera.read(image);
		imshow("Q: Quit/  space board:take photo",image);
		if(cv::waitKey(1) == 32)
		{
			buf << num++;
			std::string imageName = imageDirs + buf.str() +".jpg";
			std::cout<<imageName<<std::endl;
			cv::imwrite(imageName,image);
			buf.str("");
			std::cout<<"nums of image is: "<<num-1<<std::endl;
		}

		if(cv::waitKey(1) == 'q') break;
	}
}
void Camera::ORB_match()
{

}
void Camera::run()
{
	loadMatrix();
	std::cout<<cameraMatrix<<"	"<<discoffe<<std::endl;
	cv::Mat image,undisImage;
	if(!camera.isOpened()) { std::cout<<"camera error"<<std::endl;return ;}

	while(cv::waitKey(1)!=27)
	{
		camera.read(image);
		if(image.empty())
		{
			std::cout<<"image empty"<<std::endl;
		}
		else
		{
		cv::imshow("test",image);
		cv::undistort(image,undisImage,cameraMatrix,discoffe);
		cv::imshow("undistort Image",undisImage);
		}
	}
}
