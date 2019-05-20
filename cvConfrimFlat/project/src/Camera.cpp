#include<Camera.h>
#include<time.h>
Camera::Camera()
{
	std::cout<<"Camera init start~~"<<std::endl;
	camera.open(0);   //left camera
	camera.set(CV_CAP_PROP_FRAME_WIDTH,imageWidth);
	camera.set(CV_CAP_PROP_FRAME_HEIGHT,imageHight);
}
Camera::~Camera()
{
	std::cout<<"Camera exit~~"<<std::endl;

}
void Camera::drawPoint(std::vector<cv::Point2f> corners,cv::Mat &srcImage,int circleratio)
{

	for(int i = 0 ;i < (int)corners.size();i++)
	{
		cv::circle( srcImage, cv::Point(corners[i]), circleratio,  cv::Scalar(255,0,0), 2, 8, 0 );
	}
}
void loadMatrix(cv::Mat &cameraMatrix,cv::Mat &discoffe)
{
	cameraMatrix.create(3,3,CV_64FC1);
	discoffe.create(5,1,CV_64FC1);
//	602.2252  	0			331.0514
//  0  			604.1488    231.0949
//  0			0			1.0000
	cameraMatrix.at<double>(0,0)=602.2252;	cameraMatrix.at<double>(0,1)=0;			cameraMatrix.at<double>(0,2)=331.0514;
	cameraMatrix.at<double>(1,0)=0;			cameraMatrix.at<double>(1,1)=604.1488;	cameraMatrix.at<double>(1,2)=231.0949;
	cameraMatrix.at<double>(2,0)=0;			cameraMatrix.at<double>(2,1)=0;			cameraMatrix.at<double>(2,2)=1;
//
////	-0.4476    0.2181	0	0	0
	discoffe.at<double>(0,0) = -0.4476;
	discoffe.at<double>(1,0) = 0.2181;
	discoffe.at<double>(2,0) = 0;
	discoffe.at<double>(3,0) = 0;
	discoffe.at<double>(4,0) = 0;
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
void Camera::findcorner(cv::Mat &srcImage,int cornernums, int circleratio,std::vector<cv::Point2f> &corner)
{

	cv::Mat gray_image;

		cv::cvtColor(srcImage,gray_image,cv::COLOR_RGB2GRAY);
		cv::goodFeaturesToTrack(gray_image,corner,cornernums,0.05,10,cv::noArray(),3,false,0.04);
		drawPoint(corner,srcImage,circleratio);

}
void Camera::trackPeople()
{
	cv::Mat bkground;

	while(cv::waitKey(100) != 27)
	{
		camera.read(bkground);
		cv::imshow("This is a background picture without people,Enter ESC to save the bkground pic",bkground);
	}
	std::cout<<"save bkground picture success!~"<<std::endl;
	cv::imshow("This is background picture",bkground);
	cv::Mat detImg;
	std::cout<<"Detect people..."<<std::endl;
	int i,j;
	bool track = false;
	std::vector<int> point;
	while(cv::waitKey(100) != 27)
	{
		camera.read(detImg);
		for(i=0;i<detImg.rows;i++)
			for(j=0;j<detImg.cols;j++)
			{
				if((detImg.at<cv::Vec3b>(i,j)[0]- bkground.at<cv::Vec3b>(i,j)[0]) > 10
						&&(detImg.at<cv::Vec3b>(i,j)[1]- bkground.at<cv::Vec3b>(i,j)[1]) > 10
						&&(detImg.at<cv::Vec3b>(i,j)[2]- bkground.at<cv::Vec3b>(i,j)[2]) > 10)
				{
					point.push_back(i);
					point.push_back(j);
					detImg.at<cv::Vec3b>(i,j) = cv::Vec3b(255, 0, 0);
				}
			}
		cv::imshow("track window",detImg);
	}
}
void Camera::LKfollow(cv::Mat undisImage,cv::Mat undisImage2)
{
	cv::Mat grayimage,grayimage2;
	findcorner(undisImage,20,5,corner);
	cv::imshow("corner Image",undisImage);


	findcorner(undisImage2,20,5,corner2);
	cv::imshow("corner Image2",undisImage2);

	cv::cvtColor(undisImage,grayimage,cv::COLOR_RGB2GRAY);
	cv::cvtColor(undisImage2,grayimage2,cv::COLOR_RGB2GRAY);
	std::vector<uchar> featFound;
	cv::calcOpticalFlowPyrLK(grayimage,grayimage2,corner,corner2,featFound,cv::noArray(),cv::Size(21,21),5,cv::TermCriteria(cv::TermCriteria::MAX_ITER|cv::TermCriteria::EPS,20,0.3));


	for(int i =0;i<(int)corner.size();i++)
	{
	//	std::cout<<(int)featFound[i];
		if(featFound[i])
			cv::line(undisImage2,corner[i],corner2[i],cv::Scalar(0,255,0),1,8,0);

	}
	//std::cout<<std::endl;
	cv::imshow("shiTomas",undisImage2);
}
void Camera::foundORB(cv::Mat undisImage1,cv::Mat undisImage2)
{
	std::vector<cv::KeyPoint> keypoint1,keypoint2;
	cv::Mat descriptors1,descriptors2;
	cv::Ptr<cv::ORB> orb = cv::ORB::create(500,1.2f,8,31,0,2,0,31,20);

	orb->detect(undisImage1,keypoint1);
	orb->detect(undisImage2,keypoint2);

	orb->compute(undisImage1,keypoint1,descriptors1);
	orb->compute(undisImage2,keypoint2,descriptors2);

	cv::Mat outimg1;
	cv::drawKeypoints(undisImage1,keypoint1,outimg1,cv::Scalar::all(-1),cv::DrawMatchesFlags::DEFAULT);
	cv::imshow("feature1",outimg1);

	std::vector<cv::DMatch> matches;
	cv::BFMatcher matcher(cv::NORM_HAMMING);
	matcher.match(descriptors1,descriptors2,matches);

	double min_dist = 1000,max_dist = 0;

	for(int i=0;i<descriptors1.rows;i++)
	{

		double dist = matches[i].distance;
		if(dist < min_dist) min_dist = dist;
		if(dist > max_dist) max_dist = dist;
	}
	std::cout<<"----Max dist = "<< max_dist<<std::endl<<"------Min dist = "<<min_dist<<std::endl;
	std::vector<cv::DMatch> good_matches;
	for(int i =0;i<descriptors1.rows;i++)
	{
		if(matches[i].distance <= cv::max(2*min_dist ,30.0))
		{
			good_matches.push_back(matches[i]);
		}
	}

	cv::Mat img_match;
	cv::Mat img_goodmatch;
	cv::drawMatches(undisImage1,keypoint1,undisImage2,keypoint2,good_matches,img_goodmatch);
	cv::imshow("ORBmatch",img_goodmatch);
}

int  Camera::foundFeatures(cv::Mat srcImage1,cv::Mat srcImage2)
{
//	std::vector<cv::Point2f> features1,features2;
//	cv::Mat grayimage,grayimage2;
//	findcorner(srcImage1,20,5,features1);
//	cv::imshow("corner Image",srcImage1);
//
//
//	findcorner(srcImage2,20,5,features2);
//	cv::imshow("corner Image2",srcImage2);
//
//	cv::cvtColor(srcImage1,grayimage,cv::COLOR_RGB2GRAY);
//	cv::cvtColor(srcImage2,grayimage2,cv::COLOR_RGB2GRAY);
//	std::vector<uchar> featFound;
//	cv::calcOpticalFlowPyrLK(grayimage,grayimage2,features1,features2,featFound,cv::noArray(),cv::Size(21,21),5,cv::TermCriteria(cv::TermCriteria::MAX_ITER|cv::TermCriteria::EPS,20,0.3));
//
//	for(int i =0;i<(int)features1.size();i++)
//	{
//
//		if(featFound[i])
//		{
//
//
//		}
//	}
//	std::cout<<"the sums of the matches features is "<< worldNums<<std::endl;

	std::vector<cv::KeyPoint> keypoint1,keypoint2;
	std::vector<cv::Point2f> points1,points2;
	cv::Mat descriptors1,descriptors2;
	cv::Ptr<cv::ORB> orb = cv::ORB::create(500,1.2f,8,31,0,2,0,31,20);
	xMax=100,xMin=100,yMax=100,yMin=100;
	orb->detect(srcImage1,keypoint1);
	orb->detect(srcImage2,keypoint2);

	orb->compute(srcImage1,keypoint1,descriptors1);
	orb->compute(srcImage2,keypoint2,descriptors2);

	cv::Mat outimg1;
	cv::drawKeypoints(srcImage1,keypoint1,outimg1,cv::Scalar::all(-1),cv::DrawMatchesFlags::DEFAULT);
	cv::imshow("feature1",outimg1);
	//cv::Point feature;
	for(int i=0;i<keypoint1.size();i++)
	{
		//std::cout<<keypoint1[i].pt.x<<"	"<<keypoint1[i].pt.y<<"	";;
//		feature.x = keypoint1[i].pt.x;
//		feature.y = keypoint1[i].pt.y;
		if(xMin > keypoint1[i].pt.x) xMin = keypoint1[i].pt.x;
		if(xMax < keypoint1[i].pt.x) xMax = keypoint1[i].pt.x;
		if(yMin > keypoint1[i].pt.y) yMin = keypoint1[i].pt.y;
		if(yMax < keypoint1[i].pt.y) yMax = keypoint1[i].pt.y;
		//std::cout<<xMin<<"	"<<xMax<<"	"<<yMin<<"	"<<yMax<<std::endl;
	}

	std::vector<cv::DMatch> matches;
	cv::BFMatcher matcher(cv::NORM_HAMMING);
	matcher.match(descriptors1,descriptors2,matches);

//	double min_dist = 1000,max_dist = 0;
//	for(int i=0;i<descriptors1.rows;i++)
//	{
//		double dist = matches[i].distance;
//		if(dist < min_dist) min_dist = dist;
//		if(dist > max_dist) max_dist = dist;
//	}
	//std::cout<<"----Max dist = "<< max_dist<<std::endl<<"------Min dist = "<<min_dist<<std::endl;
//	std::vector<cv::DMatch> good_matches;
//	for(int i =0;i<descriptors1.rows;i++)
//	{
//		if(matches[i].distance <= cv::max(2*min_dist ,30.0))
//		{
//			good_matches.push_back(matches[i]);
//		}
//	}

	for(int i =0;i < (int)matches.size();i++)
	{
		points1.push_back(keypoint1[matches[i].queryIdx].pt);
		points2.push_back(keypoint2[matches[i].trainIdx].pt);
	}
	if((int)matches.size() <4)
		return 0;
	cv::Mat fundamental_matrix;
	fundamental_matrix = cv::findFundamentalMat(points1,points2,CV_FM_8POINT);
	//std::cout<<"fundamental_matrix is "<<fundamental_matrix<<std::endl;
	//	602.2252  	0			331.0514
	//  0  			604.1488    231.0949
	//  0			0			1.0000
	cv::Point2d principal_point(331.0514, 231.0949);
	int focal_length = 604.1488;
	cv::Mat essential_matrix;
	essential_matrix = findEssentialMat(points1,points2,focal_length,principal_point,cv::RANSAC);
	//std::cout<<"essential_matrix is "<<essential_matrix<<std::endl;

	cv::recoverPose(essential_matrix,points1,points2,R,t,focal_length,principal_point);

	return (int)matches.size();
}

void Camera::prestart()
{

	cv::Mat preimage,startimage;
	cv::Mat preimage2,startimage2;
	camera.read(preimage);
	cv::undistort(preimage,startimage,cameraMatrix,discoffe);

	cv::putText(startimage,"Please put the camera horizontally and click on the space",cv::Point(startimage.rows/8,startimage.cols/3),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(255,255,255),1,8,false);
	cv::imshow("prestart Image",startimage);
}
void Camera::FindPoint(cv::Mat src)
{   //遍历一遍图片需要0.014s
	binary = cv::Mat::zeros(cv::Size(imageWidth, imageHight), CV_32FC1);//全0矩阵
	int colsPoint[imageHight * 2] = { 0 };
	int Sumlens = 0;int position = 0; int cols, rows, pointNums = 0;
	cv::Point paiting1, paiting2;
	for (rows = 0; rows < imageHight; rows++)
	{
		int sePoint = 0, fPoint = 0, PointNums = 0;
		int maxLine = 0;
		for (cols = 0; cols < imageWidth; cols++)   //列遍历
		{

			if (float(src.at<cv::Vec3b>(rows, cols)[0]) > 100)
			{
				//paiting1.x = cols; paiting1.y = rows; cv::circle(src, paiting1, 0.1, cv::Scalar(255, 0, 0));
				colsPoint[PointNums++] = cols;
			}
		}
		for (int k = 0; k < PointNums-1; k++)
			if (colsPoint[k+1] - colsPoint[k] > maxLine)
			{
				maxLine = colsPoint[k+1] - colsPoint[k];
				Sumlens += maxLine;
				sePoint = colsPoint[k + 1]; fPoint = colsPoint[k];
			}
		paitingPoint[pointNums++] = fPoint; paitingPoint[pointNums++] = sePoint;
	}
	therehold = Sumlens / imageHight;

	//把每行黑线最长线段端点画出来，每行只保留两个端点，然后按行遍历
	for (int i = 0; i < imageHight; i++)
	{
		if((paitingPoint[position+1]- paitingPoint[position])> therehold)
		{
			/*paiting1.x = paitingPoint[position]; paiting1.y = i; cv::circle(src, paiting1, 1, cv::Scalar(255, 0, 0));
			paiting2.x = paitingPoint[position+1]; paiting2.y = i; cv::circle(src, paiting2, 1, cv::Scalar(255, 0, 0));*/
			LinePoint[pointsums][0] = paitingPoint[position]; LinePoint[pointsums +1][0] = paitingPoint[position+1];
			LinePoint[pointsums][1] = i; LinePoint[pointsums +1][1] = i;
			pointsums+=2;
		}
		position += 2;
	}

}

void Camera::paiting(cv::Mat src)
{
	//int cols, rows,pointNums=0;
	pointsums = pointsums - 2;
	cv::Point paiting;
	cv::Point c1, c2, c3, c4;
	std::cout << LinePoint[pointsums][0] << "	" << LinePoint[pointsums - 1][0] << std::endl;
	std::cout << LinePoint[pointsums][1] << "	" << LinePoint[pointsums - 1][1] << std::endl;
	std::cout << LinePoint[0][0] << "	" << LinePoint[1][0] << std::endl;
	std::cout << LinePoint[0][1] << "	" << LinePoint[1][1] << std::endl;
	c1.x = LinePoint[pointsums][0]; c2.x = LinePoint[pointsums - 1][0];
	c3.x = LinePoint[0][0]; c4.x = LinePoint[1][0];

	c1.y = LinePoint[pointsums][1]; c2.y = LinePoint[pointsums - 1][1];
	c3.y = LinePoint[0][1]; c4.y = LinePoint[1][1];

	cv::line(src, c1, c2, cv::Scalar(0, 0, 255), 4);
	cv::line(src, c3, c4, cv::Scalar(0, 255, 0), 4);
	cv::line(src, c2, c4, cv::Scalar(255, 0, 0), 4);
	cv::line(src, c1, c3, cv::Scalar(255, 255, 255), 4);
	cv::imshow("绘制点", src);
}
void Camera::start()
{
	loadMatrix(cameraMatrix,discoffe);

	std::cout<<cameraMatrix<<std::endl;
	std::cout<<"====================="<<std::endl;;
	std::cout<<discoffe<<std::endl;


	cv::Mat image,undisImage1,grayimage;
	cv::Mat image2,undisImage2,grayimage2;
	if(!camera.isOpened()) { std::cout<<"camera error"<<std::endl;			return ;}
	cv::namedWindow("prestart Image");
	while(cv::waitKey(1)!=32)
	{
		prestart();
	}
	cv::destroyWindow("prestart Image");

	cv::Mat cordinate ;
	//cordinate=cv::imread("/home/carllee/workplace/cvConfrimFlat/cordinate.jpg");
	int xangel=0,yangel=0,zangel=0;
	int featureNums;
	camera.read(image2);
	cv::undistort(image2,undisImage2,cameraMatrix,discoffe);
	cv::namedWindow("Flat");

	cv::Point point1,point2,point3,point4;

	while(cv::waitKey(1)!=32)
	{

		//cv::imshow("Camera postion",cordinate);
		camera.read(image);
		cv::undistort(image,undisImage1,cameraMatrix,discoffe);
		featureNums = foundFeatures(undisImage1,undisImage2);
		if(featureNums == 0) break;
		undisImage2 = undisImage1.clone();
			//cv::imshow("Camera postion",undisImage1);
			// R   r11  r12, r13
			//	   r21  r22  r23
			//	   r31  r32  r33
			//float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0)) ;
		zangel = atan2(R.at<double>(1,0) , R.at<double>(0,0))/ PI * 180;
		yangel = atan2( -1 * R.at<double>(2,0) , sqrt((R.at<double>(2,0))*(R.at<double>(2,0)) + (R.at<double>(2,2))*(R.at<double>(2,2)))) / PI * 180;
			//yangel = atan2( -1 * R.at<double>(2,0) , sy)/ PI * 180;
		xangel = atan2(R.at<double>(2,1),R.at<double>(2,2)) / PI * 180;

		if(xangel>=10 || xangel<=-10) continue;
		if(yangel>=10 || yangel<=-10) continue;
		if(zangel>=10 || zangel<=-10) continue;
		xpostion +=(int) xangel;
		ypostion +=(int) yangel;
		zpostion +=(int) zangel;
		 std::cout<<"X angel is "<<xpostion<<"   y angel is "<<ypostion<<"   z angel is "<<zpostion<<std::endl;
		if(xpostion >= -30 && xpostion <= -2)
		{
			point1.x = xMin;	point1.y = yMin;
			point2.x = xMin;	point2.y = yMax;
			point3.x = xMax;	point3.y = yMin;
			point4.x = xMax;	point4.y = yMax;
//			cv::line(src, c1, c2, cv::Scalar(0, 0, 255), 4);
			cv::line(undisImage1,point1,point2,cv::Scalar(0, 0, 255),4);
			cv::line(undisImage1,point2,point4,cv::Scalar(0, 255, 0),4);
			cv::line(undisImage1,point3,point4,cv::Scalar(255, 0, 0),4);
			cv::line(undisImage1,point3,point1,cv::Scalar(255, 255, 255),4);

		}
		cv::imshow("Flat",undisImage1);

		//LKfollow(undisImage,undisImage2);
		//foundORB(undisImage1,undisImage2);
	}
	cv::destroyWindow("Flat");
}
