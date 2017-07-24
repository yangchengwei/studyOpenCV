#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char* argv[]){

	std::cout<<"Start to test the members and the functions of Mat!"<<std::endl;
	namedWindow("Display Image", WINDOW_AUTOSIZE);


	std::cout<<"Show an image(240, 320, CV_8U)!"<<std::endl;
	Mat image(240, 320, CV_8U);
	imshow("Display Image", image);
	waitKey(0);


	std::cout<<"Create(300, 400, CV_8U) a new space to this image!"<<std::endl;
	image.create(300, 400, CV_8U);
	imshow("Display Image", image);
	waitKey(0);


	std::cout<<"Change image.at<uchar>(0~100,100~200) to white!"<<std::endl;
	int i = 0;
	int j = 0;
	for (i=0;i<100;i++){
		for (j=100;j<200;j++){
			image.at<uchar>(i,j) = 255;
		}
	}
	imshow("Display Image", image);
	waitKey(0);


	std::cout<<"Show an color image2(300, 400, CV_8UC3, scalar(200, 100, 0))!"<<std::endl;
	Mat image2(300, 400, CV_8UC3, Scalar(200, 100, 0));
	imshow("Display Image", image2);
	waitKey(0);


	std::cout<<"Use ptr to change all of the pixel to black!"<<std::endl;
	unsigned char* pointer = image2.ptr(0);
	for (i=0;i<300*400*3;i++){
		*pointer = 0;
		pointer++;
	}	
	imshow("Display Image", image2);
	waitKey(0);


	std::cout<<"Change image2.at<Vec3b>(0~100,100~200) to white!"<<std::endl;
	int k = 0;
	for (k=0;k<3;k++){
		std::cout<<"image2.at<Vec3b>(i,j)["<<k<<"] = 255;"<<std::endl;
		for (i=0;i<100;i++){
			for (j=100;j<200;j++){
				image2.at<Vec3b>(i,j)[k] = 255;
			}
		}
		imshow("Display Image", image2);
		waitKey(0);
	}



	return 0;
}
