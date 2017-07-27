#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char* argv[]){

	if (argc!=2){
		std::cout<<"usage: ./photoEncryption <image_path>"<<std::endl;
		return -1;
	}

	Mat inputImage;

	inputImage = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	namedWindow("Display Image", WINDOW_AUTOSIZE);
	imshow("Display Image", inputImage);
	waitKey(0);

	Mat encryptImage1(inputImage.rows, inputImage.cols, CV_8U);
	Mat encryptImage2(inputImage.rows, inputImage.cols, CV_8U);

	

	waitKey(0);
	return 0;
}
