#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <time.h>

using namespace cv;

void setColor(Mat image, int i, int j, int colorCase){
	switch (colorCase){
		case 0:
			// black black
			// white white
			image.at<Vec4b>(i,j) = Scalar(0,0,0,255);
			image.at<Vec4b>(i,j+1) = Scalar(0,0,0,255);
			image.at<Vec4b>(i+1,j) = Scalar(0,0,0,0);
			image.at<Vec4b>(i+1,j+1) = Scalar(0,0,0,0);
			break;
		case 1:
			// black white
			// black white
			image.at<Vec4b>(i,j) = Scalar(0,0,0,255);
			image.at<Vec4b>(i,j+1) = Scalar(0,0,0,0);
			image.at<Vec4b>(i+1,j) = Scalar(0,0,0,255);
			image.at<Vec4b>(i+1,j+1) = Scalar(0,0,0,0);
			break;
		case 2:
			// black white
			// white black
			image.at<Vec4b>(i,j) = Scalar(0,0,0,255);
			image.at<Vec4b>(i,j+1) = Scalar(0,0,0,0);
			image.at<Vec4b>(i+1,j) = Scalar(0,0,0,0);
			image.at<Vec4b>(i+1,j+1) = Scalar(0,0,0,255);
			break;
		case 3:
			// white black
			// black white
			image.at<Vec4b>(i,j) = Scalar(0,0,0,0);
			image.at<Vec4b>(i,j+1) = Scalar(0,0,0,255);
			image.at<Vec4b>(i+1,j) = Scalar(0,0,0,255);
			image.at<Vec4b>(i+1,j+1) = Scalar(0,0,0,0);
			break;
		case 4:
			// white black
			// white black
			image.at<Vec4b>(i,j) = Scalar(0,0,0,0);
			image.at<Vec4b>(i,j+1) = Scalar(0,0,0,255);
			image.at<Vec4b>(i+1,j) = Scalar(0,0,0,0);
			image.at<Vec4b>(i+1,j+1) = Scalar(0,0,0,255);
			break;
		case 5:
			// white white
			// black black
			image.at<Vec4b>(i,j) = Scalar(0,0,0,0);
			image.at<Vec4b>(i,j+1) = Scalar(0,0,0,0);
			image.at<Vec4b>(i+1,j) = Scalar(0,0,0,255);
			image.at<Vec4b>(i+1,j+1) = Scalar(0,0,0,255);
			break;
		default:
			// black black
			// black black
			image.at<Vec4b>(i,j) = Scalar(0,0,0,255);
			image.at<Vec4b>(i,j+1) = Scalar(0,0,0,255);
			image.at<Vec4b>(i+1,j) = Scalar(0,0,0,255);
			image.at<Vec4b>(i+1,j+1) = Scalar(0,0,0,255);
	}
}

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

	Mat encryptImage1(inputImage.rows, inputImage.cols, CV_8UC4);
	Mat encryptImage2(inputImage.rows, inputImage.cols, CV_8UC4);

	int i, j;
	int average;
	int colorCase;

	srand((unsigned)time(NULL));

	for (i=0;i+1<inputImage.rows;i+=2){

		for (j=0;j+1<inputImage.cols;j+=2){
			
			average = ( inputImage.at<uchar>(i,j)
			          + inputImage.at<uchar>(i,j+1)
			          + inputImage.at<uchar>(i+1,j)
			          + inputImage.at<uchar>(i+1,j+1) ) / 4;
			colorCase = rand()%6;

			if (rand()%256 < average) { // white
				setColor(encryptImage1, i, j, colorCase);
				setColor(encryptImage2, i, j, colorCase);
			}
			else {	// black
				setColor(encryptImage1, i, j, colorCase);
				setColor(encryptImage2, i, j, 5-colorCase);
			}
		}
	}

	imwrite("encryptImage1.png", encryptImage1);
	imwrite("encryptImage2.png", encryptImage2);


	return 0;
}
