#include <cstdio>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
using namespace cv;
using namespace std;

vector<string> m_filenames;
void CameraCalibratorfile() {
	m_filenames.clear();
	m_filenames.push_back("calibration/1.bmp");
	m_filenames.push_back("calibration/2.bmp");
	m_filenames.push_back("calibration/3.bmp");
	m_filenames.push_back("calibration/4.bmp");
	m_filenames.push_back("calibration/5.bmp");
	m_filenames.push_back("calibration/6.bmp");
	m_filenames.push_back("calibration/7.bmp");
	m_filenames.push_back("calibration/8.bmp");
	m_filenames.push_back("calibration/9.bmp");
	m_filenames.push_back("calibration/10.bmp");
	m_filenames.push_back("calibration/11.bmp");
	m_filenames.push_back("calibration/12.bmp");
	m_filenames.push_back("calibration/13.bmp");
	m_filenames.push_back("calibration/14.bmp");
	m_filenames.push_back("calibration/15.bmp");
	m_filenames.push_back("calibration/16.bmp");
	m_filenames.push_back("calibration/17.bmp");
	m_filenames.push_back("calibration/18.bmp");
	m_filenames.push_back("calibration/19.bmp");
	m_filenames.push_back("calibration/20.bmp");
	m_filenames.push_back("calibration/21.bmp");
}



int main() {



	CameraCalibratorfile();

	for (int i = 0; i < m_filenames.size(); i++) {
		Mat image_color = cv::imread(m_filenames[i], cv::IMREAD_COLOR);
		cv::Mat image_gray;

		cv::cvtColor(image_color, image_gray, cv::COLOR_BGR2GRAY);

		std::vector<cv::Point2f> corners;

		bool ret = cv::findChessboardCorners(image_gray,
			cv::Size(8, 11),
			corners,
			cv::CALIB_CB_ADAPTIVE_THRESH |
			cv::CALIB_CB_NORMALIZE_IMAGE);


		//指定?像素?算迭代?注  
		cv::TermCriteria criteria = cv::TermCriteria(
			cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS,
			30,
			0.1);

		//?像素??  
		cv::cornerSubPix(image_gray, corners, cv::Size(5, 5), cv::Size(-1, -1), criteria);

		//角??制  
		cv::drawChessboardCorners(image_color, cv::Size(8, 11), corners, ret);

		cv::imshow("chessboard corners", image_color);
		cv::waitKey(100);

	}
	return 0;
}