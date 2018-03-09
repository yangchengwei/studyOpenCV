
// Cv_homework1.h : PROJECT_NAME 應用程式的主要標頭檔
//
#pragma once
#ifndef __AFXWIN_H__
	#error "對 PCH 包含此檔案前先包含 'stdafx.h'"
#endif
#include "resource.h"		// 主要符號
#include <opencv2\opencv.hpp>
#include <vector>
#include <string>
using namespace cv;
using namespace std;
// CCv_homework1App: 
// 請參閱實作此類別的 Cv_homework1.cpp
//
class CCv_homework1App : public CWinApp
{
public:
	CCv_homework1App();

// 覆寫
public:
	virtual BOOL InitInstance();

// 程式碼實作

	DECLARE_MESSAGE_MAP()
};
extern CCv_homework1App theApp;





class Calibration{
public:
	bool paraGet;
	Size boardsize;
	CvMat* intrinsic_matrix = cvCreateMat(3, 3, CV_32FC1);
	CvMat* distortion_coeffs = cvCreateMat(5, 1, CV_32FC1);
	CvMat* translation_vector = cvCreateMat(3, 1, CV_32FC1);
	CvMat* rotation_vector = cvCreateMat(3, 1, CV_32FC1);
	CvMat* rotation_mat = cvCreateMat(3, 3, CV_32FC1);

	Calibration();
	IplImage* getCorners(string filename);
	void computeIntrinsicParameters();
	void computeExtrinsicParameters(string filename);
	Mat getImageWithPyramid(string filename);
	void transformation(string filename);
};

void onMouse(int event, int x, int y, int flags, void* param);