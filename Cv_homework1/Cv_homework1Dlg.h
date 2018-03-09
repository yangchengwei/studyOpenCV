
// Cv_homework1Dlg.h : 標頭檔
//
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <cmath>
#include <vector>
#include <string>
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;



// CCv_homework1Dlg 對話方塊
class CCv_homework1Dlg : public CDialogEx
{
// 建構
public:
	CCv_homework1Dlg(CWnd* pParent = NULL);	// 標準建構函式
	Calibration calibration;

// 對話方塊資料
	enum { IDD = IDD_CV_HOMEWORK1_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支援


// 程式碼實作
protected:
	HICON m_hIcon;

	// 產生的訊息對應函式
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedFindcorners();
	afx_msg void OnBnClickedIntrinsic();
	afx_msg void OnBnClickedExtrinsic();
	afx_msg void OnBnClickedDistortion();
	afx_msg void OnBnClickedAr();
	afx_msg void OnBnClickedTransformation();
	afx_msg void OnBnClickedDisparity();
	afx_msg void OnBnClickedLeftrightcheck();
	afx_msg void OnBnClickedSift();
};

double dist(vector<unsigned char> vector_1, vector<unsigned char> vector_2);
