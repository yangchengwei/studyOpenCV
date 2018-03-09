
// Cv_homework1Dlg.h : ���Y��
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



// CCv_homework1Dlg ��ܤ��
class CCv_homework1Dlg : public CDialogEx
{
// �غc
public:
	CCv_homework1Dlg(CWnd* pParent = NULL);	// �зǫغc�禡
	Calibration calibration;

// ��ܤ�����
	enum { IDD = IDD_CV_HOMEWORK1_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV �䴩


// �{���X��@
protected:
	HICON m_hIcon;

	// ���ͪ��T�������禡
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
