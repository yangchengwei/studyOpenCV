
// OpenCv_homework1Dlg.h : ���Y��
//

#include <opencv2\opencv.hpp>
#include <vector>
#include <string>
#include <cmath>

using namespace cv;
using namespace std;

#pragma once


// COpenCv_homework1Dlg ��ܤ��
class COpenCv_homework1Dlg : public CDialogEx
{
// �غc
public:
	COpenCv_homework1Dlg(CWnd* pParent = NULL);	// �зǫغc�禡

// ��ܤ�����
	enum { IDD = IDD_OPENCV_HOMEWORK1_DIALOG };

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
	afx_msg void OnBnClickedOk();
	afx_msg void OnBnClickedButton1();
	afx_msg void OnBnClickedButton2();
	afx_msg void OnBnClickedButton3();
	afx_msg void OnBnClickedButton4();
	afx_msg void OnBnClickedButton5();
	afx_msg void OnBnClickedButton6();
	afx_msg void OnBnClickedButton7();
	afx_msg void OnBnClickedButton8();
	afx_msg void OnBnClickedButton9();
	afx_msg void OnBnClickedButton10();
};

void on_trackbar(int, void*);
void onMouse(int event, int x, int y, int flags, void* param);