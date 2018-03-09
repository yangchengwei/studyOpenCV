
// OpenCv_homework1Dlg.h : 標頭檔
//

#include <opencv2\opencv.hpp>
#include <vector>
#include <string>
#include <cmath>

using namespace cv;
using namespace std;

#pragma once


// COpenCv_homework1Dlg 對話方塊
class COpenCv_homework1Dlg : public CDialogEx
{
// 建構
public:
	COpenCv_homework1Dlg(CWnd* pParent = NULL);	// 標準建構函式

// 對話方塊資料
	enum { IDD = IDD_OPENCV_HOMEWORK1_DIALOG };

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