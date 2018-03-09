#pragma once


// CInputBox 對話方塊

class CInputBox : public CDialogEx
{
	DECLARE_DYNAMIC(CInputBox)

public:
	CInputBox(CWnd* pParent = NULL);   // 標準建構函式
	virtual ~CInputBox();

// 對話方塊資料
	enum { IDD = IDD_INPUTBOX };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支援

	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedOk();
	CString user_input;
};
