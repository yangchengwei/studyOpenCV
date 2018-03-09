// InputBox.cpp : 實作檔
//

#include "stdafx.h"
#include "Cv_homework2.h"
#include "InputBox.h"
#include "afxdialogex.h"


// CInputBox 對話方塊

IMPLEMENT_DYNAMIC(CInputBox, CDialogEx)

CInputBox::CInputBox(CWnd* pParent /*=NULL*/)
	: CDialogEx(CInputBox::IDD, pParent)
	, user_input(_T(""))
{

}

CInputBox::~CInputBox()
{
}

void CInputBox::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_INPUT, user_input);
}


BEGIN_MESSAGE_MAP(CInputBox, CDialogEx)
	ON_BN_CLICKED(IDOK, &CInputBox::OnBnClickedOk)
//	ON_EN_CHANGE(IDC_INPUT, &CInputBox::OnEnChangeInput)
END_MESSAGE_MAP()


// CInputBox 訊息處理常式


void CInputBox::OnBnClickedOk()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	CDialogEx::OnOK();
}