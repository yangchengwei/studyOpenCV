// InputBox.cpp : ��@��
//

#include "stdafx.h"
#include "Cv_homework2.h"
#include "InputBox.h"
#include "afxdialogex.h"


// CInputBox ��ܤ��

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


// CInputBox �T���B�z�`��


void CInputBox::OnBnClickedOk()
{
	// TODO: �b���[�J����i���B�z�`���{���X
	CDialogEx::OnOK();
}