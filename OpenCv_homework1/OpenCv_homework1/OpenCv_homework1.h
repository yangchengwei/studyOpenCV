
// OpenCv_homework1.h : PROJECT_NAME ���ε{�����D�n���Y��
//

#pragma once

#ifndef __AFXWIN_H__
	#error "�� PCH �]�t���ɮ׫e���]�t 'stdafx.h'"
#endif

#include "resource.h"		// �D�n�Ÿ�


// COpenCv_homework1App: 
// �аѾ\��@�����O�� OpenCv_homework1.cpp
//

class COpenCv_homework1App : public CWinApp
{
public:
	COpenCv_homework1App();

// �мg
public:
	virtual BOOL InitInstance();

// �{���X��@

	DECLARE_MESSAGE_MAP()
};

extern COpenCv_homework1App theApp;