
// Cv_homework2.h : PROJECT_NAME ���ε{�����D�n���Y��
//

#pragma once

#ifndef __AFXWIN_H__
	#error "�� PCH �]�t���ɮ׫e���]�t 'stdafx.h'"
#endif

#include "resource.h"		// �D�n�Ÿ�


// CCv_homework2App:
// �аѾ\��@�����O�� Cv_homework2.cpp
//

class CCv_homework2App : public CWinApp
{
public:
	CCv_homework2App();

// �мg
public:
	virtual BOOL InitInstance();

// �{���X��@

	DECLARE_MESSAGE_MAP()
};

extern CCv_homework2App theApp;