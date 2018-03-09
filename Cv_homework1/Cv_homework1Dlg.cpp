
// Cv_homework1Dlg.cpp : 實作檔
//
#include "stdafx.h"
#include "Cv_homework1.h"
#include "Cv_homework1Dlg.h"
#include "afxdialogex.h"


#ifdef _DEBUG
#define new DEBUG_NEW
#endif
// 對 App About 使用 CAboutDlg 對話方塊
class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

	// 對話方塊資料
	enum { IDD = IDD_ABOUTBOX };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支援

	// 程式碼實作
protected:
	DECLARE_MESSAGE_MAP()
};
CAboutDlg::CAboutDlg() : CDialogEx(CAboutDlg::IDD)
{
}
void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}
BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()
// CCv_homework1Dlg 對話方塊
CCv_homework1Dlg::CCv_homework1Dlg(CWnd* pParent /*=NULL*/)
: CDialogEx(CCv_homework1Dlg::IDD, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}
void CCv_homework1Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}
BEGIN_MESSAGE_MAP(CCv_homework1Dlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(ID_FindCorners, &CCv_homework1Dlg::OnBnClickedFindcorners)
	ON_BN_CLICKED(ID_Intrinsic, &CCv_homework1Dlg::OnBnClickedIntrinsic)
	ON_BN_CLICKED(ID_Extrinsic, &CCv_homework1Dlg::OnBnClickedExtrinsic)
	ON_BN_CLICKED(ID_Distortion, &CCv_homework1Dlg::OnBnClickedDistortion)
	ON_BN_CLICKED(ID_AR, &CCv_homework1Dlg::OnBnClickedAr)
	ON_BN_CLICKED(ID_Transformation, &CCv_homework1Dlg::OnBnClickedTransformation)
	ON_BN_CLICKED(ID_Disparity, &CCv_homework1Dlg::OnBnClickedDisparity)
	ON_BN_CLICKED(ID_LeftRightCheck, &CCv_homework1Dlg::OnBnClickedLeftrightcheck)
	ON_BN_CLICKED(ID_SIFT, &CCv_homework1Dlg::OnBnClickedSift)
END_MESSAGE_MAP()
// CCv_homework1Dlg 訊息處理常式
BOOL CCv_homework1Dlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 將 [關於...] 功能表加入系統功能表。

	// IDM_ABOUTBOX 必須在系統命令範圍之中。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 設定此對話方塊的圖示。當應用程式的主視窗不是對話方塊時，
	// 框架會自動從事此作業
	SetIcon(m_hIcon, TRUE);			// 設定大圖示
	SetIcon(m_hIcon, FALSE);		// 設定小圖示

	// TODO:  在此加入額外的初始設定
	AllocConsole();
	freopen("CONOUT$", "w", stdout);

	return TRUE;  // 傳回 TRUE，除非您對控制項設定焦點
}
void CCv_homework1Dlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}
// 如果將最小化按鈕加入您的對話方塊，您需要下列的程式碼，
// 以便繪製圖示。對於使用文件/檢視模式的 MFC 應用程式，
// 框架會自動完成此作業。
void CCv_homework1Dlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 繪製的裝置內容

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 將圖示置中於用戶端矩形
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 描繪圖示
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}
// 當使用者拖曳最小化視窗時，
// 系統呼叫這個功能取得游標顯示。
HCURSOR CCv_homework1Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}





void CCv_homework1Dlg::OnBnClickedFindcorners()
{
	// TODO:  在此加入控制項告知處理常式程式碼
	for (int i = 1; i <= 1/*21*/; i++){
		string filename = "Database\\" + to_string(i) + ".bmp";
		IplImage *image;
		image = calibration.getCorners(filename);
		cvShowImage("1. Find Corners", image);
		waitKey(0);
		cvDestroyAllWindows();
	}
}
void CCv_homework1Dlg::OnBnClickedIntrinsic()
{
	// TODO:  在此加入控制項告知處理常式程式碼
	if (!calibration.paraGet){
		calibration.computeIntrinsicParameters();
	}

	vector<vector<float>> intrinsicMatrix{ { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			intrinsicMatrix[i][j] = CV_MAT_ELEM(*calibration.intrinsic_matrix, float, i, j);
		}
	}
	cout << "Intrinsic Matrix :" << endl;
	cout << "[" << intrinsicMatrix[0][0] << ", " << intrinsicMatrix[0][1] << ", " << intrinsicMatrix[0][2] << ";" << endl;
	cout << " " << intrinsicMatrix[1][0] << ", " << intrinsicMatrix[1][1] << ", " << intrinsicMatrix[1][2] << ";" << endl;
	cout << " " << intrinsicMatrix[2][0] << ", " << intrinsicMatrix[2][1] << ", " << intrinsicMatrix[2][2] << "]" << endl << endl;
}
void CCv_homework1Dlg::OnBnClickedExtrinsic()
{
	// TODO:  在此加入控制項告知處理常式程式碼
	string filename = "Database/1.bmp";
	calibration.computeExtrinsicParameters(filename);

	vector<vector<float>> rotationMatrix{ { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };
	vector<float> translationMatrix = { 0, 0, 0 };

	for (int i = 0; i < 3; i++)
	{
		translationMatrix[i] = CV_MAT_ELEM(*calibration.translation_vector, float, i, 0);
		for (int j = 0; j < 3; j++)
		{
			rotationMatrix[i][j] = CV_MAT_ELEM(*calibration.rotation_mat, float, i, j);
		}
	}
	cout << "Extrinsic Matrix :" << endl;
	cout << "[" << rotationMatrix[0][0] << ", " << rotationMatrix[0][1] << ", " << rotationMatrix[0][2] << ", " << translationMatrix[0] << ";" << endl;
	cout << " " << rotationMatrix[1][0] << ", " << rotationMatrix[1][1] << ", " << rotationMatrix[1][2] << ", " << translationMatrix[1] << ";" << endl;
	cout << " " << rotationMatrix[2][0] << ", " << rotationMatrix[2][1] << ", " << rotationMatrix[2][2] << ", " << translationMatrix[2] << "]" << endl << endl;
	/*for (int i = 1; i < 1; i++){
		string filename = "Database\\"+to_string(i)+".bmp";
		calibration.computeExtrinsicParameters(filename);

		vector<vector<float>> rotationMatrix{ { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };
		vector<float> translationMatrix = { 0, 0, 0 };

		for (int i = 0; i < 3; i++)
		{
		translationMatrix[i] = CV_MAT_ELEM(*calibration.translation_vector, float, i, 0);
		for (int j = 0; j < 3; j++)
		{
		rotationMatrix[i][j] = CV_MAT_ELEM(*calibration.rotation_mat, float, i, j);
		}
		}
		cout << "Extrinsic Matrix :" << endl;
		cout << i <<"   [" << rotationMatrix[0][0] << ", " << rotationMatrix[0][1] << ", " << rotationMatrix[0][2] << ", " << translationMatrix[0] << ";" << endl;
		cout << " " << rotationMatrix[1][0] << ", " << rotationMatrix[1][1] << ", " << rotationMatrix[1][2] << ", " << translationMatrix[1] << ";" << endl;
		cout << " " << rotationMatrix[2][0] << ", " << rotationMatrix[2][1] << ", " << rotationMatrix[2][2] << ", " << translationMatrix[2] << "]" << endl << endl;
		}*/
}
void CCv_homework1Dlg::OnBnClickedDistortion()
{
	// TODO:  在此加入控制項告知處理常式程式碼
	if (!calibration.paraGet){
		calibration.computeIntrinsicParameters();
	}
	vector<float> distortionMatrix{ 0, 0, 0, 0, 0 };

	//cout << CV_MAT_ELEM(*distortion_coeffs, float, 0, 0) << " " << CV_MAT_ELEM(*distortion_coeffs, float, 1, 0) << " " << CV_MAT_ELEM(*distortion_coeffs, float, 2, 0) << " " << CV_MAT_ELEM(*distortion_coeffs, float, 3, 0) << " " << CV_MAT_ELEM(*distortion_coeffs, float, 4, 0) << endl;
	for (int i = 0; i < 5; i++)
	{
		distortionMatrix[i] = CV_MAT_ELEM(*calibration.distortion_coeffs, float, i, 0);
	}

	cout << "Distortion Matrix :" << endl;
	cout << "[" << distortionMatrix[0] << ", " << distortionMatrix[1] << ", " << distortionMatrix[2] << ", " << distortionMatrix[3] << ", " << distortionMatrix[4] << "]" << endl << endl;
}
void CCv_homework1Dlg::OnBnClickedAr()
{
	for (int i = 1; i <= 5; i++){
		string filename = "Database/"+to_string(i)+".bmp";
		Mat output = calibration.getImageWithPyramid(filename);
		imshow("2. AR", output);
		waitKey(500);
	}
	waitKey(0);
	cvDestroyAllWindows();
}
void CCv_homework1Dlg::OnBnClickedTransformation()
{
	// TODO:  在此加入控制項告知處理常式程式碼
	string filename = "Database/QrCode.jpg";
	calibration.transformation(filename);
}
void CCv_homework1Dlg::OnBnClickedDisparity()
{
	// TODO:  在此加入控制項告知處理常式程式碼
	IplImage* img_r, *img_l;
	/*
	img_r = cvLoadImage("C:/opencv/opencv300/sources/samples/data/aloeR.jpg", 0);
	img_l = cvLoadImage("C:/opencv/opencv300/sources/samples/data/aloeL.jpg", 0);
	*/
	//img_r = cvLoadImage("Database/SceneR.png", 0);
	//img_l = cvLoadImage("Database/SceneL.png", 0);

	img_r = cvLoadImage("Database/SceneR_new.png", 0);
	img_l = cvLoadImage("Database/SceneL_new.png", 0);

	CvRect Rec = cvRect(0, 0, img_l->width, img_l->height);//右圖較大
	cvSetImageROI(img_r, Rec);

	cvShowImage("SceneR", img_r);
	cvShowImage("SceneL", img_l);

	CvMat* norm_disparity = cvCreateMat(img_l->height, img_l->width, CV_8U);

	//BM算法  
	CvMat* disparity = cvCreateMat(img_l->height, img_l->width, CV_32FC1);
	CvStereoBMState* BMState = cvCreateStereoBMState();
	BMState->preFilterSize = 9;
	BMState->preFilterCap = 31;
	BMState->SADWindowSize = 5;
	BMState->minDisparity = 0;
	BMState->numberOfDisparities = 48;
	BMState->textureThreshold = 10;
	BMState->uniquenessRatio = 15;
	BMState->speckleWindowSize = 100;
	BMState->speckleRange = 32;

	cvFindStereoCorrespondenceBM(img_l, img_r, disparity, BMState);
	cvNormalize(disparity, norm_disparity, 0, 256, CV_MINMAX, NULL); //正規化
	cvReleaseMat(&disparity);

	imshow("4.1 Disparity", cvarrToMat(norm_disparity));		//顯示
	cvWaitKey(0);

	cvDestroyAllWindows();
}
void CCv_homework1Dlg::OnBnClickedLeftrightcheck()
{
	// TODO:  在此加入控制項告知處理常式程式碼
	IplImage* img_r, *img_l;                //定义两个图像指针
	
	//img_r = cvLoadImage("C:/opencv/opencv300/sources/samples/data/aloeR.jpg", 0);   //图像指针初始化
	//img_l = cvLoadImage("C:/opencv/opencv300/sources/samples/data/aloeL.jpg", 0);   //图像指针初始化
	//img_r = cvLoadImage("Database/SceneR.png", 0);
	//img_l = cvLoadImage("Database/SceneL.png", 0);

	img_r = cvLoadImage("Database/SceneR_new.png", 0);
	img_l = cvLoadImage("Database/SceneL_new.png", 0);

	CvRect Rec = cvRect(0, 0, img_l->width, img_l->height);//右圖較大
	cvSetImageROI(img_r, Rec);

	CvMat* norm_disparity = cvCreateMat(img_l->height, img_l->width, CV_8U);
	CvMat* norm_disparity2 = cvCreateMat(img_l->height, img_l->width, CV_8U);
	CvMat* disparity = cvCreateMat(img_l->height, img_l->width, CV_32FC1);
	CvMat* disparity2 = cvCreateMat(img_l->height, img_l->width, CV_32FC1);

	//BM算法  
	CvStereoBMState* BMState = cvCreateStereoBMState();
	BMState->preFilterSize = 9;
	BMState->preFilterCap = 31;
	BMState->SADWindowSize = 5;
	BMState->minDisparity = 0;
	BMState->numberOfDisparities = 48;
	BMState->textureThreshold = 10;
	BMState->uniquenessRatio = 15;
	BMState->speckleWindowSize = 100;
	BMState->speckleRange = 32;
	BMState->disp12MaxDiff = -1;

	cvFindStereoCorrespondenceBM(img_l, img_r, disparity, BMState);  //校正
	cvNormalize(disparity, norm_disparity, 0, 256, CV_MINMAX, NULL); //正規化

	BMState->disp12MaxDiff = 1;
	cvFindStereoCorrespondenceBM(img_l, img_r, disparity2, BMState);  //校正
	cvNormalize(disparity2, norm_disparity2, 0, 256, CV_MINMAX, NULL); //正規化

	cvReleaseMat(&disparity);
	cvReleaseMat(&disparity2);

	Mat withoutDiff = cvarrToMat(norm_disparity);
	Mat withDiff = cvarrToMat(norm_disparity2);

	imshow("4.2 Disparity without checked", withoutDiff);		//顯示
	imshow("4.2 Disparity with checked", withDiff);		//顯示

	Mat diff = Mat(img_l->height, img_l->width, CV_8UC3);;




	for (int x = 0; x < img_l->height; x++){
		for (int y = 0; y < img_l->width; y++){
			if (withoutDiff.at<uchar>(x, y) == withDiff.at<uchar>(x, y))
			{
				diff.at<Vec3b>(x, y)[0] = withoutDiff.at<uchar>(x, y);
				diff.at<Vec3b>(x, y)[1] = withoutDiff.at<uchar>(x, y);
				diff.at<Vec3b>(x, y)[2] = withoutDiff.at<uchar>(x, y);

			}
			else{
				diff.at<Vec3b>(x, y)[0] = 0;
				diff.at<Vec3b>(x, y)[1] = 0;
				diff.at<Vec3b>(x, y)[2] = 255;
			}
		}
	}

	imshow("4.2 Difference", diff);

	waitKey(0);

	//關window
	cvDestroyAllWindows();
}
vector<KeyPoint> keypoints_1, keypoints_2;
void CCv_homework1Dlg::OnBnClickedSift()
{
	// TODO:  在此加入控制項告知處理常式程式碼
	Mat img_1 = imread("Database/plane1.jpg", CV_LOAD_IMAGE_COLOR);
	Mat img_2 = imread("Database/plane2.jpg", CV_LOAD_IMAGE_COLOR);
	
	Ptr<SIFT> f2d;

	f2d = SIFT::create(250);
	//cv::Ptr<Feature2D> f2d = xfeatures2d::SURF::create();
	//cv::Ptr<Feature2D> f2d = ORB::create();
	// you get the picture, i hope..

	
	cout << "Step 1: Detect the keypoints" << endl;
	f2d->detect(img_1, keypoints_1);
	f2d->detect(img_2, keypoints_2);


	cout << "Step 2: Calculate descriptors (feature vectors)" << endl;
	Mat descriptors_1, descriptors_2;
	f2d->compute(img_1, keypoints_1, descriptors_1);
	f2d->compute(img_2, keypoints_2, descriptors_2);

	
	cout << "Step 3: Matching descriptor vectors using BFMatcher" << endl;
	vector<vector<int>> match;
	for (int i = 0; i < descriptors_1.rows; i++){
		int j;
		int ind1 = -1;
		int ind2 = -1;
		float dis1 = -1;
		float dis2 = -1;
		for (j = 0; j < descriptors_2.rows; j++){
			//cout << i << " , " << j << endl;
			vector<float> vector_1;
			vector<float> vector_2;

			//cout << descriptors_1.at<float>(0,0) << endl;
			float disNow = 0;
			for (int k = 0; k < descriptors_1.cols; k++){
				//cout << k << endl;
				disNow += (descriptors_1.at<float>(i, k) - descriptors_2.at<float>(j, k)) * (descriptors_1.at<float>(i, k) - descriptors_2.at<float>(j, k));
			}

			disNow = sqrt(disNow);

			if (j == 0){
				ind1 = j;
				ind2 = j;
				dis1 = disNow;
				dis2 = disNow;
			}
			if (j == 1){
				if (disNow < dis2){
					ind1 = j;
					dis1 = disNow;
				}
				else {
					ind2 = j;
					dis2 = disNow;
				}
			}
			if (disNow < dis2){
				if (disNow < dis1){
					ind1 = j;
					dis1 = disNow;
				}
				else {
					ind2 = j;
					dis2 = disNow;
				}
			}
		}
		//cout << ind1 << " ind " << ind2 << endl;
		//cout << dis1 << " dis " << dis2 << endl;
		if (dis1 / dis2 < 0.5){
			//i to j
			//cout << "match" << i << "  " << ind1 << endl;
			match.push_back({ i, ind1 });
		}
	}
	
	for (int i = 0; i < keypoints_2.size(); i++){
		keypoints_2[i].pt.x += img_1.cols;
	}

	
	Mat outImg(img_1.rows, img_1.cols + img_2.cols, CV_8UC3);
	Mat imgROI = outImg(Rect(0, 0, img_1.cols, img_1.rows));
	addWeighted(imgROI, 0, img_1, 1, 0, imgROI);
	imgROI = outImg(Rect(img_1.cols, 0, img_2.cols, img_2.rows));
	addWeighted(imgROI, 0, img_2, 1, 0, imgROI);
	RNG rng(time(0));
	Scalar color;
	
	for (int i = 0; i < keypoints_1.size(); i++){
		color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		circle(outImg, keypoints_1[i].pt, 5, color, 2);
		color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		circle(outImg, keypoints_2[i].pt, 5, color, 2);
	}

	for (int i = 0; i < match.size(); i++){
		//cout << keypoints_1[match[i][0]].pt << endl;
		//cout << keypoints_2[match[i][1]].pt << endl;

		color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		line(outImg, keypoints_1[match[i][0]].pt, keypoints_2[match[i][1]].pt, color, 1);
	}

	imshow("5. SIFT", outImg);
	waitKey(0);
}
