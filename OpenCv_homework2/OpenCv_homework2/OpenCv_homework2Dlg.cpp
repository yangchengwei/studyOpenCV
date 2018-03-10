
// OpenCv_homework2Dlg.cpp : 實作檔
//

#include "stdafx.h"
#include "OpenCv_homework2.h"
#include "OpenCv_homework2Dlg.h"
#include "afxdialogex.h"
#include "opencv/cv.h"
#include "opencv2/opencv.hpp"
#include <vector>
#include <string>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

using namespace std;
using namespace cv;

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


// COpenCv_homework2Dlg 對話方塊



COpenCv_homework2Dlg::COpenCv_homework2Dlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(COpenCv_homework2Dlg::IDD, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void COpenCv_homework2Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(COpenCv_homework2Dlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON1, &COpenCv_homework2Dlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON2, &COpenCv_homework2Dlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDC_BUTTON3, &COpenCv_homework2Dlg::OnBnClickedButton3)
	ON_BN_CLICKED(IDC_BUTTON4, &COpenCv_homework2Dlg::OnBnClickedButton4)
	ON_BN_CLICKED(IDC_BUTTON5, &COpenCv_homework2Dlg::OnBnClickedButton5)
	ON_BN_CLICKED(IDC_BUTTON6, &COpenCv_homework2Dlg::OnBnClickedButton6)
	ON_BN_CLICKED(IDC_BUTTON7, &COpenCv_homework2Dlg::OnBnClickedButton7)
	ON_BN_CLICKED(IDC_BUTTON8, &COpenCv_homework2Dlg::OnBnClickedButton8)
END_MESSAGE_MAP()


// COpenCv_homework2Dlg 訊息處理常式

BOOL COpenCv_homework2Dlg::OnInitDialog()
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

	// TODO: 在此加入額外的初始設定
	AllocConsole();
	freopen("CONOUT$","w",stdout);

	return TRUE;  // 傳回 TRUE，除非您對控制項設定焦點
}

void COpenCv_homework2Dlg::OnSysCommand(UINT nID, LPARAM lParam)
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

void COpenCv_homework2Dlg::OnPaint()
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
HCURSOR COpenCv_homework2Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

// global variable
bool paraGet;
Size boardsize = Size(11,8);
CvMat* intrinsic_matrix = cvCreateMat(3, 3, CV_32FC1);
CvMat* distortion_coeffs = cvCreateMat(5, 1, CV_32FC1);
CvMat* translation_vector = cvCreateMat(3, 1, CV_32FC1);
CvMat* rotation_vector = cvCreateMat(3, 1, CV_32FC1);
CvMat* rotation_mat = cvCreateMat(3, 3, CV_32FC1);

IplImage* getCorners(string filename);
void computeIntrinsicParameters();
void computeExtrinsicParameters(string filename);
Mat getImageWithPyramid(string filename);
void transformation(string filename);

// function
void computeIntrinsicParameters()
{
	
	if (paraGet) return;

	CvMat* object_points = cvCreateMat(20 * boardsize.area(), 3, CV_32FC1);
	CvMat* image_points = cvCreateMat(20 * boardsize.area(), 2, CV_32FC1);
	CvMat* point_counts = cvCreateMat(20, 1, CV_32SC1);
	CvPoint2D32f* corners = new CvPoint2D32f[boardsize.area()];
	CvMat* rotation_vector = cvCreateMat(3, 1, CV_32FC1);
	CvMat* jacobian = cvCreateMat(3, 1, CV_32FC1);
	int corner_count;
	int successes = 0;
	int step, frame = 0;
	int i = 1;

	for (int i = 1; i <= 20; i++){
		//Find chessboard corners:
		string filename = "./dataset/CameraImg" + to_string(i) + ".png";
		IplImage *image = cvLoadImage(filename.c_str());
		IplImage *gray_image = cvCreateImage(cvGetSize(image), 8, 1);
		int found = cvFindChessboardCorners(image, boardsize, corners, &corner_count, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
		//Get Subpixel accuracy on those corners
		cvCvtColor(image, gray_image, CV_BGR2GRAY);
		cvFindCornerSubPix(gray_image, corners, corner_count,
			cvSize(11, 11), cvSize(-1, -1), cvTermCriteria(
			CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		//Draw it
		cvDrawChessboardCorners(image, boardsize, corners, corner_count, found);

		// If we got a good board, add it to our data
		if (corner_count == boardsize.area()) {
			step = successes*boardsize.area();
			for (int i = step, j = 0; j<boardsize.area(); ++i, ++j) {
				CV_MAT_ELEM(*image_points, float, i, 0) = corners[j].x;
				CV_MAT_ELEM(*image_points, float, i, 1) = corners[j].y;
				CV_MAT_ELEM(*object_points, float, i, 0) = j / boardsize.width;
				CV_MAT_ELEM(*object_points, float, i, 1) = j % boardsize.width;
				CV_MAT_ELEM(*object_points, float, i, 2) = 0.0f;
			}
			CV_MAT_ELEM(*point_counts, int, successes, 0) = boardsize.area();
			successes++;
		}
		cout << i << endl;
	}
	object_points->height = successes * boardsize.area();
	image_points->height = successes * boardsize.area();
	point_counts->height = successes * 1;

	CV_MAT_ELEM(*intrinsic_matrix, float, 0, 0) = 1.0f;
	CV_MAT_ELEM(*intrinsic_matrix, float, 1, 1) = 1.0f;

	cvCalibrateCamera2(
		object_points, image_points,
		point_counts, cvGetSize(cvLoadImage("./dataset/CameraImg1.png")),
		intrinsic_matrix, distortion_coeffs,
		NULL, NULL, 0 //CV_CALIB_FIX_ASPECT_RATIO
		);
	cvRodrigues2(rotation_vector, rotation_mat, jacobian = NULL);

	paraGet = true;
}
void computeExtrinsicParameters(string filename)
{
	if (!paraGet){
		computeIntrinsicParameters();
	}

	CvMat* object_points = cvCreateMat(boardsize.area(), 3, CV_32FC1);
	CvMat* image_points = cvCreateMat(boardsize.area(), 2, CV_32FC1);
	CvPoint2D32f* corners = new CvPoint2D32f[boardsize.area()];
	CvMat* jacobian = cvCreateMat(3, 1, CV_32FC1);

	int corner_count;

	//Find chessboard corners:
	IplImage *image = cvLoadImage(filename.c_str());
	int found = cvFindChessboardCorners(image, boardsize, corners, &corner_count, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
	for (int i = 0, j = 0; j<boardsize.area(); ++i, ++j) {
		CV_MAT_ELEM(*image_points, float, i, 0) = corners[j].x;
		CV_MAT_ELEM(*image_points, float, i, 1) = corners[j].y;
		CV_MAT_ELEM(*object_points, float, i, 0) = j / boardsize.width;
		CV_MAT_ELEM(*object_points, float, i, 1) = j % boardsize.width;
		CV_MAT_ELEM(*object_points, float, i, 2) = 0.0f;
	}
	cvFindExtrinsicCameraParams2(
		object_points, image_points,
		intrinsic_matrix, distortion_coeffs,
		rotation_vector, translation_vector);
	cvRodrigues2(rotation_vector, rotation_mat, jacobian = NULL);
}
Mat getImageWithPyramid(string filename){

	computeExtrinsicParameters(filename);

	vector<cv::Point3d> model_points;
	vector<cv::Point2d> image_points;
	CvPoint2D32f* corners = new CvPoint2D32f[boardsize.area()];
	int corner_count;
	
	//Find chessboard corners:
	IplImage *image = cvLoadImage(filename.c_str());
	int found = cvFindChessboardCorners(image, boardsize, corners, &corner_count, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
	for (int i = 0, j = 0; j<boardsize.area(); ++i, ++j) {
		image_points.push_back(Point2d(corners[j].x, corners[j].y));
		model_points.push_back(Point3d(j / boardsize.width, j % boardsize.width, 0.0f));
	}

	// Camera internals
	Mat camera_matrix = cvarrToMat(intrinsic_matrix);
	Mat dist_coeffs = cvarrToMat(distortion_coeffs);

	//cout << "Camera Matrix :" << endl << camera_matrix << endl;
	// Output rotation and translation
	Mat rotationVector = cvarrToMat(rotation_vector); // Rotation in axis-angle form
	Mat translationVector = cvarrToMat(translation_vector);

	// Solve for pose
	//cout << "Rotation Vector :" << endl << rotationVector << endl;
	//cout << "Translation Vector :" << endl << translationVector << endl;

	vector<Point3d> pyramid_point3D;
	Mat pyramid_point2D_Mat;

	pyramid_point3D.push_back(Point3d(0, 0, 2));
	pyramid_point3D.push_back(Point3d(1, 1, 0));
	pyramid_point3D.push_back(Point3d(1, -1, 0));
	pyramid_point3D.push_back(Point3d(-1, -1, 0));
	pyramid_point3D.push_back(Point3d(-1, 1, 0));


	projectPoints(Mat(pyramid_point3D), rotationVector, translationVector, camera_matrix, dist_coeffs, pyramid_point2D_Mat);

	//cout << Mat(pyramid_point3D) << endl;
	//cout << pyramid_point2D_Mat << endl;

	Mat image_Mat = cvarrToMat(image);

	//cout << "image_points[0] :" << endl << image_points[0] << endl;
	//cout << "nose_end_point2D :" << endl << pyramid_point2D_Mat << endl;

	vector<Point2d> pyramid_point2D;
	for (int i = 0; i < 5; i++){
		pyramid_point2D.push_back(Point2d(pyramid_point2D_Mat.at<double>(i, 0), pyramid_point2D_Mat.at<double>(i, 1)));
	}

	for (int i = 0; i < 4; i++){
		line(image_Mat, pyramid_point2D[0], pyramid_point2D[i + 1], cv::Scalar(0, 0, 255), 2);
		line(image_Mat, pyramid_point2D[i+1], pyramid_point2D[((i+1) % 4) + 1], cv::Scalar(0, 0, 255), 2);
	}
	
	return image_Mat;
}

// button function
void COpenCv_homework2Dlg::OnBnClickedButton1()
{
	// TODO:  在此加入控制項告知處理常式程式碼
	for (int i = 1; i <= 1/*20*/; i++){
		string filename = "./dataset/CameraImg" + to_string(i) + ".png";
		IplImage *image = cvLoadImage(filename.c_str());
		IplImage *gray_image = cvCreateImage(cvGetSize(image), 8, 1);
		CvPoint2D32f* corners = new CvPoint2D32f[boardsize.area()];
		int isFound;
		int corner_count;

		isFound = cvFindChessboardCorners(image, boardsize, corners, &corner_count, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
		cvCvtColor(image, gray_image, CV_BGR2GRAY);
		cvFindCornerSubPix(gray_image, corners, corner_count, cvSize(11, 11), cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

		cvDrawChessboardCorners(image, boardsize, corners, corner_count, isFound);
		cvShowImage("1.1", image);
		waitKey(0);
		cvDestroyAllWindows();
	}
}
void COpenCv_homework2Dlg::OnBnClickedButton2()
{
	// TODO:  在此加入控制項告知處理常式程式碼
	if (!paraGet){
		computeIntrinsicParameters();
	}

	vector<vector<float>> intrinsicMatrix(3,vector<float>(3));

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			intrinsicMatrix[i][j] = CV_MAT_ELEM(*intrinsic_matrix, float, i, j);
		}
	}
	cout << "Intrinsic Matrix :" << endl;
	cout << "[" << intrinsicMatrix[0][0] << ", " << intrinsicMatrix[0][1] << ", " << intrinsicMatrix[0][2] << ";" << endl;
	cout << " " << intrinsicMatrix[1][0] << ", " << intrinsicMatrix[1][1] << ", " << intrinsicMatrix[1][2] << ";" << endl;
	cout << " " << intrinsicMatrix[2][0] << ", " << intrinsicMatrix[2][1] << ", " << intrinsicMatrix[2][2] << "]" << endl << endl;
}
void COpenCv_homework2Dlg::OnBnClickedButton3()
{
	// TODO:  在此加入控制項告知處理常式程式碼
	string filename = "./dataset/CameraImg1.png";
	computeExtrinsicParameters(filename);

	vector<vector<float>> rotationMatrix(3,vector<float>(3));
	vector<float> translationMatrix(3);

	for (int i = 0; i < 3; i++)
	{
		translationMatrix[i] = CV_MAT_ELEM(*translation_vector, float, i, 0);
		for (int j = 0; j < 3; j++)
		{
			rotationMatrix[i][j] = CV_MAT_ELEM(*rotation_mat, float, i, j);
		}
	}
	cout << "Extrinsic Matrix :" << endl;
	cout << "[" << rotationMatrix[0][0] << ", " << rotationMatrix[0][1] << ", " << rotationMatrix[0][2] << ", " << translationMatrix[0] << ";" << endl;
	cout << " " << rotationMatrix[1][0] << ", " << rotationMatrix[1][1] << ", " << rotationMatrix[1][2] << ", " << translationMatrix[1] << ";" << endl;
	cout << " " << rotationMatrix[2][0] << ", " << rotationMatrix[2][1] << ", " << rotationMatrix[2][2] << ", " << translationMatrix[2] << "]" << endl << endl;
}
void COpenCv_homework2Dlg::OnBnClickedButton4()
{
	// TODO:  在此加入控制項告知處理常式程式碼
	if (!paraGet){
		computeIntrinsicParameters();
	}
	vector<float> distortionMatrix(5);

	for (int i = 0; i < 5; i++)
	{
		distortionMatrix[i] = CV_MAT_ELEM(*distortion_coeffs, float, i, 0);
	}

	cout << "Distortion Matrix :" << endl;
	cout << "[" << distortionMatrix[0] << ", " << distortionMatrix[1] << ", " << distortionMatrix[2] << ", " << distortionMatrix[3] << ", " << distortionMatrix[4] << "]" << endl << endl;
}
void COpenCv_homework2Dlg::OnBnClickedButton5()
{
	for (int i = 1; i <= 5; i++){
		string filename = "./dataset/CameraImg"+to_string(i)+".png";
		Mat output = getImageWithPyramid(filename);
		imshow("2.", output);
		waitKey(500);
	}
	waitKey(0);
	cvDestroyAllWindows();
}
void COpenCv_homework2Dlg::OnBnClickedButton6()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	int frameNumber = 454;
	string videoFilename = "./dataset/bgSub_video.mp4";
	VideoCapture capture(videoFilename);
	vector<Mat> frame(frameNumber); //current frame
	vector<Mat> fgMaskMOG2(frameNumber); //fg mask fg mask generated by MOG2 method
	BackgroundSubtractorMOG pMOG2; //MOG2 Background subtractor
	char keyboard = 0;
	int now = 0;
	for (int now = 0; now<frameNumber; now++){
		capture.read(frame[now]);
		//update the background model
		pMOG2.operator()(frame[now], fgMaskMOG2[now]);
		cout<<now<<endl;
	}
	for (int i = 0; i<frameNumber; i++)
	{
		//show the current frame and the fg masks
		imshow("Video", frame[i]);
		imshow("1.1 output", fgMaskMOG2[i]);
		//get the input from the keyboard
		keyboard = (char)waitKey( 10 );
	}
	//delete capture object
	capture.release();
	cvDestroyAllWindows();
}
void COpenCv_homework2Dlg::OnBnClickedButton7()
{
	// TODO:  在此加入控制項告知處理常式程式碼
	IplImage* img_r, *img_l;

	img_r = cvLoadImage("./dataset/SceneR.png", 0);
	img_l = cvLoadImage("./dataset/SceneL.png", 0);

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

	imshow("4.1", cvarrToMat(norm_disparity));		//顯示
	cvWaitKey(0);

	cvDestroyAllWindows();
}
void COpenCv_homework2Dlg::OnBnClickedButton8()
{
	// TODO:  在此加入控制項告知處理常式程式碼
	IplImage* img_r, *img_l;

	img_r = cvLoadImage("./dataset/SceneR.png", 0);
	img_l = cvLoadImage("./dataset/SceneL.png", 0);

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

	imshow("4.2", diff);

	waitKey(0);

	cvDestroyAllWindows();
}