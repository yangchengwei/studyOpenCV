
// Cv_homework2Dlg.cpp : 實作檔
//

#include "stdafx.h"
#include "Cv_homework2.h"
#include "Cv_homework2Dlg.h"
#include "afxdialogex.h"
#include "opencv/cv.h"
#include "opencv2/opencv.hpp"
#include "InputBox.h"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

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


// CCv_homework2Dlg 對話方塊



CCv_homework2Dlg::CCv_homework2Dlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CCv_homework2Dlg::IDD, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CCv_homework2Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CCv_homework2Dlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON6, &CCv_homework2Dlg::OnBnClickedButton6)
	ON_BN_CLICKED(IDC_BUTTON1, &CCv_homework2Dlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON2, &CCv_homework2Dlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDC_BUTTON3, &CCv_homework2Dlg::OnBnClickedButton3)
	ON_BN_CLICKED(IDC_BUTTON4, &CCv_homework2Dlg::OnBnClickedButton4)
	ON_BN_CLICKED(IDC_BUTTON5, &CCv_homework2Dlg::OnBnClickedButton5)
	ON_BN_CLICKED(IDC_BUTTON7, &CCv_homework2Dlg::OnBnClickedButton7)
END_MESSAGE_MAP()


// CCv_homework2Dlg 訊息處理常式

BOOL CCv_homework2Dlg::OnInitDialog()
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

void CCv_homework2Dlg::OnSysCommand(UINT nID, LPARAM lParam)
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

void CCv_homework2Dlg::OnPaint()
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
HCURSOR CCv_homework2Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}


// global variable
int initPoint;
int modelExist = 0;
Ptr<FaceRecognizer> model;
vector<vector<Point2i>> facesRect(0,vector<Point2i>(2));


// some function
void onMouse(int event, int x, int y, int flags, void* param){
	if(event==CV_EVENT_LBUTTONDOWN){
		initPoint = initPoint+1;
		cout<<"x="<<x<<", y="<<y<<endl;
		ofstream myfile;
		myfile.open ("hw2_1.txt",ios::app);
		myfile << "Point" << initPoint << ":(" << x << "," << y << ")\n";
		myfile.close();
	}
}
void drawAxis(Mat& img, Point p, Point q, Scalar colour, const float scale = 0.2)
{
    double angle;
    double hypotenuse;
    angle = atan2( (double) p.y - q.y, (double) p.x - q.x ); // angle in radians
    hypotenuse = sqrt( (double) (p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));
//    double degrees = angle * 180 / CV_PI; // convert radians to degrees (0-180 range)
//    cout << "Degrees: " << abs(degrees - 180) << endl; // angle in 0-360 degrees range
    // Here we lengthen the arrow by a factor of scale
    q.x = (int) (p.x - scale * hypotenuse * cos(angle));
    q.y = (int) (p.y - scale * hypotenuse * sin(angle));
    line(img, p, q, colour, 1, CV_AA);
    // create the arrow hooks
    p.x = (int) (q.x + 9 * cos(angle + CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle + CV_PI / 4));
    line(img, p, q, colour, 1, CV_AA);
    p.x = (int) (q.x + 9 * cos(angle - CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle - CV_PI / 4));
    line(img, p, q, colour, 1, CV_AA);
}


// button function
void CCv_homework2Dlg::OnBnClickedButton1()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	int frameNumber = 454;
	string videoFilename = "./Database/bgSub_video.mp4";
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
	destroyAllWindows();
}
void CCv_homework2Dlg::OnBnClickedButton2()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	string videoFilename = "./Database/tracking_video.mp4";
	VideoCapture capture(videoFilename);
	Mat frame; //current frame
	char keyboard = 0;
	int windowSize = 0;

	capture.read(frame);
	imshow("Frame1", frame);
	setMouseCallback("Frame1", onMouse);

	initPoint = 0;
	ofstream myfile;
	myfile.open ("hw2_1.txt",ios::trunc);
	myfile.close();

	while (initPoint < 7){
		keyboard = (char)waitKey( 10 );
	}

	CInputBox inputbox;
	inputbox.user_input="10";
	if(IDOK==inputbox.DoModal())
	{
		CString input;
		input=inputbox.user_input;
		MessageBox(CString("Window size is ")+input+CString(" ."));
		ofstream myfile;
		myfile.open ("hw2_1.txt",ios::app);
		myfile << CString("Window size:")+input+"\n";
		myfile.close();
	}
	// delete capture object
	capture.release();

	// start read line
	string newLine;

	ifstream myfileIn;
	myfileIn.open ("hw2_1.txt",ios::in);

	vector<Point2d> centers(7);

	for (int i = 0; i < 7 ;i++)
	{
		myfileIn >> newLine;
		size_t found1 = newLine.find('(');
		size_t found2 = newLine.find(',');
		size_t found3 = newLine.find(')');

		centers[i] = Point2d(stoi(newLine.substr(found1+1,found2-found1-1),nullptr,10),
			stoi(newLine.substr(found2+1,found3-found2-1),nullptr,10));
	}
	//cout<< (centers);
	myfileIn >> newLine;//Window 
	myfileIn >> newLine;//Size:(value)
	size_t found1 = newLine.find(':');
	size_t found2 = newLine.length();

	int windowSizeIn = stoi(newLine.substr(found1+1,found2-found1-1),nullptr,10);

	//***** draw these points
	Mat output(frame);

	for (int i = 0; i < 7 ;i++)
	{
		rectangle(output, centers[i]-Point2d(windowSizeIn/2,windowSizeIn/2), centers[i]+Point2d(windowSizeIn/2,windowSizeIn/2), Scalar(0,0,255),-1);
	}

	imshow("Initial", output);

	waitKey(0);
	destroyAllWindows();
}
void CCv_homework2Dlg::OnBnClickedButton3()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	//***** read 2_1.txt
	ifstream myfileIn;
	myfileIn.open ("hw2_1.txt",ios::in);
	string newLine;

	vector<vector<Point2f>> centers(1,vector<Point2f>(7));
	int windowSizeIn;

	for (int i = 0; i < 7 ;i++)
	{
		myfileIn >> newLine;
		size_t found1 = newLine.find('(');
		size_t found2 = newLine.find(',');
		size_t found3 = newLine.find(')');

		centers[0][i] = Point2f(float(stoi(newLine.substr(found1+1,found2-found1-1),nullptr,10)),
								float(stoi(newLine.substr(found2+1,found3-found2-1),nullptr,10)));
	}
	//cout<< (centers);
	myfileIn >> newLine;//Window 
	myfileIn >> newLine;//Size:(value)
	size_t found1 = newLine.find(':');
	size_t found2 = newLine.length();

	windowSizeIn = stoi(newLine.substr(found1+1,found2-found1-1),nullptr,10);
	myfileIn.close();

	//***** tracking 
	string videoFilename = "./Database/tracking_video.mp4";
	VideoCapture capture(videoFilename);
	vector<uchar> status;
	vector<float> err;
	Mat frame1; //front frame
	Mat frame2; //behind frame
	capture.read(frame1);
	int frameNumber = 0;

	while( capture.read(frame2))
	{
		centers.push_back(vector<Point2f>(7));
		
		calcOpticalFlowPyrLK(frame1, frame2, centers[frameNumber], centers[frameNumber+1], status, err, Size(windowSizeIn,windowSizeIn));
		
		frame2.copyTo(frame1);
		frameNumber += 1;
		cout << frameNumber << endl;
	}
	capture.release();
	
	//***** record data
	ofstream myfile;
	myfile.open ("hw2_2.txt",ios::trunc); // delete old data
	myfile.close();
	myfile.open ("hw2_2.txt",ios::app);
	for (int i = 0 ; i < frameNumber ; i++)
	{
		myfile << "frame " << i+1 << ":";
		for (int j = 0 ; j < 7 ; j++ )
		{
			myfile << "(" << int(centers[i][j].x) << "," << int(centers[i][j].y) << ")" << ", ";
		}
		myfile << endl;
	}
	myfile.close();

	//***** draw these points
	myfileIn.open ("hw2_2.txt",ios::in);
	
	vector<vector<Point2d>> centersIn(479,vector<Point2d>(7));

	for (int j = 0; j < 479; j++ )
	{
		myfileIn >> newLine; // frame
		for (int i = 0; i < 7 ;i++)
		{
			myfileIn >> newLine;
			size_t found1 = newLine.find('(');
			size_t found2 = newLine.find(',');
			size_t found3 = newLine.find(')');

			centersIn[j][i] = Point2d(stoi(newLine.substr(found1+1,found2-found1-1),nullptr,10),
									  stoi(newLine.substr(found2+1,found3-found2-1),nullptr,10));	
		}
	}

	vector<Mat> output(479);
	capture.open(videoFilename);
	frameNumber = 0;
	
	while (frameNumber<479) 
	{
		capture.read(output[frameNumber++]);
	}
	
	for (int j=0; j<479; j++)
	{
		for (int i=0;i<7;i++)
		{
			rectangle(output[j], centersIn[j][i]-Point2d(windowSizeIn/2,windowSizeIn/2), centersIn[j][i]+Point2d(windowSizeIn/2,windowSizeIn/2), Scalar(0,0,255),-1);
			for (int k=0; k<j; k++)
			{
				line(output[j], centersIn[k][i], centersIn[k+1][i], Scalar(0,255,255));
			}
		}
		cout<<j<<endl;
	}

	for (int i = 0; i<479; i++)
	{
		imshow("Optical Flow", output[i]);
		waitKey(10);
	}
	waitKey(0);
	destroyAllWindows();
}
void CCv_homework2Dlg::OnBnClickedButton4()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	Mat src = imread("./Database/pca.jpg", CV_LOAD_IMAGE_COLOR);
	Mat gray;
	Mat bw;
	cvtColor(src, gray, CV_RGB2GRAY);
	threshold(gray, bw, 127, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

    vector<Point2d> points(0);

	for (int x = 0; x<bw.cols; x++)
	{
		for (int y = 0; y<bw.rows; y++)
		{
			if (bw.at<uchar>(y, x) != 0)
			{
				points.push_back(Point2d(x, y));
			}
		}
	}
	
	int sz = int(points.size());
	Mat data_pts = Mat(sz, 2, CV_64F);
    for (int i = 0; i < data_pts.rows; ++i)
    {
        data_pts.at<double>(i, 0) = points[i].x;
        data_pts.at<double>(i, 1) = points[i].y;
    }
    //Perform PCA analysis
    PCA pca_analysis(data_pts, Mat(), CV_PCA_DATA_AS_ROW);
    //Store the center of the object
    Point cntr = Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
                      static_cast<int>(pca_analysis.mean.at<double>(0, 1)));
    //Store the eigenvalues and eigenvectors
    vector<Point2d> eigen_vecs(2);
    vector<double> eigen_val(2);
    for (int i = 0; i < 2; ++i)
    {
        eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
                                pca_analysis.eigenvectors.at<double>(i, 1));
        eigen_val[i] = pca_analysis.eigenvalues.at<double>(i);

    }
    // Draw the principal components
	circle(src, cntr, 3, Scalar(255, 0, 255), 2);
	Point p1 = cntr + 0.02 * Point(int(eigen_vecs[0].x * eigen_val[0]), int(eigen_vecs[0].y * eigen_val[0]));
    Point p2 = cntr - 0.02 * Point(int(eigen_vecs[1].x * eigen_val[1]), int(eigen_vecs[1].y * eigen_val[1]));
    drawAxis(src, cntr, p1, Scalar(0, 255, 0), 1);
    drawAxis(src, cntr, p2, Scalar(255, 255, 0), 5);
    imshow("output", src);
	waitKey(0);
	destroyAllWindows();
}
void CCv_homework2Dlg::OnBnClickedButton5()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	vector<Mat> images;
    vector<int> labels;
	images.push_back( imread("./Database/0.jpg", CV_LOAD_IMAGE_GRAYSCALE) );
	labels.push_back( 0 );
	images.push_back( imread("./Database/1.jpg", CV_LOAD_IMAGE_GRAYSCALE) );
	labels.push_back( 1 );
	images.push_back( imread("./Database/2.jpg", CV_LOAD_IMAGE_GRAYSCALE) );
	labels.push_back( 2 );

	model = createEigenFaceRecognizer();
    model->train(images, labels);

	Mat mean = model->getMat("mean");
	normalize(mean, mean, 0, 255, NORM_MINMAX, CV_8UC1);
	mean = mean.reshape(1, images[0].rows);
	imshow("mean", mean);
	waitKey(0);
	destroyAllWindows();

	int predictedLabel = model->predict(imread("./Database/test.jpg", CV_LOAD_IMAGE_GRAYSCALE) );

	cout << "The image is : ";
	switch (predictedLabel)
	{
	case 0:
		cout << "Harry Potter." << endl;
		break;
	case 1:
		cout << "Hermione Granger." << endl;
		break;
	case 2:
		cout << "Ron Weasley." << endl;
		break;
	default:
		cout << "WTF is this?!" << endl;
		break;
	}
	modelExist = 1;
}
void CCv_homework2Dlg::OnBnClickedButton6()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	facesRect = vector<vector<Point2i>>(0,vector<Point2i>(2));
	string image_name="./Database/face.jpg";
    // Load image
    IplImage* image_detect=cvLoadImage(image_name.c_str(), 1);
    string cascade_name="./haarcascade_frontalface_default.xml";
    // Load cascade
    CvHaarClassifierCascade* classifier=(CvHaarClassifierCascade*)cvLoad(cascade_name.c_str(), 0, 0, 0);
    if(!classifier){
        cout << "ERROR: Could not load classifier cascade." << endl;
        return;
    }
    CvMemStorage* facesMemStorage=cvCreateMemStorage(0);
    IplImage* tempFrame=cvCreateImage(cvSize(image_detect->width, image_detect->height), IPL_DEPTH_8U, image_detect->nChannels);
    if(image_detect->origin==IPL_ORIGIN_TL){
        cvCopy(image_detect, tempFrame, 0);    }
    else{
        cvFlip(image_detect, tempFrame, 0);    }
    cvClearMemStorage(facesMemStorage);
	CvSeq* faces=cvHaarDetectObjects(tempFrame, classifier, facesMemStorage, 1.1, 3,
		CV_HAAR_DO_CANNY_PRUNING, cvSize(50, 50));
    if(faces){
		cout << faces->total << " faces detected." << endl;
        for(int i=0; i<faces->total; ++i){
            // Setup two points that define the extremes of the rectangle,
            // then draw it to the image
            CvPoint point1, point2;
            CvRect* rectangle = (CvRect*)cvGetSeqElem(faces, i);
            point1.x = rectangle->x;
            point2.x = rectangle->x + rectangle->width;
            point1.y = rectangle->y;
            point2.y = rectangle->y + rectangle->height;
            cvRectangle(tempFrame, point1, point2, CV_RGB(255,0,0), 3, 8, 0);

			facesRect.push_back(vector<Point2i>(2));
			facesRect[i][0] = Point2i(point1.x, point1.y);
			facesRect[i][1] = Point2i(rectangle->width, rectangle->height);
			//cout<<facesRect[i]<<endl;
        }
    }
    // Show the result in the window
    cvNamedWindow("Face Detection Result", 1);
    cvShowImage("Face Detection Result", tempFrame);
    cvWaitKey(0);
    cvDestroyWindow("Face Detection Result");
    // Clean up allocated OpenCV objects
    cvReleaseMemStorage(&facesMemStorage);
    cvReleaseImage(&tempFrame);
    cvReleaseHaarClassifierCascade(&classifier);
    cvReleaseImage(&image_detect);

}
void CCv_homework2Dlg::OnBnClickedButton7()
{
	if (!modelExist)
	{
		return;
	}
	// TODO: 在此加入控制項告知處理常式程式碼
	Mat image = imread("./Database/face.jpg");
	Mat face;

	for (int i = 0; i<facesRect.size(); i++)
	{
		int faceX = facesRect[i][0].x;
		int faceY = facesRect[i][0].y;
		int faceW = facesRect[i][1].x;
		int faceH = facesRect[i][1].y;

		face = image(Rect(faceX, faceY, faceW, faceH));

		Mat faceResize;
		faceResize.cols = 100;
		faceResize.rows = 120;
		resize(face, faceResize, Size(100,120));
		Mat faceGray;
		cvtColor(faceResize, faceGray, COLOR_BGR2GRAY);
		
		int pd = model->predict(faceGray);
		
		rectangle(image, Point(faceX, faceY), Point(faceX+faceW, faceY+faceH), Scalar(0,0,255), 3, 8, 0);
		switch (pd)
		{
		case 0:
			putText(image, "Harry Potter", Point(faceX, faceY), 0, 1, Scalar(0,0,255), 2);
			break;
		case 1:
			putText(image, "Hermione Granger", Point(faceX, faceY), 0, 1, Scalar(0,0,255), 2);
			break;
		case 2:
			putText(image, "Ron Weasley", Point(faceX, faceY), 0, 1, Scalar(0,0,255), 2);
			break;
		default:
			break;
		}
	}
	imshow("image",image);
	waitKey(0);
	destroyAllWindows();
}