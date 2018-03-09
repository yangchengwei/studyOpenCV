
// OpenCv_homework1Dlg.cpp : 實作檔
//

#include "stdafx.h"
#include "OpenCv_homework1.h"
#include "OpenCv_homework1Dlg.h"
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


// COpenCv_homework1Dlg 對話方塊



COpenCv_homework1Dlg::COpenCv_homework1Dlg(CWnd* pParent /*=NULL*/)
: CDialogEx(COpenCv_homework1Dlg::IDD, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void COpenCv_homework1Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(COpenCv_homework1Dlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDOK, &COpenCv_homework1Dlg::OnBnClickedOk)
	ON_BN_CLICKED(IDC_BUTTON1, &COpenCv_homework1Dlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON2, &COpenCv_homework1Dlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDC_BUTTON3, &COpenCv_homework1Dlg::OnBnClickedButton3)
	ON_BN_CLICKED(IDC_BUTTON4, &COpenCv_homework1Dlg::OnBnClickedButton4)
	ON_BN_CLICKED(IDC_BUTTON5, &COpenCv_homework1Dlg::OnBnClickedButton5)
	ON_BN_CLICKED(IDC_BUTTON6, &COpenCv_homework1Dlg::OnBnClickedButton6)
	ON_BN_CLICKED(IDC_BUTTON7, &COpenCv_homework1Dlg::OnBnClickedButton7)
	ON_BN_CLICKED(IDC_BUTTON8, &COpenCv_homework1Dlg::OnBnClickedButton8)
	ON_BN_CLICKED(IDC_BUTTON9, &COpenCv_homework1Dlg::OnBnClickedButton9)
	ON_BN_CLICKED(IDC_BUTTON10, &COpenCv_homework1Dlg::OnBnClickedButton10)
END_MESSAGE_MAP()


// COpenCv_homework1Dlg 訊息處理常式

BOOL COpenCv_homework1Dlg::OnInitDialog()
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

void COpenCv_homework1Dlg::OnSysCommand(UINT nID, LPARAM lParam)
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

void COpenCv_homework1Dlg::OnPaint()
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
HCURSOR COpenCv_homework1Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void COpenCv_homework1Dlg::OnBnClickedOk()
{
	// TODO:  在此加入控制項告知處理常式程式碼
	CDialogEx::OnOK();
}


void COpenCv_homework1Dlg::OnBnClickedButton1()
{
	// TODO:  在此加入控制項告知處理常式程式碼
	Mat image = imread("dog.bmp");
	imshow("1.1", image);
	waitKey(0);
	cvDestroyAllWindows();
}
void COpenCv_homework1Dlg::OnBnClickedButton2()
{
	// TODO:  在此加入控制項告知處理常式程式碼
	Mat input = imread("color.png");
	Mat output;
	output.create(input.rows, input.cols, CV_8UC3);

	for (int x = 0; x < input.rows; x++){
		for (int y = 0; y < input.cols; y++){
			output.at<Vec3b>(x, y)[0] = input.at<Vec3b>(x, y)[1];
			output.at<Vec3b>(x, y)[1] = input.at<Vec3b>(x, y)[2];
			output.at<Vec3b>(x, y)[2] = input.at<Vec3b>(x, y)[0];
		}
	}

	imshow("1.2 input", input);
	imshow("1.2 output", output);
	waitKey(0);
	cvDestroyAllWindows();
}
void COpenCv_homework1Dlg::OnBnClickedButton3()
{
	// TODO:  在此加入控制項告知處理常式程式碼
	Mat input = imread("dog.bmp");
	Mat output;
	output.create(input.rows, input.cols, CV_8UC3);

	flip(input, output, 1);

	imshow("1.3 input", input);
	imshow("1.3 output", output);
	waitKey(0);
	cvDestroyAllWindows();
}
int sliderValue;
Mat src1, src2;
void COpenCv_homework1Dlg::OnBnClickedButton4()
{
	// TODO:  在此加入控制項告知處理常式程式碼
	src1 = imread("dog.bmp");
	flip(src1, src2, 1);
	sliderValue = 0;
	int sliderMaxValue = 100;

	namedWindow("1.4", 0);
	createTrackbar("Blend", "1.4", &sliderValue, sliderMaxValue, on_trackbar);
	on_trackbar(sliderValue, 0);

	waitKey(0);
	cvDestroyAllWindows();
}
void on_trackbar(int, void*){
	double alpha = (double)sliderValue / 100.0;
	double beta = (1.0 - alpha);
	Mat dst;

	addWeighted(src2, alpha, src1, beta, 0.0, dst);
	imshow("1.4", dst);
}
void COpenCv_homework1Dlg::OnBnClickedButton5()
{
	// TODO:  在此加入控制項告知處理常式程式碼
	/*Mat src = imread("eye.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	GaussianBlur(src, src, Size(5, 5), 0, 0);
	Mat dst;
	Canny(src, dst, 50, 150, 3);

	imshow("2.1 Grayscale", src);
	imshow("2.1 Canny", dst);
	waitKey(0);*/
	IplImage* image = cvLoadImage("eye.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	IplImage* dst = cvCreateImage(cvSize(image->width, image->height), 8, 1);
	cvSmooth(image, dst, CV_GAUSSIAN, 5, 5);
	cvCanny(dst, dst, 50, 150, 3);
	namedWindow("2.1 Grayscale");
	namedWindow("2.1 Canny");
	cvShowImage("2.1 Grayscale", image);
	cvShowImage("2.1 Canny", dst);
	cvWaitKey(0);
	cvDestroyAllWindows();
}
void COpenCv_homework1Dlg::OnBnClickedButton6()
{
	// TODO:  在此加入控制項告知處理常式程式碼
	/*Mat src = imread("eye.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	GaussianBlur(src, src, Size(3, 3), 0, 0);
	Mat dst;
	Canny(src, dst, 50, 150, 3);

	vector<Vec3f> circles;

	cout << "111" << endl;
	HoughCircles(dst, circles, HOUGH_GRADIENT, 1, dst.rows / 8, 200, 100, 0, 0);
	cout << "222" << endl;

	for (int i = 0; i<circles.size(); i++){
	Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
	cout << center << endl;
	int radius = cvRound(circles[i][2]);
	circle(dst, center, radius, Scalar(255, 0, 0), 3, 8, 0);
	}

	imshow("2.1 Grayscale", src);
	imshow("2.1 Canny", dst);
	waitKey(0);
	cvDestroyAllWindows();*/
	IplImage* image = cvLoadImage("eye.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	CvMemStorage* storage = cvCreateMemStorage(0);
	cvSmooth(image, image, CV_GAUSSIAN, 5, 5);
	CvSeq* results = cvHoughCircles(
		image,						//image
		storage,					//storage
		CV_HOUGH_GRADIENT, 2, 50, 200, 100);
	cvCanny(image, image, 50, 150, 3);
	IplImage* output = cvCreateImage(cvSize(image->width, image->height), 8, 3);
	cvMerge(image, image, image, NULL, output);
	for (int i = 0; i < results->total; i++)
	{
		float* p = (float*)cvGetSeqElem(results, i);
		CvPoint pt = cvPoint(cvRound(p[0]), cvRound(p[1]));
		cvCircle(output, pt, cvRound(p[2]), CV_RGB(255, 0, 0), 2, CV_AA);
	}

	namedWindow("2.2 Canny", 1);
	cvShowImage("2.2 Canny", output);
	cvWaitKey(0);
	cvDestroyAllWindows();
}
void COpenCv_homework1Dlg::OnBnClickedButton7()
{
	// TODO:  在此加入控制項告知處理常式程式碼
	Mat input = imread("chessboard.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	int srcHeight = input.rows;
	int srcWidth = input.cols;
	vector<vector<int>> Gx(input.rows, vector<int>(input.cols, 0));
	vector<vector<int>> Gy(input.rows, vector<int>(input.cols, 0));
	vector<vector<int>> Gxy(input.rows, vector<int>(input.cols, 0));
	vector<vector<int>> sobelFilterX{ { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
	vector<vector<int>> sobelFilterY{ { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };

	for (int y = 0; y < srcHeight; y++)
	{
		for (int x = 0; x < srcWidth; x++)
		{
			if (x != 0 && x != srcWidth - 1 && y != 0 && y != srcHeight - 1)
			{
				for (int i = 0; i <= 2; i++)
				{
					for (int j = 0; j <= 2; j++)
					{
						Gx[y][x] += ((int)input.at<uchar>(y - 1 + i, x - 1 + j)) * sobelFilterX[i][j];
						Gy[y][x] += ((int)input.at<uchar>(y - 1 + i, x - 1 + j)) * sobelFilterY[i][j];
					}
				}
			}
			else
			{
				Gx[y][x] = 0;
				Gy[y][x] = 0;
			}
			Gx[y][x] = abs(Gx[y][x]) > 255 ? 255 : abs(Gx[y][x]);
			Gy[y][x] = abs(Gy[y][x]) > 255 ? 255 : abs(Gy[y][x]);
			Gxy[y][x] = sqrt(Gx[y][x] * Gx[y][x] + Gy[y][x] * Gy[y][x]) > 255 ? 255 : (int)sqrt(Gx[y][x] * Gx[y][x] + Gy[y][x] * Gy[y][x]);
		}
	}
	Mat vertical(Size(srcWidth, srcHeight), CV_8UC1);
	Mat horizontal(Size(srcWidth, srcHeight), CV_8UC1);
	Mat result(Size(srcWidth, srcHeight), CV_8UC1);
	for (int y = 0; y < srcHeight; y++)
	{
		for (int x = 0; x < srcWidth; x++)
		{
			vertical.at<uchar>(y, x) = (unsigned char)Gx[y][x];
			horizontal.at<uchar>(y, x) = (unsigned char)Gy[y][x];
			result.at<uchar>(y, x) = (unsigned char)Gxy[y][x];
		}
	}
	imshow("2.3 Vertical", vertical);
	imshow("2.3 Horizontal", horizontal);
	imshow("2.3 Combination", result);
	waitKey(0);
	cvDestroyAllWindows();
}
int inputQuad_index;
Point2f inputQuad[4];
void COpenCv_homework1Dlg::OnBnClickedButton8()
{
	// TODO:  在此加入控制項告知處理常式程式碼
	// Output Quadilateral or World plane coordinates
	Point2f outputQuad[4];

	// Lambda Matrix
	Mat lambda(3, 3, CV_32FC1);
	//Input and Output Image;
	Mat input;
	Mat input2;
	Mat output(470, 470, CV_8U);

	//Load the image
	input = imread("QrCode.jpg", 1);
	input2 = imread("QrCode.jpg", 1);
	resize(input2, input2, Size(470, 470));

	namedWindow("3. Input");
	inputQuad_index = 0;
	setMouseCallback("3. Input", onMouse);
	do
	{
		if (inputQuad_index >= 1){
			circle(input, inputQuad[inputQuad_index - 1], 5, Scalar(0, 0, 255), -1);
		}
		imshow("3. Input", input);
		waitKey(30);
	} while (inputQuad_index < 4);
	circle(input, inputQuad[inputQuad_index - 1], 5, Scalar(0, 0, 255), -1);
	imshow("3. Input", input);

	/*outputQuad[0] = Point2f(0, 0);
	outputQuad[1] = Point2f(input.cols, 0);
	outputQuad[2] = Point2f(input.cols, input.rows);
	outputQuad[3] = Point2f(0, input.rows);*/

	inputQuad[0] = Point2f(inputQuad[0].x * 470 / input.cols, inputQuad[0].y * 470 / input.rows);
	inputQuad[1] = Point2f(inputQuad[1].x * 470 / input.cols, inputQuad[1].y * 470 / input.rows);
	inputQuad[2] = Point2f(inputQuad[2].x * 470 / input.cols, inputQuad[2].y * 470 / input.rows);
	inputQuad[3] = Point2f(inputQuad[3].x * 470 / input.cols, inputQuad[3].y * 470 / input.rows);

	outputQuad[0] = Point2f(20, 20);
	outputQuad[1] = Point2f(450, 20);
	outputQuad[2] = Point2f(450, 450);
	outputQuad[3] = Point2f(20, 450);

	lambda = getPerspectiveTransform(inputQuad, outputQuad);
	resize(input2, input2, Size(input2.cols, 470));
	warpPerspective(input2, output, lambda, input2.size());

	namedWindow("3. Output");

	imshow("3. Output", output);
	resizeWindow("3. Output", 470, 470);

	waitKey(0);
	cvDestroyAllWindows();
}
void onMouse(int event, int x, int y, int flags, void* param)
{
	if (event == CV_EVENT_LBUTTONDOWN){
		inputQuad[inputQuad_index++] = Point2f((float)x, (float)y);
		//cout << x << " " << y << endl;
	}
}
void COpenCv_homework1Dlg::OnBnClickedButton9()
{
	// TODO:  在此加入控制項告知處理常式程式碼
	Mat src = imread("shoes.jpg", CV_LOAD_IMAGE_COLOR);
	Mat gray;
	cvtColor(src, gray, CV_RGB2GRAY);
	Mat dst;
	threshold(gray, dst, 127, 255, THRESH_BINARY);
	imshow("4.1 Input", src);
	imshow("4.1 Global Threshold", dst);
	waitKey(0);
	destroyAllWindows();
}
void COpenCv_homework1Dlg::OnBnClickedButton10()
{
	// TODO:  在此加入控制項告知處理常式程式碼
	Mat src = imread("shoes.jpg", CV_LOAD_IMAGE_COLOR);
	Mat gray;
	Mat dst2;
	Mat dst3;
	Mat dst4;

	cvtColor(src, gray, CV_RGB2GRAY);
	adaptiveThreshold(gray, dst2, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 101, -50);
	GaussianBlur(dst2, dst3, Size(5, 5), 0, 0);
	medianBlur(dst3, dst4, 5);

	imshow("4.2 Input", src);
	imshow("4.2 Local Threshold", dst2);
	imshow("4.2 Gaussian Smooth", dst3);
	imshow("4.2 Median Filter", dst4);
	waitKey(0);
	destroyAllWindows();
}
