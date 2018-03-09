
// Cv_homework1.cpp : 定義應用程式的類別行為。
//
#include "stdafx.h"
#include "Cv_homework1.h"
#include "Cv_homework1Dlg.h"
#ifdef _DEBUG
#define new DEBUG_NEW
#endif
// CCv_homework1App
BEGIN_MESSAGE_MAP(CCv_homework1App, CWinApp)
	ON_COMMAND(ID_HELP, &CWinApp::OnHelp)
END_MESSAGE_MAP()
// CCv_homework1App 建構
CCv_homework1App::CCv_homework1App()
{
	// 支援重新啟動管理員
	m_dwRestartManagerSupportFlags = AFX_RESTART_MANAGER_SUPPORT_RESTART;

	// TODO:  在此加入建構程式碼，
	// 將所有重要的初始設定加入 InitInstance 中
}
// 僅有的一個 CCv_homework1App 物件
CCv_homework1App theApp;
// CCv_homework1App 初始設定
BOOL CCv_homework1App::InitInstance()
{
	// 假如應用程式資訊清單指定使用 ComCtl32.dll 6 (含) 以後版本，
	// 來啟動視覺化樣式，在 Windows XP 上，則需要 InitCommonControls()。
	// 否則任何視窗的建立都將失敗。
	INITCOMMONCONTROLSEX InitCtrls;
	InitCtrls.dwSize = sizeof(InitCtrls);
	// 設定要包含所有您想要用於應用程式中的
	// 通用控制項類別。
	InitCtrls.dwICC = ICC_WIN95_CLASSES;
	InitCommonControlsEx(&InitCtrls);

	CWinApp::InitInstance();


	AfxEnableControlContainer();

	// 建立殼層管理員，以防對話方塊包含
	// 任何殼層樹狀檢視或殼層清單檢視控制項。
	CShellManager *pShellManager = new CShellManager;

	// 啟動 [Windows 原生] 視覺化管理員可啟用 MFC 控制項中的主題
	CMFCVisualManager::SetDefaultManager(RUNTIME_CLASS(CMFCVisualManagerWindows));

	// 標準初始設定
	// 如果您不使用這些功能並且想減少
	// 最後完成的可執行檔大小，您可以
	// 從下列程式碼移除不需要的初始化常式，
	// 變更儲存設定值的登錄機碼
	// TODO:  您應該適度修改此字串
	// (例如，公司名稱或組織名稱)
	SetRegistryKey(_T("本機 AppWizard 所產生的應用程式"));

	CCv_homework1Dlg dlg;
	m_pMainWnd = &dlg;
	INT_PTR nResponse = dlg.DoModal();
	if (nResponse == IDOK)
	{
		// TODO:  在此放置於使用 [確定] 來停止使用對話方塊時
		// 處理的程式碼
	}
	else if (nResponse == IDCANCEL)
	{
		// TODO:  在此放置於使用 [取消] 來停止使用對話方塊時
		// 處理的程式碼
	}
	else if (nResponse == -1)
	{
		TRACE(traceAppMsg, 0, "警告: 對話方塊建立失敗，因此，應用程式意外終止。\n");
		TRACE(traceAppMsg, 0, "警告: 如果您要在對話方塊上使用 MFC 控制項，則無法 #define _AFX_NO_MFC_CONTROLS_IN_DIALOGS。\n");
	}

	// 刪除上面所建立的殼層管理員。
	if (pShellManager != NULL)
	{
		delete pShellManager;
	}

	// 因為已經關閉對話方塊，傳回 FALSE，所以我們會結束應用程式，
	// 而非提示開始應用程式的訊息。
	return FALSE;
}






int inputQuad_index;
Point2f inputQuad[4];
Calibration::Calibration()
{
	boardsize = Size(11,8);
	paraGet = false;
}
IplImage* Calibration::getCorners(string filename)
{
	IplImage *image = cvLoadImage(filename.c_str());
	IplImage *gray_image = cvCreateImage(cvGetSize(image), 8, 1);
	CvPoint2D32f* corners = new CvPoint2D32f[boardsize.area()];
	int isFound;
	int corner_count;

	isFound = cvFindChessboardCorners(image, boardsize, corners, &corner_count, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
	cvCvtColor(image, gray_image, CV_BGR2GRAY);
	cvFindCornerSubPix(gray_image, corners, corner_count, cvSize(11, 11), cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

	cvDrawChessboardCorners(image, boardsize, corners, corner_count, isFound);
	return image;
}
void Calibration::computeIntrinsicParameters()
{
	if (paraGet) return;

	CvMat* object_points = cvCreateMat(21 * boardsize.area(), 3, CV_32FC1);
	CvMat* image_points = cvCreateMat(21 * boardsize.area(), 2, CV_32FC1);
	CvMat* point_counts = cvCreateMat(21, 1, CV_32SC1);
	CvPoint2D32f* corners = new CvPoint2D32f[boardsize.area()];
	CvMat* rotation_vector = cvCreateMat(3, 1, CV_32FC1);
	CvMat* jacobian = cvCreateMat(3, 1, CV_32FC1);

	int corner_count;
	int successes = 0;
	int step, frame = 0;
	int i = 1;

	while (successes < 21) {
		//Find chessboard corners:
		string filename = "Database\\" + to_string(i) + ".bmp";
		IplImage *image = cvLoadImage(filename.c_str());
		IplImage *gray_image = cvCreateImage(cvGetSize(image), 8, 1);
		int found = cvFindChessboardCorners(image, boardsize, corners, &corner_count, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS
			);
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
		i++;
	}
	CV_MAT_ELEM(*intrinsic_matrix, float, 0, 0) = 1.0f;
	CV_MAT_ELEM(*intrinsic_matrix, float, 1, 1) = 1.0f;

	cvCalibrateCamera2(
		object_points, image_points,
		point_counts, cvGetSize(cvLoadImage("Database\\1.bmp")),
		intrinsic_matrix, distortion_coeffs,
		NULL, NULL, 0 //CV_CALIB_FIX_ASPECT_RATIO
		);
	cvRodrigues2(rotation_vector, rotation_mat, jacobian = NULL);

	paraGet = true;
}
void Calibration::computeExtrinsicParameters(string filename)
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
Mat Calibration::getImageWithPyramid(string filename){

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
		line(image_Mat, pyramid_point2D[i], pyramid_point2D[(i % 4) + 1], cv::Scalar(0, 0, 255), 2);
	}

	//cout << "Rotation Vector :" << endl << rotationVector << endl;
	//cout << "Translation Vector :" << endl << translationVector << endl;

	return image_Mat;
}
void Calibration::transformation(string filename)
{
	// Output Quadilateral or World plane coordinates
	Point2f outputQuad[4];

	// Lambda Matrix
	Mat lambda(3, 3, CV_32FC1);
	//Input and Output Image;
	Mat input, output;
	Mat input2;

	//Load the image
	input = imread(filename.c_str(), 1);
	input2 = imread(filename.c_str(), 1);

	namedWindow("3. Perspective Transformation");
	inputQuad_index = 0;
	setMouseCallback("3. Perspective Transformation", onMouse);
	do
	{
		if (inputQuad_index >= 1){
			circle(input, inputQuad[inputQuad_index - 1], 5, Scalar(0, 0, 255), -1);
		}
		imshow("3. Perspective Transformation", input);
		waitKey(30);
	} while (inputQuad_index < 4);
	circle(input, inputQuad[inputQuad_index - 1], 5, Scalar(0, 0, 255), -1);
	imshow("3. Perspective Transformation", input);

	/*outputQuad[0] = Point2f(0, 0);
	outputQuad[1] = Point2f(input.cols, 0);
	outputQuad[2] = Point2f(input.cols, input.rows);
	outputQuad[3] = Point2f(0, input.rows);*/

	outputQuad[0] = Point2f(20, 20);
	outputQuad[1] = Point2f(450, 20);
	outputQuad[2] = Point2f(450, 450);
	outputQuad[3] = Point2f(20, 450);

	lambda = getPerspectiveTransform(inputQuad, outputQuad);
	warpPerspective(input2, output, lambda, input.size());

	namedWindow("3. Perspective Transformation Output", WINDOW_AUTOSIZE);
	imshow("3. Perspective Transformation Output", output);
	resizeWindow("3. Perspective Transformation Output", 470, 470);
	waitKey(0);
	cvDestroyAllWindows();
}
void onMouse(int event, int x, int y, int flags, void* param)
{
	if (event == CV_EVENT_LBUTTONDOWN){
		inputQuad[inputQuad_index++] = Point2f(x, y);
		//cout << x << " " << y << endl;
	}
}