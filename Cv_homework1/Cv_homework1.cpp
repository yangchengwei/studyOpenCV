
// Cv_homework1.cpp : �w�q���ε{�������O�欰�C
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
// CCv_homework1App �غc
CCv_homework1App::CCv_homework1App()
{
	// �䴩���s�Ұʺ޲z��
	m_dwRestartManagerSupportFlags = AFX_RESTART_MANAGER_SUPPORT_RESTART;

	// TODO:  �b���[�J�غc�{���X�A
	// �N�Ҧ����n����l�]�w�[�J InitInstance ��
}
// �Ȧ����@�� CCv_homework1App ����
CCv_homework1App theApp;
// CCv_homework1App ��l�]�w
BOOL CCv_homework1App::InitInstance()
{
	// ���p���ε{����T�M����w�ϥ� ComCtl32.dll 6 (�t) �H�᪩���A
	// �ӱҰʵ�ı�Ƽ˦��A�b Windows XP �W�A�h�ݭn InitCommonControls()�C
	// �_�h����������إ߳��N���ѡC
	INITCOMMONCONTROLSEX InitCtrls;
	InitCtrls.dwSize = sizeof(InitCtrls);
	// �]�w�n�]�t�Ҧ��z�Q�n�Ω����ε{������
	// �q�α�����O�C
	InitCtrls.dwICC = ICC_WIN95_CLASSES;
	InitCommonControlsEx(&InitCtrls);

	CWinApp::InitInstance();


	AfxEnableControlContainer();

	// �إߴ߼h�޲z���A�H����ܤ���]�t
	// ����߼h���˵��δ߼h�M���˵�����C
	CShellManager *pShellManager = new CShellManager;

	// �Ұ� [Windows ���] ��ı�ƺ޲z���i�ҥ� MFC ��������D�D
	CMFCVisualManager::SetDefaultManager(RUNTIME_CLASS(CMFCVisualManagerWindows));

	// �зǪ�l�]�w
	// �p�G�z���ϥγo�ǥ\��åB�Q���
	// �̫᧹�����i�����ɤj�p�A�z�i�H
	// �q�U�C�{���X�������ݭn����l�Ʊ`���A
	// �ܧ��x�s�]�w�Ȫ��n�����X
	// TODO:  �z���ӾA�׭ק惡�r��
	// (�Ҧp�A���q�W�٩β�´�W��)
	SetRegistryKey(_T("���� AppWizard �Ҳ��ͪ����ε{��"));

	CCv_homework1Dlg dlg;
	m_pMainWnd = &dlg;
	INT_PTR nResponse = dlg.DoModal();
	if (nResponse == IDOK)
	{
		// TODO:  �b����m��ϥ� [�T�w] �Ӱ���ϥι�ܤ����
		// �B�z���{���X
	}
	else if (nResponse == IDCANCEL)
	{
		// TODO:  �b����m��ϥ� [����] �Ӱ���ϥι�ܤ����
		// �B�z���{���X
	}
	else if (nResponse == -1)
	{
		TRACE(traceAppMsg, 0, "ĵ�i: ��ܤ���إߥ��ѡA�]���A���ε{���N�~�פ�C\n");
		TRACE(traceAppMsg, 0, "ĵ�i: �p�G�z�n�b��ܤ���W�ϥ� MFC ����A�h�L�k #define _AFX_NO_MFC_CONTROLS_IN_DIALOGS�C\n");
	}

	// �R���W���ҫإߪ��߼h�޲z���C
	if (pShellManager != NULL)
	{
		delete pShellManager;
	}

	// �]���w�g������ܤ���A�Ǧ^ FALSE�A�ҥH�ڭ̷|�������ε{���A
	// �ӫD���ܶ}�l���ε{�����T���C
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