#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;

Ptr<SVM> train_hog_svm(const HOGDescriptor& hog);
void on_mouse(int event, int x, int y, int flags, void* userdata);

int main()
{
	// HOGDescriptor 클래스 구현상의 문제로 인해 디버그 모드로 실행하면, 에러발생
# if _DEBUG
	cout << "svmdigit.exe should be built as Release mode!" << endl;
	return 0;
#endif

	// 필기체 숫자 영상에서 HOG 벡터를 추출하기 위해 HOGDescriptor 클래스 사용
	// 20x20 영상, 5x5 셀, 10x10 블록, 각 셀마다 9개의 그라디언트 방향 히스토그램
	// HOGDescriptor 객체 hog 생성
	HOGDescriptor hog(Size(20, 20), Size(10, 10), Size(5, 5), Size(5, 5), 9);

	// SVM 학습 진행
	Ptr<SVM> svm = train_hog_svm(hog);

	// SVM 학습 실패시 프로그램 종료
	if (svm.empty()) {
		cerr << "훈련 없음" << endl;
		return -1;
	}

	Mat img = Mat::zeros(400, 400, CV_8U);
	imshow("img", img);

	// 마우스 이벤트 연동
	setMouseCallback("img", on_mouse, (void*)&img);

	while (true) {
		// 키보드 입력 대기
		int c = waitKey();

		// ESC 키
		if (c == 27) {
			break;
		}

		// Space 키
		else if (c == ' ') {
			Mat img_resize;

			// img 를 20x20 크기로 변환
			resize(img, img_resize, Size(20, 20), 0, 0, INTER_AREA);

			// HOG 벡터 계산
			vector<float> desc;
			hog.compute(img_resize, desc);

			Mat desc_mat(desc);

			// svm 결과 예측
			int res = cvRound(svm->predict(desc_mat.t()));

			// svm 결과를 콘솔 창에 출력
			cout << (char)res << endl;

			img.setTo(0);
			imshow("img", img);
		}
	}

	return 0;
}

Ptr<SVM> train_hog_svm(const HOGDescriptor& hog)
{
	// Alphabet.png 이미지 파일 읽기
	Mat digits = imread("Alphabet.png", IMREAD_GRAYSCALE);

	// 이미지 파일 읽기 실패시 0 반환
	if (digits.empty()) {
		cerr << "이미지 없음" << endl;
		return 0;
	}

	// 트레이닝 결과를 저장하기 위한 객체 
	Mat train_hog, train_labels;

	// Alphabet.png 이미지 내 알파벳의 개수 및 사이즈
	int row = 26;
	int col = 39;
	int sizex = 20;
	int sizey = 20;

	// 26 * 39 개의 필기체 숫자 영상으로부터 HOG 벡터를 추출
	for (int j = 0; j < row; j++) {
		for (int i = 0; i < col; i++) {
			Mat roi = digits(Rect(i * 20, j * 20, sizex, 20));

			// HOG 벡터 계산
			vector<float> desc;
			hog.compute(roi, desc);

			// 트레이닝 하기 위한 정보들을 저장
			Mat desc_mat(desc);
			train_hog.push_back(desc_mat.t());
			train_labels.push_back(j+65);		// 65 는 A 의 아스키코드

			// (디버깅용) 콘솔에 출력 
			//cout << roi << endl;
		}
	}

	// SVM 객체 생성
	Ptr<SVM> svm = SVM::create();
	
	// SVM 타입을 C_SVC 로 설정
	svm->setType(SVM::Types::C_SVC);
	
	// 커널 함수를 RBF 로 설정
	svm->setKernel(SVM::KernelTypes::RBF);

	// 파라미터 C 의 값을 62.5로 설정
	svm->setC(62.5);

	// 파라미터 Gamma 의 값을 0.03375 로 설정
	svm->setGamma(0.03375);

	// SVM 학습 진행
	svm->train(train_hog, ROW_SAMPLE, train_labels);

	return svm;
}

Point ptPrev(-1, -1);

// 마우스 이벤트
void on_mouse(int event, int x, int y, int flags, void* userdata)
{
	Mat img = *(Mat*)userdata;

	// 마우스 왼쪽 버튼을 누르면
	if (event == EVENT_LBUTTONDOWN)
		// 누른 위치값을 ptPrev 에 저장
		ptPrev = Point(x, y);

	// 마우스 왼쪽 버튼을 떼면
	else if (event == EVENT_LBUTTONUP)
		// ptPrev 값을 (-1, -1) 로 초기화
		ptPrev = Point(-1, -1);

	// 마우스 드래그 시
	else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON))
	{
		// ptPrev 좌표부터 (x, y) 까지 연결
		line(img, ptPrev, Point(x, y), Scalar::all(255), 40, LINE_AA, 0);

		// ptPrev 좌표를 (x, y) 로 변경
		ptPrev = Point(x, y);

		imshow("img", img);
	}
}