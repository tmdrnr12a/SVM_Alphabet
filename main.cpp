#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;

Ptr<SVM> train_hog_svm(const HOGDescriptor& hog);
void on_mouse(int event, int x, int y, int flags, void* userdata);

int main()
{
	// HOGDescriptor Ŭ���� �������� ������ ���� ����� ���� �����ϸ�, �����߻�
# if _DEBUG
	cout << "svmdigit.exe should be built as Release mode!" << endl;
	return 0;
#endif

	// �ʱ�ü ���� ���󿡼� HOG ���͸� �����ϱ� ���� HOGDescriptor Ŭ���� ���
	// 20x20 ����, 5x5 ��, 10x10 ���, �� ������ 9���� �׶���Ʈ ���� ������׷�
	// HOGDescriptor ��ü hog ����
	HOGDescriptor hog(Size(20, 20), Size(10, 10), Size(5, 5), Size(5, 5), 9);

	// SVM �н� ����
	Ptr<SVM> svm = train_hog_svm(hog);

	// SVM �н� ���н� ���α׷� ����
	if (svm.empty()) {
		cerr << "�Ʒ� ����" << endl;
		return -1;
	}

	Mat img = Mat::zeros(400, 400, CV_8U);
	imshow("img", img);

	// ���콺 �̺�Ʈ ����
	setMouseCallback("img", on_mouse, (void*)&img);

	while (true) {
		// Ű���� �Է� ���
		int c = waitKey();

		// ESC Ű
		if (c == 27) {
			break;
		}

		// Space Ű
		else if (c == ' ') {
			Mat img_resize;

			// img �� 20x20 ũ��� ��ȯ
			resize(img, img_resize, Size(20, 20), 0, 0, INTER_AREA);

			// HOG ���� ���
			vector<float> desc;
			hog.compute(img_resize, desc);

			Mat desc_mat(desc);

			// svm ��� ����
			int res = cvRound(svm->predict(desc_mat.t()));

			// svm ����� �ܼ� â�� ���
			cout << (char)res << endl;

			img.setTo(0);
			imshow("img", img);
		}
	}

	return 0;
}

Ptr<SVM> train_hog_svm(const HOGDescriptor& hog)
{
	// Alphabet.png �̹��� ���� �б�
	Mat digits = imread("Alphabet.png", IMREAD_GRAYSCALE);

	// �̹��� ���� �б� ���н� 0 ��ȯ
	if (digits.empty()) {
		cerr << "�̹��� ����" << endl;
		return 0;
	}

	// Ʈ���̴� ����� �����ϱ� ���� ��ü 
	Mat train_hog, train_labels;

	// Alphabet.png �̹��� �� ���ĺ��� ���� �� ������
	int row = 26;
	int col = 39;
	int sizex = 20;
	int sizey = 20;

	// 26 * 39 ���� �ʱ�ü ���� �������κ��� HOG ���͸� ����
	for (int j = 0; j < row; j++) {
		for (int i = 0; i < col; i++) {
			Mat roi = digits(Rect(i * 20, j * 20, sizex, 20));

			// HOG ���� ���
			vector<float> desc;
			hog.compute(roi, desc);

			// Ʈ���̴� �ϱ� ���� �������� ����
			Mat desc_mat(desc);
			train_hog.push_back(desc_mat.t());
			train_labels.push_back(j+65);		// 65 �� A �� �ƽ�Ű�ڵ�

			// (������) �ֿܼ� ��� 
			//cout << roi << endl;
		}
	}

	// SVM ��ü ����
	Ptr<SVM> svm = SVM::create();
	
	// SVM Ÿ���� C_SVC �� ����
	svm->setType(SVM::Types::C_SVC);
	
	// Ŀ�� �Լ��� RBF �� ����
	svm->setKernel(SVM::KernelTypes::RBF);

	// �Ķ���� C �� ���� 62.5�� ����
	svm->setC(62.5);

	// �Ķ���� Gamma �� ���� 0.03375 �� ����
	svm->setGamma(0.03375);

	// SVM �н� ����
	svm->train(train_hog, ROW_SAMPLE, train_labels);

	return svm;
}

Point ptPrev(-1, -1);

// ���콺 �̺�Ʈ
void on_mouse(int event, int x, int y, int flags, void* userdata)
{
	Mat img = *(Mat*)userdata;

	// ���콺 ���� ��ư�� ������
	if (event == EVENT_LBUTTONDOWN)
		// ���� ��ġ���� ptPrev �� ����
		ptPrev = Point(x, y);

	// ���콺 ���� ��ư�� ����
	else if (event == EVENT_LBUTTONUP)
		// ptPrev ���� (-1, -1) �� �ʱ�ȭ
		ptPrev = Point(-1, -1);

	// ���콺 �巡�� ��
	else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON))
	{
		// ptPrev ��ǥ���� (x, y) ���� ����
		line(img, ptPrev, Point(x, y), Scalar::all(255), 40, LINE_AA, 0);

		// ptPrev ��ǥ�� (x, y) �� ����
		ptPrev = Point(x, y);

		imshow("img", img);
	}
}