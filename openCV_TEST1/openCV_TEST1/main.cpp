#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>       // std::vector
//using namespace cv;

void showHistogram(const cv::Mat& image, const std::string& title) {
    // ���������� ���-�� ������� (��� �������� �����������):
    int histSize = 256;  // ���������� ������� ������
    float range[] = { 0, 256 }; // �������� ��������
    const float* histRange = { range };

    cv::Mat b_hist, g_hist, r_hist;

    // ��������� ����������� ��� ������� ������
    std::vector<cv::Mat> bgr_planes;
    cv::split(image, bgr_planes); // ��������� ������� ����������� �� ��� ������

    cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange);
    cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange);
    cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange);

    // ������� ����������� ��� ����������� �����������
    int hist_w = 512; // ������ �����������
    int hist_h = 400; // ������ �����������
    int bin_w = cvRound((double)hist_w / histSize); // ������ ������� ����

    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));

    // ����������� �����������
    cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX);
    cv::normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX);
    cv::normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX);

    // ������ �����������
    for (int i = 1; i < histSize; i++) {
        cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
            cv::Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
            cv::Scalar(255, 0, 0), 2, 8, 0); // �����

        cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
            cv::Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
            cv::Scalar(0, 255, 0), 2, 8, 0); // �������

        cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
            cv::Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
            cv::Scalar(0, 0, 255), 2, 8, 0); // �������
    }

    // ���������� ����������� � ������������
    cv::imshow(title, histImage);
}


/* 1 �������
int main()
{
    cv::Mat img = cv::imread("C:/Users/s20di/OneDrive/Desktop/image.png");
    namedWindow("First OpenCV Application", WINDOW_AUTOSIZE);
    cv::imshow("First OpenCV Application", img);
    cv::moveWindow("First OpenCV Application", 0, 45);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
*/
/*
int main(int argc, char** argv)
{
    // READ RGB color image and convert it to Lab
    cv::Mat bgr_image = cv::imread("C:/Users/s20di/OneDrive/Desktop/image.png");

    // Check if the image was loaded successfully
    if (bgr_image.empty()) {
        std::cerr << "Error: Unable to load image!" << std::endl;
        return -1;
    }

    cv::Mat lab_image;
    cv::cvtColor(bgr_image, lab_image, cv::COLOR_BGR2Lab);

    // Extract the L channel
    std::vector<cv::Mat> lab_planes(3);
    cv::split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]

    // apply the CLAHE algorithm to the L channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);
    cv::Mat dst;
    clahe->apply(lab_planes[0], dst);

    // Merge the color planes back into an Lab image
    dst.copyTo(lab_planes[0]);
    cv::merge(lab_planes, lab_image);

    // Convert back to RGB
    cv::Mat image_clahe;
    cv::cvtColor(lab_image, image_clahe, cv::COLOR_Lab2BGR);

    // Display the results
    cv::imshow("Image Original", bgr_image);
    cv::imshow("Image CLAHE", image_clahe);
    cv::waitKey(0);

    return 0;
}
*/



int main(int argc, char** argv)
{
    // ������ RGB ������� ����������� � ������������ ��� � Lab
    cv::Mat bgr_image = cv::imread("C:/Users/s20di/OneDrive/Desktop/image.png");

    // ��������� ���������� �������� �����������
    if (bgr_image.empty()) {
        std::cerr << "Error: Unable to load image!" << std::endl;
        return -1;
    }

    cv::Mat lab_image;
    cv::cvtColor(bgr_image, lab_image, cv::COLOR_BGR2Lab);

    // ��������� L-�����
    std::vector<cv::Mat> lab_planes(3);
    cv::split(lab_image, lab_planes);  // ������ � ��� ���� L ����������� � lab_planes[0]

    // ��������� �������� CLAHE � L-������
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);
    cv::Mat dst;
    clahe->apply(lab_planes[0], dst);

    // ���������� �������� ������ ������� � ����������� Lab
    dst.copyTo(lab_planes[0]);
    cv::merge(lab_planes, lab_image);

    // ������������ ������� � RGB
    cv::Mat image_clahe;
    cv::cvtColor(lab_image, image_clahe, cv::COLOR_Lab2BGR);

    // ���������� ����������
    cv::imshow("Image Original", bgr_image);
    cv::imshow("Image CLAHE", image_clahe);

    // ���������� �����������
    showHistogram(bgr_image, "Histogram Original");
    showHistogram(image_clahe, "Histogram CLAHE");

    cv::waitKey(0);
    return 0;
}
