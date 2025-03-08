#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>       // std::vector
//using namespace cv;

void showHistogram(const cv::Mat& image, const std::string& title) {
    // Определяем кол-во каналов (для цветного изображения):
    int histSize = 256;  // количество уровней серого
    float range[] = { 0, 256 }; // диапазон значений
    const float* histRange = { range };

    cv::Mat b_hist, g_hist, r_hist;

    // Вычисляем гистограммы для каждого канала
    std::vector<cv::Mat> bgr_planes;
    cv::split(image, bgr_planes); // Разделяем цветное изображение на три канала

    cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange);
    cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange);
    cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange);

    // Создаем изображение для отображения гистограммы
    int hist_w = 512; // ширина гистограммы
    int hist_h = 400; // высота гистограммы
    int bin_w = cvRound((double)hist_w / histSize); // ширина каждого бина

    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));

    // Нормализуем гистограммы
    cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX);
    cv::normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX);
    cv::normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX);

    // Рисуем гистограммы
    for (int i = 1; i < histSize; i++) {
        cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
            cv::Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
            cv::Scalar(255, 0, 0), 2, 8, 0); // синий

        cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
            cv::Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
            cv::Scalar(0, 255, 0), 2, 8, 0); // зеленый

        cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
            cv::Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
            cv::Scalar(0, 0, 255), 2, 8, 0); // красный
    }

    // Показываем изображение с гистограммой
    cv::imshow(title, histImage);
}


/* 1 вариант
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
    // Читаем RGB цветное изображение и конвертируем его в Lab
    cv::Mat bgr_image = cv::imread("C:/Users/s20di/OneDrive/Desktop/image.png");

    // Проверяем успешность загрузки изображения
    if (bgr_image.empty()) {
        std::cerr << "Error: Unable to load image!" << std::endl;
        return -1;
    }

    cv::Mat lab_image;
    cv::cvtColor(bgr_image, lab_image, cv::COLOR_BGR2Lab);

    // Извлекаем L-канал
    std::vector<cv::Mat> lab_planes(3);
    cv::split(lab_image, lab_planes);  // сейчас у нас есть L изображение в lab_planes[0]

    // Применяем алгоритм CLAHE к L-каналу
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);
    cv::Mat dst;
    clahe->apply(lab_planes[0], dst);

    // Объединяем цветовые каналы обратно в изображение Lab
    dst.copyTo(lab_planes[0]);
    cv::merge(lab_planes, lab_image);

    // Конвертируем обратно в RGB
    cv::Mat image_clahe;
    cv::cvtColor(lab_image, image_clahe, cv::COLOR_Lab2BGR);

    // Отображаем результаты
    cv::imshow("Image Original", bgr_image);
    cv::imshow("Image CLAHE", image_clahe);

    // Показываем гистограммы
    showHistogram(bgr_image, "Histogram Original");
    showHistogram(image_clahe, "Histogram CLAHE");

    cv::waitKey(0);
    return 0;
}
