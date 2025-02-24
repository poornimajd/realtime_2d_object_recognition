/*
  Poornima Jaykumar Dharamdasani
  feb 2025

  Showing different effects on video based on the key press.

*/
#include <opencv2/opencv.hpp>   
#include <opencv2/imgproc.hpp> 
#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>

#include <vector>

#include "threshold.h"

/*
 Applies a dynamic threshold using k-means clustering.
 Arguments:
 - img: Input color image.

 Functionality:
 - Blurs the image to reduce noise.
 - Converts the image to the Lab color space.
 - Flattens and converts the image for k-means clustering.
 - Clusters pixels into two groups (K=2).
 - Computes a threshold as the average of the two cluster centers.
 - Applies the threshold to the L channel to create a binary mask.
 Returns:
 - A binary image where pixels above the threshold are white (255), and others are black (0).
 */


cv::Mat dynamic_threshold(cv::Mat &img)
{

  cv::Mat blurred;
  cv::GaussianBlur(img,blurred, cv::Size(5,5), 0);
  cv::Mat labimage;
  cv::cvtColor(blurred, labimage, cv::COLOR_BGR2Lab);

  cv::Mat flatten_img = labimage.reshape(1,labimage.rows*labimage.cols);

  flatten_img.convertTo(flatten_img,CV_32F);

  int K=2;
  cv::Mat labels, centers;

  cv::kmeans(flatten_img, K, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers ); // apply kmeans to get the clusters.

  float center_1 = centers.at<float>(0,0); // get the two centers, the avg will likely be the pixel that can be used for thresholding.
  float center_2 =centers.at<float>(1,0);

  int threshval = static_cast<int>((center_1+center_2)/2.0); 

  std::vector<cv::Mat> labchannels;
  cv::split(labimage, labchannels);
  cv::Mat L_channel = labchannels[0];

 

  cv::Mat dst = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
  for(int i=0;i<dst.rows;i++)
    for(int j=0;j<dst.cols;j++)
    {
      uchar pixelval = L_channel.at<uchar>(i,j); 
      dst.at<uchar>(i,j)= (pixelval>=threshval)? 255:0;
    }

    return dst;
}