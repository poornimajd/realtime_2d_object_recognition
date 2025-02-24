

/*
  Poornima Jaykumar Dharamdasani
  FEb 2025

  Include file for segment.cpp

*/


#ifndef SEGMENT_H
#define SEGMENT_H
#include<opencv2/opencv.hpp>
// struct to store all the features for a region. Assigned default values.
struct RegionFeatures {
    int region_id = -1;  // -1 means "no valid region"
    cv::RotatedRect bounding_box = cv::RotatedRect(cv::Point2f(-1, -1), cv::Size2f(0, 0), 0);
    double aspectRatio = -1.0;  
    double percentfilled = -1.0;  
    cv::Vec2d major_axis = cv::Vec2d(0, 0);  
    cv::Vec2d minor_axis = cv::Vec2d(0, 0);
    cv::Point2f centroid = cv::Point2f(-1, -1);  
    double huMoments[7];
};


std::vector<RegionFeatures> segment_and_features(cv::Mat &src, cv::Mat &dst);

#endif