// /*
//   Poornima Jaykumar Dharamdasani
//   feb 2025

//   Code to compute the regions and compute the features for each region.

// */

#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "segment.h"



void findfeatures(cv::Mat &src, RegionFeatures &region);
/*
 Segments connected regions in a binary image and assigns random colors.
 Arguments:
 - src: Input binary image.
 - dst: Output color image with labeled segments.
 - minsize: Minimum size threshold for segments to be kept.

 Functionality:
 - Inverts the binary image.
 - Finds connected components (segments).
 - Filters out small segments.
 - Extracts features for each segment.
 - Assigns random colors to different segments.
 - Converts the output to a color image if needed.
 Returns:
 - A vector of RegionFeatures for detected segments.
 */

std::vector<RegionFeatures> color_the_segments(cv::Mat &src, cv::Mat &dst, int minsize)
{
	dst = src.clone();

	cv::Mat labelimg, stats, centroids,invert_img;
	cv::bitwise_not(dst,invert_img);
	std::vector<RegionFeatures> image_regions;
	int nLabels = cv::connectedComponentsWithStats(invert_img, labelimg, stats, centroids, 8, CV_32S);

	for(int i = 1;i<nLabels;i++)
    {   

    	
		if(stats.at<int>(i,cv::CC_STAT_AREA)<minsize)
		{	
    	cv::Mat mask = (labelimg == i);
    	labelimg.setTo(0,mask);
    	continue;
		}
		RegionFeatures region;
        region.region_id = i;
        int posx = static_cast<int>(centroids.at<double>(i,0));
        int posy=static_cast<int>(centroids.at<double>(i,1));
        region.centroid=cv::Point2f(posx,posy); //store as centroid

		cv::Mat mask = (labelimg == i);

        findfeatures(mask, region); //find the features for the given labeled region.
        image_regions.push_back(region);

}


	if (dst.channels() == 1) {  
        cv::cvtColor(dst, dst, cv::COLOR_GRAY2BGR);
    }


	std::vector<cv::Vec3b> colors(nLabels);
    colors[0] = cv::Vec3b(0, 0, 0);//background
    for(int label = 1; label < nLabels; ++label){
        colors[label] = cv::Vec3b( (rand()&255), (rand()&255), (rand()&255));
    }
    

    for(int r = 0; r < dst.rows; ++r){
        for(int c = 0; c < dst.cols; ++c){
            int label = labelimg.at<int>(r, c);
            dst.at<cv::Vec3b>(r, c) = colors[label];  

         }
     }
     
     return image_regions;

}

/*
 Calls the segmentation function with a default minimum size.
 Arguments:
 - src: Input binary image.
 - dst: Output color image with labeled segments.

 Functionality:
 - Calls `color_the_segments` with a minimum size of 500.
 Returns:
 - A vector of RegionFeatures.
 */


std::vector<RegionFeatures> segment_and_features(cv::Mat &src, cv::Mat &dst)
{


	return color_the_segments(src,dst, 500);
	
}
/*
 Extracts features from a segmented region.
 Arguments:
 - src: Binary mask of a single region.
 - region: Structure to store extracted features.

 Functionality:
 - Computes Hu moments for shape analysis.
 - Finds contours to get the bounding box.
 - Calculates aspect ratio and fill percentage.
 - Computes covariance matrix and eigenvalues to get major/minor axes.
 */

void findfeatures(cv::Mat &src, RegionFeatures &region)
{

		cv::Moments moments1 = cv::moments(src, true);


	    double huMoments[7];
	    cv::HuMoments(moments1, huMoments);

        std::vector<std::vector<cv::Point>> contours;
        
        
        cv::findContours(src, contours,  cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        if (contours.empty()) {
        	std::cout << "No contours found!" << std::endl;
        	return;
        }
        double M00,M01,M10;

        M00 = moments1.m00;
        M10 = moments1.m10;
        M01 = moments1.m01;

        if (M00 == 0) return; 


        if (!contours.empty())  // Ensure enough points for minAreaRect
        {
            cv::RotatedRect rotatedBox = cv::minAreaRect(contours[0]); // Get rotated rectangle

            // Convert to a box (4 corner points)
            cv::Point2f boxPoints[4];
            rotatedBox.points(boxPoints);

            double boxarea = rotatedBox.size.width*rotatedBox.size.height;

            double cntarea = cv::contourArea(contours[0]);
            double percentfilled = (cntarea/boxarea) * 100 ;

            double aspectRatio = rotatedBox.size.height / rotatedBox.size.width;
            

            region.aspectRatio =aspectRatio;
            region.percentfilled =percentfilled;

            for (int i = 0; i < 7; i++) 
            {
			    region.huMoments[i] = huMoments[i]; //scale invariant.
			}

			region.bounding_box = rotatedBox;

			double m20 = moments1.mu20/M00;
            double m02 = moments1.mu02/M00;
            double m11 = moments1.mu11/M00;

            cv::Mat covMat = cv::Mat::zeros(2,2,CV_64F); //construct cov matrix
            covMat.at<double>(0,0) = m20;
            covMat.at<double>(0,1) = m11;
            covMat.at<double>(1,0) = m11;
            covMat.at<double>(1,1) = m02;
            // use pca based method to get the axis.

            cv::Mat eigenvalues, eigenvectors;
            cv::eigen(covMat, eigenvalues, eigenvectors);

            cv::Vec2d majoraxis(eigenvectors.at<double>(0,0),eigenvectors.at<double>(0,1));
            cv::Vec2d minoraxis(eigenvectors.at<double>(1,0),eigenvectors.at<double>(1,1));

            region.major_axis = majoraxis;
            region.minor_axis=minoraxis;


        }

        

}
