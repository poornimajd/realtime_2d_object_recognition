/*
  Poornima Jaykumar Dharamdasani
  feb 2025

  Implemented erosion and dilation from scratch

*/
#include <opencv2/opencv.hpp>   
#include <opencv2/imgproc.hpp> 
#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>

#include <vector>

#include "morph.h"

/*
 Applies erosion to a binary image.
 Arguments:
 - src: Input binary image.
 - dst: Output image after erosion.
 - kernel: Structuring element for erosion.

 Functionality:
 - Iterates over each pixel in the source image.
 - Checks if all pixels in the neighborhood (defined by the kernel) are white (255).
 - If any pixel in the kernel region is black (0), sets the output pixel to black.
 - Otherwise, keeps it white.
 */

void erode_img(cv::Mat &src, cv::Mat &dst ,cv::Mat &kernel)
{
  dst = src.clone();
  int krows = kernel.rows/2;
  int kcols = kernel.cols/2;

  for(int i=0;i<src.rows;i++)
  {
    for(int j=0;j<src.cols;j++)
    {
      bool erode = true;
      for(int x = -krows;x<=krows && erode;x++)
      {
        for(int y= -kcols;y<=kcols;y++)
        {
          int kx = i+x;
          int ky = j+y;
          if(kx>=0 && kx<src.rows && ky>=0 && ky<src.cols)
          {
            if(src.at<uchar>(kx,ky)==0)
            {
              erode=false;
              break;
            }
          }

        }
      }
      dst.at<uchar>(i,j)=erode?255:0;
    }
  }
}


/*
 Applies dilation to a binary image.
 Arguments:
 - src: Input binary image.
 - dst: Output image after dilation.
 - kernel: Structuring element for dilation.

 Functionality:
 - Iterates over each pixel in the source image.
 - Checks if any pixel in the neighborhood (defined by the kernel) is white (255).
 - If at least one pixel is white, sets the output pixel to white.
 - Otherwise, keeps it black.
 */

void dilate_img(cv::Mat &src, cv::Mat &dst ,cv::Mat &kernel)
{
  dst = src.clone();
  int krows = kernel.rows/2;
  int kcols = kernel.cols/2;

  for(int i=0;i<src.rows;i++)
  {
    for(int j=0;j<src.cols;j++)
    {
      bool dilate = true;
      for(int x = -krows;x<=krows && dilate;x++)
      {
        for(int y= -kcols;y<=kcols;y++)
        {
          int kx = i+x;
          int ky = j+y;
          if(kx>=0 && kx<src.rows && ky>=0 && ky<src.cols)
          {
            if(src.at<uchar>(kx,ky)==255)
            {
              dilate=false;
              break;
            }
          }

        }
      }
      dst.at<uchar>(i,j)=dilate?0:255;
    }
  }
}

/*
 Performs morphological opening on an image.
 Arguments:
 - src: Input binary image.
 - dst: Output image after morphological opening.

 Functionality:
 - Creates a 5x5 kernel.
 - First applies erosion to remove small noise.
 - Then applies dilation to restore the main structure.
 */

void morphological_op(cv::Mat &src, cv::Mat &dst)
{
  cv::Mat Kernel=cv::Mat::ones(5,5, CV_8U);
  cv::Mat erodeimg;
  erode_img(src,erodeimg,Kernel);
  dilate_img(erodeimg,dst,Kernel);
}