/*******************************************************
* File: gray_sobel_neon_filtering.h
*
* Description: brief description of file purpose
*
* Author: Tyler Hamilton
*
* Revision history
*
********************************************************/ 

#ifndef _GRAY_SOBEL_NEON_FILTERING_H
#define _GRAY_SOBEL_NEON_FILTERING_H

// opencv and opening an image
#include <opencv2/opencv.hpp>   
#include <iostream>             
#include <cmath>                
#include <cstdlib>
#include <arm_neon.h>


// function for gray scaling
cv::Mat to442_grayscale_neon(const cv::Mat& bgr, cv::Mat& gray) ;
// function for sobel filtering
cv::Mat to442_sobel_neon(const cv::Mat& gray, cv::Mat& sobel, int local_start, int local_end);
#endif
