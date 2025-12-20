/*******************************************************
* File: gray_sobel_filtering.h
*
* Description: brief description of file purpose
*
* Author: Tyler Hamilton
*
* Revision history
*
********************************************************/ 

#ifndef _GRAY_SOBEL_FILTERING_H
#define _GRAY_SOBEL_FILTERING_H

// opencv and opening an image
#include <opencv2/opencv.hpp>   
#include <iostream>             
#include <cmath>                
#include <cstdlib>   

#pragma once

// function for gray scaling
cv::Mat to442_grayscale(const cv::Mat& bgr, cv::Mat& gray) ;
// function for sobel filtering
cv::Mat to442_sobel(const cv::Mat& gray, cv::Mat& sobel);
#endif
