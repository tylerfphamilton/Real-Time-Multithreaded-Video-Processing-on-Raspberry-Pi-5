// /*********************************************************
// * File: gray_sobel_filtering.cpp
// *
// * Description: brief description of file purpose
// *
// * Author: Tyler Hamilton
// *
// * Revisions:
// *
// **********************************************************/

#include "gray_sobel_filtering.h"

// /*-----------------------------------------------------
// * Function: gray_sobel_filtering
// *
// * Description: reads an input frame, converts it to grayscale, then adds a sobel filter 
// *
// *
// *--------------------------------------------------------*/


// function for gray scaling
cv::Mat to442_grayscale(const cv::Mat& bgr, cv::Mat& gray) {

   // memory allocation for the var gray
    gray.create(bgr.rows, bgr.cols, CV_8UC1);

    // looping through the rows of bgr
    for (int row = 0; row < bgr.rows; ++row){

	    // creating a temp dest to store the grayscaled values
        const cv::Vec3b* temp_src = bgr.ptr<cv::Vec3b>(row);
        uchar* temp_dst = gray.ptr<uchar>(row);

	    // for loop through the cols
        for (int col = 0; col < bgr.cols; ++col){

	        // getting the pixel and assiging variables to it
            const cv::Vec3b& pixel = temp_src[col];
            uchar blue = pixel[0];
            uchar green = pixel[1];
            uchar red = pixel[2];

	        // applying the BT.709 changes
            float output = (0.2126f*red) + (0.7152f*green) + (0.0722f*blue);
            int vector = (int) std::lround(output);

	        // clamping
            if (vector < 0) vector = 0;
            else if (vector > 255) vector = 255;
		
	        // writing grayscaled value to variable
            temp_dst[col] = static_cast<uchar>(vector);
        }
    }

    // returning new grayscaled matrix
    return gray;
}


// function for sobel filtering
cv::Mat to442_sobel(const cv::Mat& gray, cv::Mat& sobel){

    // memory allocation for the sobel var 
    sobel.create(gray.rows, gray.cols, CV_8UC1);
    sobel.setTo(0);

    // looping through the rows
    for (int row = 1; row < gray.rows -1; ++row){

	// assigning variables for boundary checking
        const uchar* prev = gray.ptr<uchar>(row - 1);
        const uchar* curr = gray.ptr<uchar>(row);
        const uchar* next = gray.ptr<uchar>(row + 1);
        uchar* temp_dst = sobel.ptr<uchar>(row);

	    // looping through cols
        for (int col = 1; col < gray.cols -1; ++col){

	        // setting the values in the matrix
            int a = prev[col-1], b = prev[col], c = prev[col+1];
            int d = curr[col-1], f = curr[col+1];
            int g = next[col-1], h = next[col], i = next[col+1];

	        // applying gradient math
            int Gx = (c + 2*f + i) - (a + 2*d + g);
            int Gy = (g + 2*h + i) - (a + 2*b + c);
            int G = std::abs(Gx) + std::abs(Gy);
            
	        // clamping
            if (G > 255) G = 255;

	        // writing to temp var
            temp_dst[col] = static_cast<uchar>(G);
        }
    }

    // returning new frames w sobel filter
    return sobel;
}

