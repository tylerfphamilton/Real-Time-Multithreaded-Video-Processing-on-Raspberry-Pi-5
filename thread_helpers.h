
/*******************************************************
* File: gray_sobel_filtering.h
*
* Description: 
*
* Author: Tyler Hamilton
*
* Revision history
*
********************************************************/ 

#ifndef THREAD_HELPERS_H
#define THREAD_HELPERS_H

#include <pthread.h>
#include <opencv2/opencv.hpp>
#include <vector>

// struct for each horizontal band (stripe) for threading
struct Stripe {

    cv::Rect readROI;
    cv::Rect writeROI;
    int start;
};

// struct for the thread data
struct ThreadData {

    int id;                         // the specific thread 

    std::vector<Stripe> stripes;    // stripes for processing

    const cv::Mat* frame;           // pointer to shared frame
    cv::Mat* output;                // pointer to outpur frame
    pthread_barrier_t* barrier;     // pointer to barrier to coordinate frames
    bool* stopFlag;                 // signals when to stop

    cv::Mat gray;                   // per thread buffer for gray
};

// function prototypes
void* worker_func(void* arg);
void start_threads(std::vector<pthread_t>& threads, std::vector<ThreadData>& data, pthread_barrier_t* barrier, cv::Mat* frame, cv::Mat* output, bool* stopFlag);
void stop_threads(std::vector<pthread_t>& threads, pthread_barrier_t* barrier);

#endif