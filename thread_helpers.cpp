/*********************************************************
* File: thread_helpers.cpp
*
* Description: brief description of file purpose
*
* Author: Tyler Hamilton
*
* Revisions:
*
**********************************************************/

// header files to inclue
#include "thread_helpers.h"     
#include "gray_sobel_neon_filtering.h"          


// function for worker functions
void* worker_func(void* input){

    // type casting the argument into a ThreadData struct
    ThreadData* thread_data = static_cast<ThreadData*>(input);

    // while the stop flag is false
    while (true){

        // waiting for the first barrier
        pthread_barrier_wait(thread_data->barrier);

        // check stopFlag
        if (*(thread_data->stopFlag)) {
            break; 
        }

        // // creating the Matrix variables for gray and sobel 
        cv::Mat& gray  = thread_data->gray;
        // cv::Mat& sobel = thread_data->sobel;

        for (const Stripe& stripe : thread_data->stripes){

            // ROI for stripe
            cv::Mat roi = (*thread_data->frame)(stripe.readROI);

            // calling both the grayscle filter function using neon vectors
            to442_grayscale_neon(roi,gray);

            // Destination is exactly this stripe in the final output
            cv::Mat dest = (*thread_data->output)(stripe.writeROI);
            
            // local start/stop declarations
            int local_start = stripe.start;
            int local_stop = stripe.start + stripe.writeROI.height;

            // Copy from sobel stripe into output stripe
            to442_sobel_neon(gray, dest, local_start, local_stop);
        }

        // Barrier #2: signal that this thread is done with the frame
        pthread_barrier_wait(thread_data->barrier);
    }

    return nullptr;
}


// function for starting the threads
void start_threads(std::vector<pthread_t>& threads, std::vector<ThreadData>& data, pthread_barrier_t* barrier,
                   cv::Mat* frame, cv::Mat* output, bool* stopFlag){

    // need to create the 4 threads
    for (long unsigned int i = 0; i < threads.size(); ++i){

        // assigning function inputs to struct variables 
        data[i].id = i;
        data[i].barrier = barrier;
        data[i].frame = frame;
        data[i].output = output;
        data[i].stopFlag = stopFlag;

        // creating the thread with input data
        pthread_create(&threads[i], NULL, worker_func, &data[i]);
    }
}


// function for stopping the threads
void stop_threads(std::vector<pthread_t>& threads, pthread_barrier_t* barrier){

    // joining the threads together
    for (size_t i = 0; i < threads.size(); ++i) {
        pthread_t& t = threads[i];  
        pthread_join(t, NULL);
    }

    // destroying the barrier
    pthread_barrier_destroy(barrier);
}