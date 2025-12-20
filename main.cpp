/*********************************************************
* File: main.cpp
*
* Description: brief description of file purpose
*
* Author: Tyler Hamilton
*
* Revisions:
*
**********************************************************/

// header files to include
#include "thread_helpers.h"
#include "gray_sobel_filtering.h"

/*-----------------------------------------------------
* Function: main
*
* Description: main, reads the command line arguments, calls threading and image processing functions (grayscale and sobel)
*
* param argc: int: number of arguments for command line
* param argv[]: char*: string of arguments
*
* return: int
*--------------------------------------------------------*/

// Macros for number of workers and stripes
constexpr int NUM_WORKERS = 4;
constexpr int NUM_STRIPES = 16; 

// main function
int main (int argc, char* argv[]){

    // argument checker
    if (argc != 2){
        perror("There are not two arguments\n");
    }

    // open the video
    cv::VideoCapture cap(argv[1]);

    // check to see if there is an error
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file, usgae: " << argv[0] << std::endl;
        return -1; 
    }

    // name the window and Matrix for the frame and gray scale conversion
    const char* new_window = "gray and sobel filter";
    cv::namedWindow(new_window, cv::WINDOW_NORMAL);
    cv::Mat frame;

    // check to see if there is an error opening the frames
    if (!cap.read(frame) || frame.empty()) {
        std::cerr << "Error: Could not read first frame from video." << std::endl;
        return -1;
    }

    // height and width of frame
    int width  = frame.cols;
    int height = frame.rows;

    // output sobel image
    cv::Mat output(height, width, CV_8UC1);

    // threading
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, NUM_WORKERS + 1); // 4 workers + main thread

    // setting the stop flag to false
    bool stopFlag = false;

    // creating the threads and data placeholder variables
    std::vector<pthread_t> threads(NUM_WORKERS);
    std::vector<ThreadData> data(NUM_WORKERS);

    // building the stripes
    int stripeHeight = height / NUM_STRIPES;
    std::vector<Stripe> stripes;
    stripes.reserve(NUM_STRIPES);

    // looping through each stripe and filling out the struct
    for (int s = 0; s < NUM_STRIPES; s++){

        // getting each stripe start and end height
        int startHeight = s * stripeHeight;
        int endHeight = (s == NUM_STRIPES - 1) ? height : (s + 1) * stripeHeight;

        // getting the correct values to read for the top and bottom of stripe
        int readTop = std::max(0, startHeight - 1);
        int readBottom = std::min(height, endHeight + 1);

        // creating a stripe and filling in the fields with the correct data
        Stripe stripe;
        stripe.readROI = cv::Rect(0, readTop, width, readBottom - readTop);
        stripe.writeROI = cv::Rect(0, startHeight, width, endHeight - startHeight);
        stripe.start = startHeight - readTop;

        // adding the stripe to the end of vector stack
        stripes.push_back(stripe);
    }

    // iterating through each of the 4 rows and writing data to each of the threads
    for (int w = 0; w < NUM_WORKERS; ++w){

        // writing data to the thread struct
        data[w].id = w;
        data[w].barrier = &barrier;
        data[w].frame = &frame;
        data[w].output = &output;
        data[w].stopFlag = &stopFlag;

        for (int s = w; s < NUM_STRIPES; s+= NUM_WORKERS){
            data[w].stripes.push_back(stripes[s]);
        }
    }

    // start the threads
    start_threads(threads, data, &barrier, &frame, &output, &stopFlag);

    // getting the frames
    int frameCount = 0;

    // while loop for reading the video
    while (true){

        // checking to see if I am still reading a frame
        bool frames = cap.read(frame);

        if (!frames || frame.empty()){

            // stop the processing and wait for threads to stop
            stopFlag = true;
            pthread_barrier_wait(&barrier);
            // pthread_barrier_wait(&barrier);
            break;
        }

        // incrementing frame count
        frameCount++;

        // barrier #1: signal new frame ready
        pthread_barrier_wait(&barrier);

        // barrier #2: wait for all threads to finish
        pthread_barrier_wait(&barrier);

        // showing the frame
        cv::imshow(new_window,output);

        // if there is an esc key pressed
        if (cv::waitKey(1) == 27){
            
            // stop the processing and wait for threads to stop
            stopFlag = true;
            pthread_barrier_wait(&barrier);  
            // pthread_barrier_wait(&barrier);
            break;
        } 
    }

   // end of the video
    std::cout << "End of the video" << std::endl;

    // printing total frames
    std::cout << "Frames processed: " << frameCount << std::endl;

    // stop the threads
    stop_threads(threads, &barrier);

    // Release the VideoCapture object
    cap.release();

    // Destroy all OpenCV windows
    cv::destroyAllWindows();

    // end of function (success)
    return 0;
}
