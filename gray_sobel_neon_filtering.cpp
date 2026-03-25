// /*********************************************************
// * File: gray_sobel_neon_filtering.cpp
// *
// * Description: brief description of file purpose
// *
// * Author: Tyler Hamilton
// *
// * Revisions:
// *
// **********************************************************/

#include "gray_sobel_neon_filtering.h"

// /*-----------------------------------------------------
// * Function: gray and sobel filterin using neon vecotrs
// *
// * Description: reads an input frame, converts it to grayscale, then adds a sobel filter using SIMD neon intrinsics
// *
// *
// *--------------------------------------------------------*/

// function for gray scaling
cv::Mat to442_grayscale_neon(const cv::Mat& bgr, cv::Mat& gray) {

    // memory allocation for the var gray
    gray.create(bgr.rows, bgr.cols, CV_8UC1);

    for (int y = 0; y < bgr.rows; ++y){

        // for neon intrinsics (pointers to image data)
        const uint8_t* source = bgr.ptr<uint8_t>(y);
        uint8_t* destination = gray.ptr<uint8_t>(y);

        int x = 0;
        // Because I am using a pi 5, I can do 16 8 bit operations, which is why we iterate with +=16
        for (; x + 15 < bgr.cols; x+=16){

            // for the offset
            const uint8_t* offset = source + (x*3);

            // splitting into blue, green, and red
            uint8x16x3_t split_bgr = vld3q_u8(offset);

            // getting the lower and upper byte for blue, green and red
            uint16x8_t lower_blue = vmull_u8(vget_low_u8(split_bgr.val[0]), vdup_n_u8(18));
            uint16x8_t lower_green = vmull_u8(vget_low_u8(split_bgr.val[1]), vdup_n_u8(183));
            uint16x8_t lower_red = vmull_u8(vget_low_u8(split_bgr.val[2]), vdup_n_u8(54));
            uint16x8_t upper_blue = vmull_u8(vget_high_u8(split_bgr.val[0]), vdup_n_u8(18));
            uint16x8_t upper_green = vmull_u8(vget_high_u8(split_bgr.val[1]), vdup_n_u8(183));
            uint16x8_t upper_red = vmull_u8(vget_high_u8(split_bgr.val[2]), vdup_n_u8(54));

            // summing them
            uint16x8_t lower_total = vaddq_u16(vaddq_u16(lower_blue,lower_green) , lower_red);
            uint16x8_t upper_total = vaddq_u16(vaddq_u16(upper_blue,upper_green), upper_red);

            // dividing the summed values by 256
            uint16x8_t rnd = vdupq_n_u16(128);
            uint16x8_t lower_output = vshrq_n_u16(vaddq_u16(lower_total, rnd), 8);
            uint16x8_t upper_output = vshrq_n_u16(vaddq_u16(upper_total, rnd), 8);
            
            // getting ready to convert back to uint8x16
            uint8x8_t lower_final = vqmovn_u16(lower_output);
            uint8x8_t upper_final = vqmovn_u16(upper_output);

            // combining back into uint8x16
            uint8x16_t final_gray_vector = vcombine_u8(lower_final, upper_final);

            // storing back into the pointer to the grayscale output
            vst1q_u8(destination + x, final_gray_vector);
        }

        // getting the remaining cols (if there are any)
        for (; x < bgr.cols ; ++x) {

            // getting the values for BGR
            const uint8_t B = source[(3*x) + 0];
            const uint8_t G = source[(3*x) + 1];
            const uint8_t R = source[(3*x) + 2];

            // applying the math
            const unsigned sum = 18u * B + 183u * G + 54u * R;

            // shifting them 8 spots to the right and then storing back in pointer to gray
            destination[x] = static_cast<uint8_t>((sum + 128u) >> 8); 
        }
    }

    // returning them Mat of the grayscale output
    return gray;
}


// function for sobel filtering
cv::Mat to442_sobel_neon(const cv::Mat& gray, cv::Mat& dest, int local_start, int local_stop){

    // allocation memory for the dest Mat and intially setting all the frames to black 
    // sobel.create(gray.rows, gray.cols, CV_8UC1);
    dest.setTo(0);

    // start and stop values for each row
    int row_start = std::max(local_start, 1);
    int row_stop = std::min(local_stop, gray.rows - 1);

    for (int row = row_start; row <row_stop; ++row){

        // need to grab the three pointers (prev, curr, and next)
        const uint8_t* prev = gray.ptr<uint8_t>(row - 1);
        const uint8_t* curr = gray.ptr<uint8_t>(row);
        const uint8_t* next = gray.ptr<uint8_t>(row + 1);

        // direct write-out path
        uint8_t* temp_dst = dest.ptr<uint8_t>(row - local_start);

        // iterating through the cols but starting at one
        int col = 1;
        for (; col + 16 < gray.cols; col+=16){

            // calculating a,b,c,d,f,g,h,i
            uint8x16_t a = vld1q_u8(prev + col-1), b = vld1q_u8(prev + col), c = vld1q_u8(prev + col+1);
            uint8x16_t d = vld1q_u8(curr + col-1), f = vld1q_u8(curr + col+1);
            uint8x16_t g = vld1q_u8(next + col-1), h = vld1q_u8(next + col), i = vld1q_u8(next + col+1);

            // widening the values
            uint16x8_t lower_byte_a = vmovl_u8(vget_low_u8(a));
            uint16x8_t upper_byte_a = vmovl_u8(vget_high_u8(a));
            uint8x8_t lower_byte_b = vget_low_u8(b);
            uint8x8_t upper_byte_b = vget_high_u8(b);
            uint16x8_t lower_byte_c = vmovl_u8(vget_low_u8(c));
            uint16x8_t upper_byte_c = vmovl_u8(vget_high_u8(c));
            uint8x8_t lower_byte_d = vget_low_u8(d);
            uint8x8_t upper_byte_d = vget_high_u8(d);
            uint8x8_t lower_byte_f = vget_low_u8(f);
            uint8x8_t upper_byte_f = vget_high_u8(f);
            uint16x8_t lower_byte_g = vmovl_u8(vget_low_u8(g));
            uint16x8_t upper_byte_g = vmovl_u8(vget_high_u8(g));
            uint8x8_t lower_byte_h = vget_low_u8(h);
            uint8x8_t upper_byte_h = vget_high_u8(h);
            uint16x8_t lower_byte_i = vmovl_u8(vget_low_u8(i));
            uint16x8_t upper_byte_i = vmovl_u8(vget_high_u8(i));

            // widening the values and mutipling the vectors by 2
            uint16x8_t multiplied_f_lower = vmull_u8(lower_byte_f, vdup_n_u8(2));
            uint16x8_t multiplied_f_upper = vmull_u8(upper_byte_f, vdup_n_u8(2));
            uint16x8_t multiplied_d_lower = vmull_u8(lower_byte_d, vdup_n_u8(2));
            uint16x8_t multiplied_d_upper = vmull_u8(upper_byte_d, vdup_n_u8(2));
            uint16x8_t multiplied_h_lower = vmull_u8(lower_byte_h, vdup_n_u8(2));
            uint16x8_t multiplied_h_upper = vmull_u8(upper_byte_h, vdup_n_u8(2));
            uint16x8_t multiplied_b_lower = vmull_u8(lower_byte_b, vdup_n_u8(2));
            uint16x8_t multiplied_b_upper = vmull_u8(upper_byte_b, vdup_n_u8(2));

            // need to get the lower and upper vectors to computer Gx
            uint16x8_t lower_c2fi = vaddq_u16(vaddq_u16(lower_byte_c, lower_byte_i), multiplied_f_lower);
            uint16x8_t upper_c2fi = vaddq_u16(vaddq_u16(upper_byte_c, upper_byte_i), multiplied_f_upper);
            uint16x8_t lower_a2dg = vaddq_u16(vaddq_u16(lower_byte_a, lower_byte_g), multiplied_d_lower);
            uint16x8_t upper_a2dg = vaddq_u16(vaddq_u16(upper_byte_a, upper_byte_g), multiplied_d_upper);
            uint16x8_t lower_g2hi = vaddq_u16(vaddq_u16(lower_byte_g, lower_byte_i), multiplied_h_lower);
            uint16x8_t upper_g2hi = vaddq_u16(vaddq_u16(upper_byte_g, upper_byte_i), multiplied_h_upper);
            uint16x8_t lower_a2bc = vaddq_u16(vaddq_u16(lower_byte_a, lower_byte_c), multiplied_b_lower);
            uint16x8_t upper_a2bc = vaddq_u16(vaddq_u16(upper_byte_a, upper_byte_c), multiplied_b_upper);

            // computing Gx and Gy
            int16x8_t lower_Gx = vsubq_s16(vreinterpretq_s16_u16(lower_c2fi), vreinterpretq_s16_u16(lower_a2dg));
            int16x8_t upper_Gx = vsubq_s16(vreinterpretq_s16_u16(upper_c2fi), vreinterpretq_s16_u16(upper_a2dg));
            int16x8_t lower_Gy = vsubq_s16(vreinterpretq_s16_u16(lower_g2hi), vreinterpretq_s16_u16(lower_a2bc));
            int16x8_t upper_Gy = vsubq_s16(vreinterpretq_s16_u16(upper_g2hi), vreinterpretq_s16_u16(upper_a2bc));

            // getting the absolute gradients
            int16x8_t abs_lower_Gx = vabsq_s16(lower_Gx);
            int16x8_t abs_upper_Gx = vabsq_s16(upper_Gx);
            int16x8_t abs_lower_Gy = vabsq_s16(lower_Gy);
            int16x8_t abs_upper_Gy = vabsq_s16(upper_Gy);

            // need to get magnitude 
            int16x8_t mag_lower = vaddq_s16(abs_lower_Gx, abs_lower_Gy);
            int16x8_t mag_upper = vaddq_s16(abs_upper_Gx, abs_upper_Gy);

            // scale by 1/8th
            uint8x8_t out_lower = vqmovun_s16(mag_lower);
            uint8x8_t out_upper = vqmovun_s16(mag_upper);

            // combine lower and upper
            uint8x16_t final_output = vcombine_u8(out_lower, out_upper);

            // storing
            vst1q_u8(temp_dst + col, final_output);
        }

        for (; col < gray.cols -1; ++col){

            // gradient math for Gx and Gy
            int Gx = (prev[col+1] + 2*curr[col+1] + next[col+1]) - (prev[col-1] + 2*curr[col-1] + next[col-1]);
            int Gy = (next[col-1] + 2*next[col] + next[col+1]) - (prev[col-1] + 2*prev[col] + prev[col+1]);
            
            // add four for the nearest integer
            int G = std::abs(Gx) + std::abs(Gy);

            // clamping
            if (G > 255) G = 255;

            // writing to the temp variable
            temp_dst[col] = static_cast<uint8_t>(G);
        }
    }

    // returning new frames w sobel filter
    return dest;
}
