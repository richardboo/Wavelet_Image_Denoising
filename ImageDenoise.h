#ifndef IMAGEDENOISE_H_INCLUDED
#define IMAGEDENOISE_H_INCLUDED

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>

Mat imdenoise (Mat B,float noise_mean)
{
    int r = B.rows;
    int c = B.cols;

    float temp;
    Mat Variance = Mat::zeros(r,c,CV_32F);
    Mat Bdenoise = Mat::zeros(r,c,CV_32F);
//***************** ML Estimate of Variance ********************************************//
     for (int i = 2; i < r-1; i++)
        {
            for (int j=2; j< c-1; j++)
            {
                temp = pow(B.at<float>(i-1,j-1),2) + pow(B.at<float>(i-1,j),2) + pow(B.at<float>(i-1,j+1),2) +
                        pow(B.at<float>(i,j-1),2) + pow(B.at<float>(i,j),2) + pow(B.at<float>(i,j+1),2) +
                        pow(B.at<float>(i+1,j-1),2) + pow(B.at<float>(i+1,j),2) + pow(B.at<float>(i+1,j+1),2);
                Variance.at<float>(i,j) = (temp/9) - noise_mean;
                 if (Variance.at<float>(i,j) < 0 )
                 {
                     Variance.at<float>(i,j) = 0;
                 }
            }

        }
//************************** MMSE Estimate of Wavelet Coefficients ***********************************//
    for (int i = 2; i < r-1; i++)
    {
        for (int j = 2; j < c-1; j++)
        {

            Bdenoise.at<float>(i,j) = (Variance.at<float>(i,j) / (Variance.at<float>(i,j) + noise_mean) ) * B.at<float>(i,j);
        }
    }

    return Bdenoise;
}

#endif // IMAGEDENOISE_H_INCLUDED
