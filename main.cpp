#include <iostream>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "WaveletTransform.h"
#include "ImageDenoise.h"


using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    Mat Anoisy = imread(argv[1],1);
    namedWindow("Noisy Image",1);
    imshow("Noisy Image", Anoisy);
    waitKey(0);
    vector<Mat> B1, B2, B3;
    Mat Adenoise;
    if(Anoisy.empty())
    {
        cout<<"Error in reading image"<<endl;
        return -1;
    }

    vector<Mat> Asplit;
    split(Anoisy,Asplit);
// ************************ Wavelet Transform ******************************************************//
    wavelettransform W1, W2, W3;
    B1 = W1.wt(Asplit[0]);               // wavelet coefficients of channel 1 (Blue)
    //W1.display_transform(B1[0]);
    B2 = W2.wt(Asplit[1]);               // wavelet coefficients of channel 2 (Green)
    //W2.display_transform(B2[0]);
    B3 = W3.wt(Asplit[2]);               // wavelet coefficients of channel 3 (Red)
    //W3.display_transform(B3[0]);
// *************************** De-noising in wavelet domain ************************************//
    vector<Mat> Bdenoise;
    float noise_mean = 150;
    Bdenoise.push_back(imdenoise(B1[0],noise_mean));
    Bdenoise.push_back(imdenoise(B2[0],noise_mean));
    Bdenoise.push_back(imdenoise(B3[0],noise_mean));

// ************************** Reconstruct De-noised Image (Inverse Wavelet Transform) ********************************//
    Mat A1, A2, A3;
    A1 = W1.inv_wt(Anoisy,Bdenoise[0]);
    //W1.display_invtransform(A1);
    A2 = W2.inv_wt(Anoisy, Bdenoise[1]);
    //W2.display_invtransform(A2);
    A3 = W3.inv_wt(Anoisy, Bdenoise[2]);
    //W3.display_invtransform(A3);
    Mat g = Mat::zeros(Anoisy.rows, Anoisy.cols, CV_8UC1);
    vector<Mat> channels;
    channels.push_back(A1);
    channels.push_back(A2);
    channels.push_back(A3);

    merge(channels, Adenoise);
    imwrite("De-noised_Image.jpg", Adenoise);
    namedWindow("De-noised Image",1);
    imshow("De-noised Image",Adenoise);
    waitKey(0);
    return 0;
}
