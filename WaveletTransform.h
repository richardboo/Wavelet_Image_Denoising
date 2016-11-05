#ifndef WAVELETTRANSFORM_H_INCLUDED
#define WAVELETTRANSFORM_H_INCLUDED
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include<iostream>
#include<math.h>
#include<conio.h>
  using namespace std;
  using namespace cv;

  class wavelettransform
            {
            public:
                Mat im,im1,im2,im3,im4,im5,im6,temp,im11,im12,im13,im14,imi,imd,imr, imdisplay;
                vector<Mat> imdecomp;
                float a,b,c,d;
                vector<Mat> wt(Mat Im);
                Mat inv_wt(Mat, Mat);
                void display_transform(Mat);
                void display_invtransform(Mat);
            };


#endif // WAVELETTRANSFORM_H_INCLUDED
