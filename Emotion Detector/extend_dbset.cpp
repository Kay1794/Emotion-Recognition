//
// Created by Yiqi Zhong on 3/17/17.
//

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <fstream>
#include <string>
#include <iostream>

using namespace cv;
using namespace std;
String save_file="/Users/zhongyiqi/Documents/code/final/new_dataset/";
int betas[4]={-20,-10,10,20};
double alphas[4]={1.0,1.0,1.0,1.0};
const int stride_list[3]={10,15,20};
const int N=6;

void ChangeTheBrightness(String filename, double alpha, int beta)
{
    // Enter the alpha value [1.0-3.0]
    //Enter the beta value [0-100]
    Mat image = imread(save_file+filename);
    Mat new_image=Mat::zeros(image.size(), image.type());
    for( int y = 0; y < image.rows; y++ )
    { for( int x = 0; x < image.cols; x++ )
        { for( int c = 0; c < 3; c++ )
            {
                new_image.at<Vec3b>(y,x)[c] =
                        saturate_cast<uchar>( alpha*( image.at<Vec3b>(y,x)[c] ) + beta );
            }
        }
    }
    filename=std::to_string(beta)+"_"+filename;
    resize(new_image,new_image, Size(244,244),0,0,INTER_LINEAR);
    imwrite(save_file+filename,new_image);
}

void ChangeTheCrop(string filename, int direction, int stride)
{
    Mat image = imread(filename);
    Mat new_image;
    string dic;
    if(image.empty())
    {
        std::cout<<"Can't read the data from file:"<<filename<<std::endl;
        exit(0);
    }

    if(direction==0)
    {
        dic="up";
        new_image = Mat::zeros(image.rows-stride,image.cols,image.type());
        for(int y=stride;y<image.rows;y++)
            for(int x = 0;x<image.cols;x++)
                for(int c=0;c<3;c++)
                {
                    new_image.at<Vec3b>(y-stride,x)[c] =
                            saturate_cast<uchar>(image.at<Vec3b>(y,x)[c]);
                }
    }
    if(direction==1)
    {
        dic="down";
        new_image = Mat::zeros(image.rows-stride,image.cols,image.type());
        for(int y=0;y<image.rows-stride;y++)
            for(int x = 0;x<image.cols;x++)
                for(int c=0;c<3;c++)
                {
                    new_image.at<Vec3b>(y,x)[c] =
                            saturate_cast<uchar>(image.at<Vec3b>(y,x)[c]);
                }
    }
    if(direction==2)
    {
        dic="left";
        new_image = Mat::zeros(image.rows,image.cols-stride,image.type());
        for(int y=0;y<image.rows;y++)
            for(int x = stride;x<image.cols;x++)
                for(int c=0;c<3;c++)
                {
                    new_image.at<Vec3b>(y,x-stride)[c] =
                            saturate_cast<uchar>(image.at<Vec3b>(y,x)[c]);
                }
    }
    if(direction==3)
    {
        dic="right";
        new_image = Mat::zeros(image.rows,image.cols-stride,image.type());
        for(int y=0;y<image.rows;y++)
            for(int x = 0;x<image.cols-stride;x++)
                for(int c=0;c<3;c++)
                {
                    new_image.at<Vec3b>(y,x)[c] =
                            saturate_cast<uchar>(image.at<Vec3b>(y,x)[c]);
                }
    }

    filename.erase(0,51);
    imwrite(save_file+dic+std::to_string(stride)+filename,new_image);
    for(int i=0;i<4;i++)
    {
        ChangeTheBrightness(dic+std::to_string(stride)+filename,alphas[i],betas[i]);
    }

}

int main(int argc, char** argv)
{
    string fileList=argv[1];
    fstream fin(fileList.c_str());
    string imgfile;
    //Rate1
//    while(getline(fin,imgfile))
//    {
//        for(int i=0;i<4;i++)
//            {
//
//                ChangeTheCrop(imgfile,i,15);
//            }
//    }
    //Rate2
//    while(getline(fin,imgfile))
//    {
//        for(int i=0;i<4;i++)
//        {
//            for(int j=0;j<2;j++)
//            ChangeTheCrop(imgfile,i,stride_list[j]);
//        }
//    }
    //Rate3
    while(getline(fin,imgfile))
    {
        for(int i=0;i<4;i++)
        {
            for(int j=0;j<3;j++)
            ChangeTheCrop(imgfile,i,stride_list[j]);
        }
    }

    return 0;
}