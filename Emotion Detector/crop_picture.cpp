//
// Created by Yiqi Zhong on 3/14/17.
//

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <string.h>
#include <fstream>

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;
int main (int argc,char** argv)
{
    string filenames=argv[1];
    string files;
    fstream fin(filenames.c_str());
    while(getline(fin,files)){
        Mat input=imread(files);
        if(input.empty()){
            cout<<"Can't decode the image: "<<files<<endl;
            continue;
        }
        Mat output;
        resize(input,output,Size(224,224),0,0,INTER_LINEAR);
        imwrite(files, output);
    }

}