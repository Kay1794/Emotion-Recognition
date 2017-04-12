//
// Created by Yiqi Zhong on 3/28/17.
//
#include <fstream>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <string>
using namespace std;
using namespace cv;
const string saveFilePath = "/Users/zhongyiqi/Documents/code/final/CKPlus/LabeledImages/";
const string emotion[8]={"NE","AN","CM","DI","FE","HA","SD","SP"};
int main(int argc, char** argv)
{
    string filenames="/Users/zhongyiqi/Documents/code/final/CKPlus/Labels/fileList.txt";
    string files;
    fstream fin(filenames.c_str());
    string fnd1 = "Labels";
    string rep1 = "Images";
    string rep3 = "LabeledImages";
    string fnd2 = "_emotion.txt";
    string rep2 = ".png";
    while(getline(fin,files)) {

        ifstream flabel(files);
        string image_file=files;
        string image_save;
        int label;
        flabel >> label;
        image_file = image_file.replace(image_file.find(fnd1), fnd1.length(), rep1);
        image_file = image_file.replace(image_file.find(fnd2), fnd2.length(), rep2);
        Mat image=imread(image_file);
        image_save=image_file;
        image_save = image_save.erase(61,12);
        image_save = image_save.replace(image_save.find(rep1),rep1.length(),rep3);
        image_save = image_save+emotion[label]+".png";
        imwrite(image_save,image);
        //cout<<files<<endl;



    }
}

