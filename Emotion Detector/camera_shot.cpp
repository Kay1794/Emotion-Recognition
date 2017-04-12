//
// Created by Yiqi Zhong on 3/17/17.
//

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <string.h>
#include <stdio.h>

using namespace std;
using namespace cv;

const string file_prefix="/Users/zhongyiqi/Documents/code/final/Dataset_Name/";
const string file_happy="/Users/zhongyiqi/Documents/code/final/Dataset_tag/Happy/";
const string file_angry="/Users/zhongyiqi/Documents/code/final/Dataset_tag/Angry/";
const string file_sad="/Users/zhongyiqi/Documents/code/final/Dataset_tag/Sad/";
const string file_surprise="/Users/zhongyiqi/Documents/code/final/Dataset_tag/Surprise/";
const string file_neutral="/Users/zhongyiqi/Documents/code/final/Dataset_tag/Neutral/";
const string file_fear="/Users/zhongyiqi/Documents/code/final/Dataset_tag/Fear/";
int main() {
    cout<<"###########################################"<<endl;
    cout<<"# _______________________________________ #"<<endl;
    cout<<"#| Happy Angry Sad Surprise Neutral Fear |#"<<endl;
    cout<<"# --------------------------------------- #"<<endl;
    cout<<"# Press Space to do screenshot            #"<<endl;
    cout<<"# Press 1 if you want to save the image   #"<<endl;
    cout<<"# Press random key to skip the image      #"<<endl;
    cout<<"# Press ESC to end the program            #"<<endl;
    cout<<"# ~~~~~~~~~~~~~~~~~~                      #"<<endl;
    cout<<"# 'HA'for Happy                           #"<<endl;
    cout<<"# 'AN'for Angry                           #"<<endl;
    cout<<"# 'SD'for Sad                             #"<<endl;
    cout<<"# 'SP'for Surprise                        #"<<endl;
    cout<<"# 'NE'for Neutral                         #"<<endl;
    cout<<"# 'FE'for Fear                            #"<<endl;
    cout<<"# ~~~~~~~~~~~~~~~~~~                      #"<<endl;
    cout<<"###########################################"<<endl;

    VideoCapture camera(0);
    camera.set(CV_CAP_PROP_FRAME_WIDTH,512);
    camera.set(CV_CAP_PROP_FRAME_HEIGHT,512);


    cvNamedWindow("Camera",WINDOW_NORMAL);

    cvNamedWindow("Sample",WINDOW_NORMAL);
    //cvResizeWindow("Sample",256,256);
    Mat frame;
    char c;
    int number=0;
    string name;

    cout<<"Please enter your Name(or Nickname :),end with a 'return'):";
    cin>>name;
    while(1){

        camera >> frame;
        imshow("Camera",frame);
        c = cvWaitKey(20);
        if(c==27)
        {
            destroyAllWindows();
            break;
        }
        if(c==32)
        {
            Mat current;
            current=frame;
            imshow("Sample",current);

            char h=cvWaitKey(-1);
            if(h==49)//press '1' to save the image
            {
                string face;
                cout<<"Please enter the initial of the emotion:";
                cin>>face;
                imwrite(file_prefix + name + "_" + face + ".jpg", frame);
                if(face[0]=='H'||face[0]=='h')imwrite(file_happy+name + "_" + face + ".jpg", frame);
                else if(face[0]=='A'||face[0]=='a')imwrite(file_angry+name + "_" + face + ".jpg", frame);
                else if(face[0]=='S'&&face[1]=='D')imwrite(file_sad+name + "_" + face + ".jpg", frame);
                else if(face[0]=='S'&&face[1]=='P')imwrite(file_surprise+name + "_" + face + ".jpg", frame);
                else if(face[0]=='F'||face[0]=='f')imwrite(file_fear+name + "_" + face + ".jpg", frame);
                else if(face[0]=='N'||face[0]=='n')imwrite(file_neutral+name + "_" + face + ".jpg", frame);
                cout<<"Image"<<number<<": "<<face<<" has been successfully saved"<<endl;
                number++;
            }
            else
            {
                destroyWindow("Sample");
                continue;
            }//press random key to continue
        }
    }

    return 0;
}