//
// Created by Yiqi Zhong on 3/30/17.
//
#include <iostream>
#include <fstream>
#include <string>


//Angry
//        Disgust
//Fear
//        Happy
//Neutral
//        Sad
//Surprise

using namespace std;

int main(int argc, char** argv)
{
    string filename=argv[1];
    fstream fin(filename.c_str());
    ofstream fout("/Users/zhongyiqi/Documents/code/final/new_val.txt");
    //ofstream fout("val.txt");
    string imgname;
    int count=0;
    while(getline(fin,imgname))
    {
        cout<<count++<<endl;
        if(imgname.find("AN")<imgname.length())
        {
            fout<<imgname<<" "<<0<<endl;
        }
        else if(imgname.find("DI")<imgname.length())
        {
            fout<<imgname<<" "<<1<<endl;
        }
        else if(imgname.find("FE")<imgname.length())
        {
            fout<<imgname<<" "<<2<<endl;
        }
        else if(imgname.find("HA")<imgname.length())
        {
            fout<<imgname<<" "<<3<<endl;
        }
        else if(imgname.find("NE")<imgname.length())
        {
            fout<<imgname<<" "<<4<<endl;
        }
        else if(imgname.find("SD")<imgname.length())
        {
            fout<<imgname<<" "<<5<<endl;
        }
        else if(imgname.find("SP")<imgname.length())
        {
            fout<<imgname<<" "<<6<<endl;
        }
    }
}

