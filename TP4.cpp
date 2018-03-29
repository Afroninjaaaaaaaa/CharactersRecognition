#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <string>
#include <dirent.h>
#include <sys/stat.h>

#define IMGDIMENSIONS   1000
#define TRAINDIMENSIONS 10
#define SIZEFACTOR      10

using namespace cv;
using namespace std;

string windowName = "TP4";
string windowName2 = "_TP4";

void readFilenames(std::vector<std::string> &filenames, const std::string &directory)
{
    DIR *dir;
    class dirent *ent;
    class stat st;
    dir = opendir(directory.c_str());
    while ((ent = readdir(dir)) != NULL) {
        const std::string file_name = ent->d_name;
        const std::string full_file_name = directory + "/" + file_name;
       std::cout<<full_file_name<<std::endl;
        if (file_name[0] == '.')
            continue;
        if (stat(full_file_name.c_str(), &st) == -1)
            continue;
        const bool is_directory = (st.st_mode & S_IFDIR) != 0;
        if (is_directory)
            continue;
        filenames.push_back(full_file_name); // returns just filename
    }
    closedir(dir);
}

void dilatation(Mat &image, int size, int elementType)
{
    Mat element = getStructuringElement(elementType, Size(2*size-1, 2*size-1),
                                           Point(size, size));
    dilate(image, image, element);
}

void erosion(Mat &image, int size, int elementType)
{
    Mat element = getStructuringElement(elementType, Size(2*size-1, 2*size-1),
                                           Point(size, size));
    erode(image, image, element);
}

void prepareImage(Mat &image, string filename)
{
    image = imread(filename);
    if (image.data == NULL)
    {
        cout <<  "Image not found." << endl ;
		return;
    }
}

void processImage(Mat &image, string filename)
{
    prepareImage(image, filename);
    cvtColor(image, image, CV_BGR2GRAY);
    threshold(image, image, 140, 255, 1);
    resize(image, image, Size(TRAINDIMENSIONS, TRAINDIMENSIONS));
    cv::imshow("sample", image);
                cv::waitKey(100);
                
    image = image.reshape(1,1);
}

void imgFindNearest(string filename, Ptr<ml::KNearest> knn)
{
    Mat image;
    prepareImage(image, filename);
    Mat resized = image.clone();
    cvtColor(resized, resized, CV_BGR2GRAY);
    threshold(resized, resized, 140, 255, 1);
    dilatation(resized, 4, 2);
    erosion(resized, 4, 2);

    //namedWindow(windowName);
    //imshow(windowName, image);
    
    Mat out;
    float res;

    
    vector<Rect> rects;
    Mat clone, img;

    resized.convertTo(resized, CV_32F, 1/255.0);
    clone = resized.clone();

    for (int i = 0; i < clone.rows; ++i)
    {
        for (int j = 0; j < clone.cols; ++j)
        {
            if (clone.at<float>(i, j) == 1.0)
            {
                Rect rect;
                floodFill(clone, Point(j, i), Scalar(0, 0, 0), &rect);
                rects.push_back(rect);
                img = resized(rect);
                
                resize(img, img, Size(TRAINDIMENSIONS, TRAINDIMENSIONS));
                cv::imshow("sample", img);
                cv::waitKey(10);
                img = img.reshape(1,1);
                Mat result;
                float res = knn->findNearest(img, 3, result);
                rectangle(image, rect, Scalar(0, 0, 255));
                std::ostringstream stream;
                stream << res;
                string label(stream.str());
                putText(image, label, Point(j, i), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));
               // cout << result << endl;
                

            } 
        }
    }

    imshow("TEST", image);



    /*for (int i = 0; i < rects.size(); ++i)
    {
        rectangle(resized, rects[i], Scalar(255, 255, 255));
    }

    // TODO: FIX
    for (int i = 0; i < rects.size(); ++i)
    {
        img = resized(rects[i]);
        img.convertTo(img, CV_32FC1, 1/255.0);
        resize(img, img, Size(TRAINDIMENSIONS, TRAINDIMENSIONS));
        img = img.reshape(1,1);
        
        res = knn->findNearest(img, 10, out);
        cout << res << endl;
    }*/
}

void addTrainData(Mat &trainData, Mat image, int index)
{
    image.convertTo(image, CV_32F, 1/255.0);
    for (int i = 0; i < image.rows; ++i)
    {
        for (int j = 0; j < image.cols; ++j)
        {
            trainData.at<float>(index, (i * image.cols) + j) = image.at<float>(i, j);
        }
    }
}

int main(int argc, const char** argv)
{
    Mat image, floatData;
    Mat classes((SIZEFACTOR+1)*(SIZEFACTOR+1), 1, CV_32S);
    Mat trainData((SIZEFACTOR+1)*(SIZEFACTOR+1), TRAINDIMENSIONS*TRAINDIMENSIONS, CV_32FC1);
    string rootDir, currDir;
    vector<string> filenames;
    Ptr<ml::KNearest> knn(ml::KNearest::create());
    
    if (argc < 2)
    {
        cout <<  "Base dir name missing (directory containing images)." << endl;
		return -1;
    }
	rootDir = argv[1];
    for (int i = 0; i <= SIZEFACTOR; ++i)
	{
        if (i == SIZEFACTOR)
            currDir = (rootDir + "/X");
		else
            currDir = (rootDir + "/" + to_string(i));
		filenames.clear();
	    readFilenames(filenames, currDir);
        for (int j = 0; j < SIZEFACTOR; ++j)
		{
            processImage(image, filenames[j]);
            addTrainData(trainData, image, (i * SIZEFACTOR) + j);
            classes.at<int>((i * SIZEFACTOR) + j, 0) = i;
            cout << classes.at<int>((i * SIZEFACTOR) + j, 0) << endl;
		}
	}
    knn->train(trainData, ml::ROW_SAMPLE, classes);
    imgFindNearest("./set/test.jpg", knn);


	waitKey(0);
	return 0;
}