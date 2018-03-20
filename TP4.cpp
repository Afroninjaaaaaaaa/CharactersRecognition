#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <string>
#include <dirent.h>
#include <sys/stat.h>

#define IMGDIMENSIONS 100
#define SIZEFACTOR    10
#define DILATSIZE     10
#define EROSIOSIZE    5

using namespace cv;
using namespace std;

string windowName = "TP4";

void readFilenames(std::vector<std::string> &filenames, const std::string &directory)
{
    DIR *dir;
    class dirent *ent;
    class stat st;
    dir = opendir(directory.c_str());
    while ((ent = readdir(dir)) != NULL) {
        const std::string file_name = ent->d_name;
        const std::string full_file_name = directory + "/" + file_name;
        //std::cout<<full_file_name<<std::endl;
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

void dilatation(Mat &image)
{
    Mat element = getStructuringElement(1, Size(2*DILATSIZE-1, 2*DILATSIZE-1),
                                           Point(DILATSIZE, DILATSIZE));
    dilate(image, image, element);
}

void erosion(Mat &image)
{
    Mat element = getStructuringElement(1, Size(2*EROSIOSIZE-1, 2*EROSIOSIZE-1),
                                           Point(EROSIOSIZE, EROSIOSIZE));
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
    resize(image, image, Size(IMGDIMENSIONS, IMGDIMENSIONS));
}

void processImage(Mat &image, string filename)
{
    prepareImage(image, filename);
    inRange(image, Scalar(0, 0, 0), Scalar(120, 120, 120), image);
    dilatation(image);
    erosion(image);
}

void imgFindNearest(string filename, Ptr<ml::KNearest> knn)
{
    Mat image;
    prepareImage(image, filename);
    namedWindow(windowName);
    imshow(windowName, image);
    
    Mat resized(1, image.rows * image.cols, CV_32F);
    Mat out;
    float res;
    for (int i = 0; i < image.rows; ++i)
    {
        for (int j = 0; j < image.cols; ++j)
            resized.at<float>(0, (i * image.cols) + j) = image.at<float>(i, j);
    }
    res = knn->findNearest(resized, 5, out);
    cout << res << endl;
}

void addTrainData(Mat &trainData, Mat image, int index)
{
    for (int i = 0; i < image.rows; ++i)
    {
        for (int j = 0; j < image.cols; ++j)
            trainData.at<float>(index, (i * image.cols) + j) = image.at<float>(i, j);
    }
    
}

int main(int argc, const char** argv)
{
    Mat image, floatData;
    Mat classes(SIZEFACTOR*SIZEFACTOR, 1, CV_32F);
    Mat trainData(SIZEFACTOR*SIZEFACTOR, IMGDIMENSIONS*IMGDIMENSIONS, CV_32F);
    string rootDir, currDir;
    vector<string> filenames;
    Ptr<ml::KNearest> knn(ml::KNearest::create());
    
    
	// Load Images
    if (argc < 2)
    {
        cout <<  "Base dir name missing (directory containing images)." << endl ;
		return -1;
    }
	rootDir = argv[1];
	for (int i = 0; i < SIZEFACTOR; ++i)
	{
		currDir = (rootDir + "/" + to_string(i));
		readFilenames(filenames, currDir);
        for (int j = 0; j < SIZEFACTOR; ++j)
		{
            processImage(image, filenames[j]);
            addTrainData(trainData, image, (i * SIZEFACTOR) + j);
            classes.at<int>((i * SIZEFACTOR) + j, 0) = i;
		}
	}

    knn->train(trainData, ml::ROW_SAMPLE, classes);
    imgFindNearest("./set/0/roi0042.jpg", knn);


	waitKey(0);
	return 0;
}