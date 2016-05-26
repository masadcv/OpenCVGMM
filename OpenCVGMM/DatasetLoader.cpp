#include "DatasetLoader.h"

DatasetLoader::DatasetLoader(void)
{
	

}

// readMatlabFile: Loads a matlab binary file with a Mat
cv::Mat DatasetLoader::readMatlabFile(const char* filename)
{
	cv::Mat outVals;
	std::fstream file;
	file.open(filename, std::ios::in | std::ios::binary);
	if(!file.is_open())
	{
		std::cout << "Error opening matlab binary file" << std::endl;
		std::cout << filename << std::endl;
		exit(1);
	}


	// read the size of the Mat
	double colsMat, rowsMat;

	file.read((char*)&rowsMat, sizeof(rowsMat));
	file.read((char*)&colsMat, sizeof(colsMat));

	outVals = cv::Mat::zeros(rowsMat, colsMat, CV_32FC1);

	for(int i = 0; i < outVals.cols; i++)
	{	
		for(int j = 0; j < outVals.rows; j++)
		{

			double buff;
			file.read((char*)&buff, sizeof(buff));
			//file.read((char*)&outVals.at<double>(j, i), sizeof(outVals.at<double>(j, i)));
			outVals.at<float>(j,i) = (float)buff;
		}
	}
	return outVals;
}


// writeMatlab File : writes OpenCV Mat as binary file, which can be opened in matlab
bool DatasetLoader::writeMatlabFile(const char* filename, const cv::Mat& inVals)
{
	//cv::Mat outVals;
	std::fstream file;
	file.open(filename, std::ios::out | std::ios::binary);
	if(!file.is_open())
	{
		std::cout << "Error opening matlab binary file" << std::endl;
		std::cout << filename << std::endl;
		exit(1);
	}

	// read the size of the Mat
	double colsMat, rowsMat;

	rowsMat = inVals.rows;
	colsMat = inVals.cols;

	file.write((char*)&rowsMat, sizeof(rowsMat));
	file.write((char*)&colsMat, sizeof(colsMat));

	//outVals = cv::Mat::zeros(rowsMat, colsMat, CV_32FC1);

	for(int i = 0; i < inVals.cols; i++)
	{	
		for(int j = 0; j < inVals.rows; j++)
		{

			double buff = inVals.at<float>(j,i);
			file.write((char*)&buff, sizeof(buff));
			//file.read((char*)&outVals.at<double>(j, i), sizeof(outVals.at<double>(j, i)));
			//= (float)buff;
		}
	}
	// all done
	file.close();
	return true;
	//return outVals;
}

DatasetLoader::~DatasetLoader(void)
{

}
