#pragma once
#ifndef DATASETLOADER_H
#define DATASETLOADER_H
#include "common.h"

class DatasetLoader
{
private:

public:

	DatasetLoader(void);
	cv::Mat readMatlabFile(const char* filename);
	bool writeMatlabFile(const char* filename, const cv::Mat& inVals);
	
	~DatasetLoader(void);
};

#endif