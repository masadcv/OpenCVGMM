#pragma once
#include "common.h"

#define C_RED cv::Vec3b(0, 0, 255)
#define C_GREEN cv::Vec3b(0, 255, 0)
#define C_BLUE cv::Vec3b(255, 0, 0)
#define C_WHITE cv::Vec3b(255, 255, 255)
#define C_BLACK cv::Vec3b(0, 0, 0)

class CPlot
{

	cv::Size m_figSize;
	cv::Mat m_xMat, m_yMat;
	double m_xStep, m_yStep, m_xMin, m_yMin, m_xMax, m_yMax, m_xRange, m_yRange;
	int m_figNum;
public:
	cv::Mat m_display;
	CPlot(int figNum =  0,int width = 500, int height = 500);
	void drawErrorEllipse(cv::Mat mean, cv::Mat covmat, cv::Vec3b inColor);
	void figure(int figNum);
	void plot(cv::Mat xMat, cv::Mat yMat,cv::Vec3b color = C_BLUE);
	void plot(cv::Mat xMat, cv::Mat yMat, float xMin, float xMax, float yMin, float yMax, cv::Vec3b color = C_BLUE);
	void plot(float xMat, float yMat,float xMin, float xMax, float yMin, float yMax, cv::Vec3b color = C_BLUE);
	void drawNow();
	void clear();
	~CPlot(void);
};

