#include "Plot.h"

CPlot::CPlot(int figNum, int width, int height)
{
	this->m_figSize.width = width;
	this->m_figSize.height = height;
	// finally create a displayMat and put the values the graph
	this->m_display = cv::Mat::ones(m_figSize, CV_8UC3);
	m_display.setTo(C_WHITE);
	this->m_figNum = figNum;
}


CPlot::~CPlot(void)
{

}

// plots 2D plot to display the output
// xMat and yMat contains the data for corresponding axis.
// both should have the same size and the data should be arranged in corresponding columns;
void CPlot::plot(cv::Mat xMat, cv::Mat yMat, cv::Vec3b color)
{


	// get the mapping of x and y onto the width and height of the figure, such that everything fits within the figure;
	cv::Mat xMin, xMax, yMin, yMax;
	cv::reduce(xMat, xMax, 1 /*means reduced to single column*/ , CV_REDUCE_MAX);
	cv::reduce(xMat, xMin, 1 /*means reduced to single column*/ , CV_REDUCE_MIN);

	cv::reduce(yMat, yMax, 1 /*means reduced to single column*/ , CV_REDUCE_MAX);
	cv::reduce(yMat, yMin, 1 /*means reduced to single column*/ , CV_REDUCE_MIN);

	// get the range for both x and y
	double xRange = xMax.at<double>(0,0) - xMin.at<double>(0,0);
	double yRange = yMax.at<double>(0,0) - yMin.at<double>(0,0) ;

	double xPadding = xRange/10, yPadding = yRange/10;

	xRange = xRange + 2*xPadding;
	yRange = yRange + 2*yPadding;


	// define relation between the index map and the values
	double xStep = (m_figSize.width-1) / xRange;
	double yStep = (m_figSize.height-1) / yRange;



	xMin.at<double>(0,0) = xMin.at<double>(0,0) - xPadding;
	yMin.at<double>(0,0) = yMin.at<double>(0,0) - yPadding;
	// displaying some stats for debugging and troubleshooting
	std::cout << "xMin: " << xMin << " xMax: " << xMax << std::endl;
	std::cout << std::endl; 

	std::cout << "yMin: " << yMin << " yMax: " << yMax << std::endl;
	std::cout << std::endl;

	std::cout << "xRange: " << xRange << std::endl;
	std::cout << "yRamge: " << yRange << std::endl;

	std::cout << "xStep: " << xStep << std::endl;
	std::cout << "yStep: " << yStep << std::endl;



	// draw the origin
	for(int i = 0; i < m_display.cols; i++)
	{
		int iIdx, jIdx;
		//iIdx = (int)(xStep * (i - xMin.at<double>(0,0)));
		jIdx = (int)(yStep * (0 - yMin.at<double>(0,0)));
		m_display.at<cv::Vec3b>(jIdx,i) = C_BLACK;
	}

	for(int j = 0; j < m_display.rows; j++)
	{
		int iIdx, jIdx;
		iIdx = (int)(xStep * (1 - xMin.at<double>(0,0)));
		//jIdx = (int)(yStep * (j - yMin.at<double>(0,0)));
		if(iIdx < m_display.cols && iIdx > 0)
			m_display.at<cv::Vec3b>(j,iIdx) = C_BLACK;
	}


	for(int i = 0; i < xMat.cols; i++)
	{
		int iIdx, jIdx;

		iIdx = (int)(xStep * (xMat.at<double>(0, i) - xMin.at<double>(0,0)));
		jIdx = (int)(yStep * (yMat.at<double>(0, i) - yMin.at<double>(0,0)));

		//std::cout << "iIdx: " << iIdx << " jIdx: " << jIdx << std::endl;
		//std::cout << "x Val: " << xMat.at<double>(0,i) << " y Val: " << yMat.at<double>(0,i) << std::endl;

		cv::circle(m_display, cv::Point(iIdx, jIdx), 3, cv::Scalar(color[0],color[1],color[2]));
		m_display.at<cv::Vec3b>(jIdx,iIdx) = color;


	}
	cv::Mat displayMat;
	cv::flip(m_display, displayMat, 0);
	char buffer[50];
	sprintf(buffer, "Figure %d", m_figNum);
	cv::imshow(buffer, displayMat);
	cv::waitKey(1);
}

void CPlot::plot(cv::Mat xMat, cv::Mat yMat, float xMin, float xMax, float yMin, float yMax, cv::Vec3b color)
{
	//// get the mapping of x and y onto the width and height of the figure, such that everything fits within the figure;
	//cv::Mat xMin, xMax, yMin, yMax;
	//cv::reduce(xMat, xMax, 1 /*means reduced to single column*/ , CV_REDUCE_MAX);
	//cv::reduce(xMat, xMin, 1 /*means reduced to single column*/ , CV_REDUCE_MIN);

	//cv::reduce(yMat, yMax, 1 /*means reduced to single column*/ , CV_REDUCE_MAX);
	//cv::reduce(yMat, yMin, 1 /*means reduced to single column*/ , CV_REDUCE_MIN);

	// get the range for both x and y
	double xRange = xMax - xMin;
	double yRange = yMax - yMin ;

	double xPadding = xRange/10, yPadding = yRange/10;

	xRange = xRange + 2*xPadding;
	yRange = yRange + 2*yPadding;


	// define relation between the index map and the values
	double xStep = (m_figSize.width-1) / xRange;
	double yStep = (m_figSize.height-1) / yRange;



	xMin = xMin - xPadding;
	yMin = yMin - yPadding;
	// displaying some stats for debugging and troubleshooting
	std::cout << "xMin: " << xMin << " xMax: " << xMax << std::endl;
	std::cout << std::endl; 

	std::cout << "yMin: " << yMin << " yMax: " << yMax << std::endl;
	std::cout << std::endl;

	std::cout << "xRange: " << xRange << std::endl;
	std::cout << "yRamge: " << yRange << std::endl;

	std::cout << "xStep: " << xStep << std::endl;
	std::cout << "yStep: " << yStep << std::endl;



	// draw the origin
	//for(int i = 0; i < m_display.cols; i++)
	//{
	//	int iIdx, jIdx;
	//	//iIdx = (int)(xStep * (i - xMin.at<float>(0,0)));
	//	jIdx = (int)(yStep * (0 - yMin));
	//	m_display.at<cv::Vec3b>(jIdx,i) = C_BLACK;
	//}

	//for(int j = 0; j < m_display.rows; j++)
	//{
	//	int iIdx, jIdx;
	//	iIdx = (int)(xStep * (1 - xMin));
	//	//jIdx = (int)(yStep * (j - yMin.at<float>(0,0)));
	//	if(iIdx < m_display.cols && iIdx > 0)
	//		m_display.at<cv::Vec3b>(j,iIdx) = C_BLACK;
	//}


	for(int i = 0; i < xMat.cols; i++)
	{
		if(xMat.at<double>(0, i) > xMin  && xMat.at<double>(0,i) < xMax  && yMat.at<double>(0, i) > yMin  && yMat.at<double>(0,i) < yMax)
		{
			int iIdx, jIdx;

			iIdx = (int)(xStep * (xMat.at<double>(0, i) - xMin));
			jIdx = (int)(yStep * (yMat.at<double>(0, i) - yMin));

			//std::cout << "iIdx: " << iIdx << " jIdx: " << jIdx << std::endl;
			//std::cout << "x Val: " << xMat.at<double>(0,i) << " y Val: " << yMat.at<double>(0,i) << std::endl;

			cv::circle(m_display, cv::Point(iIdx, jIdx), 3, cv::Scalar(color[0],color[1],color[2]));
			m_display.at<cv::Vec3b>(jIdx,iIdx) = color;

		}
	}
	cv::Mat displayMat;
	cv::flip(m_display, displayMat, 0);
	char buffer[50];
	sprintf(buffer, "Figure %d", m_figNum);
	cv::imshow(buffer, displayMat);
	cv::waitKey(1);
}

void CPlot::plot(float xMat, float yMat,float xMin, float xMax, float yMin, float yMax, cv::Vec3b color)
{
	//// get the mapping of x and y onto the width and height of the figure, such that everything fits within the figure;
	//cv::Mat xMin, xMax, yMin, yMax;
	//cv::reduce(xMat, xMax, 1 /*means reduced to single column*/ , CV_REDUCE_MAX);
	//cv::reduce(xMat, xMin, 1 /*means reduced to single column*/ , CV_REDUCE_MIN);

	//cv::reduce(yMat, yMax, 1 /*means reduced to single column*/ , CV_REDUCE_MAX);
	//cv::reduce(yMat, yMin, 1 /*means reduced to single column*/ , CV_REDUCE_MIN);

	// get the range for both x and y
	double xRange = xMax - xMin;
	double yRange = yMax - yMin ;

	double xPadding = xRange/10, yPadding = yRange/10;

	xRange = xRange + 2*xPadding;
	yRange = yRange + 2*yPadding;


	// define relation between the index map and the values
	double xStep = (m_figSize.width-1) / xRange;
	double yStep = (m_figSize.height-1) / yRange;



	xMin = xMin - xPadding;
	yMin = yMin - yPadding;
	// displaying some stats for debugging and troubleshooting
	std::cout << "xMin: " << xMin << " xMax: " << xMax << std::endl;
	std::cout << std::endl; 

	std::cout << "yMin: " << yMin << " yMax: " << yMax << std::endl;
	std::cout << std::endl;

	std::cout << "xRange: " << xRange << std::endl;
	std::cout << "yRamge: " << yRange << std::endl;

	std::cout << "xStep: " << xStep << std::endl;
	std::cout << "yStep: " << yStep << std::endl;



	// draw the origin
	for(int i = 0; i < m_display.cols; i++)
	{
		int iIdx, jIdx;
		//iIdx = (int)(xStep * (i - xMin.at<float>(0,0)));
		jIdx = (int)(yStep * (0 - yMin));
		m_display.at<cv::Vec3b>(jIdx,i) = C_BLACK;
	}

	for(int j = 0; j < m_display.rows; j++)
	{
		int iIdx, jIdx;
		iIdx = (int)(xStep * (1 - xMin));
		//jIdx = (int)(yStep * (j - yMin.at<float>(0,0)));
		if(iIdx < m_display.cols && iIdx > 0)
			m_display.at<cv::Vec3b>(j,iIdx) = C_BLACK;
	}
	////////////////////

	if(xMat > xMin  && xMat < xMax  && yMat > yMin  && yMat < yMax)
	{
		int iIdx, jIdx;

		iIdx = (int)(xStep * (xMat - xMin));
		jIdx = (int)(yStep * (yMat - yMin));

		//std::cout << "iIdx: " << iIdx << " jIdx: " << jIdx << std::endl;
		//std::cout << "x Val: " << xMat.at<float>(0,i) << " y Val: " << yMat.at<float>(0,i) << std::endl;

		cv::circle(m_display, cv::Point(iIdx, jIdx), 3, cv::Scalar(color[0],color[1],color[2]));
		m_display.at<cv::Vec3b>(jIdx,iIdx) = color;

	}
	cv::Mat displayMat;
	cv::flip(m_display, displayMat, 0);
	char buffer[50];
	sprintf(buffer, "Figure %d", m_figNum);
	cv::imshow(buffer, displayMat);
	cv::waitKey(1);
}

void CPlot::figure(int figNum)
{
	m_figNum = figNum;
}

void CPlot::clear()
{

	m_display.setTo(C_WHITE);

}
