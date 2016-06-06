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

void CPlot::drawErrorEllipse(cv::Mat mean, cv::Mat covmat, cv::Vec3b inColor)
{
	
	//Get the eigenvalues and eigenvectors
	cv::Mat eigenvalues, eigenvectors;
	cv::eigen(covmat, true, eigenvalues, eigenvectors);

	//Calculate the angle between the largest eigenvector and the x-axis
	double angle = atan2(eigenvectors.at<double>(0,1), eigenvectors.at<double>(0,0));

	//Shift the angle to the [0, 2pi] interval instead of [-pi, pi]
	if(angle < 0)
		angle += 6.28318530718;

	//Conver to degrees instead of radians
	angle = 180*angle/3.14159265359;

	//Calculate the size of the minor and major axes
	double halfmajoraxissize=(m_xStep*(sqrt(eigenvalues.at<double>(0))-m_xMin));
	double halfminoraxissize=(m_yStep*(sqrt(eigenvalues.at<double>(1))-m_yMin));

	cv::Point2f pts;
	pts.x = (m_xStep * (mean.at<double>(0, 0) - m_xMin));
	pts.y = (m_yStep * (mean.at<double>(1, 0) - m_yMin));
	

	std::cout << "X Axis Size: " << halfmajoraxissize << std::endl;
	std::cout << "Y Axis Size: " << halfminoraxissize << std::endl;
	std::cout << "Mean X Size: " << pts.x << std::endl;
	std::cout << "Mean Y Size: " << pts.y << std::endl;
	std::cout << "ANgles: " << angle <<std::endl;

	//jIdx = (int)((yMat - yMin));
	//cv::RotatedRect(
	//Return the oriented ellipse
	//The -angle is used because OpenCV defines the angle clockwise instead of anti-clockwise
	cv::RotatedRect rectDraw =  cv::RotatedRect(pts, cv::Size2f(halfmajoraxissize, halfminoraxissize), -angle);
	//cv::ellipse(m_display, rectDraw, inColor, 2);
	cv::ellipse(m_display, rectDraw, cv::Scalar(inColor.val[0], inColor.val[1], inColor.val[2]), 2);
	

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

	m_xMin = xMin.at<double>(0, 0);
	m_xMax = xMax.at<double>(0, 0);
	m_yMin = yMin.at<double>(0, 0);
	m_yMax = yMax.at<double>(0, 0);

	// get the range for both x and y
	m_xRange = m_xMax - m_xMin;
	m_yRange = m_yMax - m_yMin;

	double xPadding = m_xRange/10, yPadding = m_yRange/10;

	m_xRange = m_xRange + 2*xPadding;
	m_yRange = m_yRange + 2*yPadding;


	// define relation between the index map and the values
	 m_xStep = (m_figSize.width-1) / m_xRange;
	 m_yStep = (m_figSize.height-1) / m_yRange;



	m_xMin = m_xMin - xPadding;
	m_yMin = m_yMin - yPadding;

	
	// displaying some stats for debugging and troubleshooting
	//std::cout << "xMin: " << m_xMin << " xMax: " << m_xMax << std::endl;
	//std::cout << std::endl; 

	//std::cout << "yMin: " << m_yMin << " yMax: " << m_yMax << std::endl;
	//std::cout << std::endl;

	//std::cout << "xRange: " << m_xRange << std::endl;
	//std::cout << "yRamge: " << m_yRange << std::endl;

	//std::cout << "m_xStep: " << m_xStep << std::endl;
	//std::cout << "m_yStep: " << m_yStep << std::endl;



	// draw the origin
	for(int i = 0; i < m_display.cols; i++)
	{
		int iIdx, jIdx;
		//iIdx = (int)(m_xStep * (i - xMin.at<double>(0,0)));
		jIdx = (int)(m_yStep * (0 - m_yMin));
		m_display.at<cv::Vec3b>(jIdx,i) = C_BLACK;
	}

	for(int j = 0; j < m_display.rows; j++)
	{
		int iIdx, jIdx;
		iIdx = (int)(m_xStep * (1 - m_xMin));
		//jIdx = (int)(m_yStep * (j - yMin.at<double>(0,0)));
		if(iIdx < m_display.cols && iIdx > 0)
			m_display.at<cv::Vec3b>(j,iIdx) = C_BLACK;
	}


	for(int i = 0; i < xMat.cols; i++)
	{
		int iIdx, jIdx;

		iIdx = (int)(m_xStep * (xMat.at<double>(0, i) - m_xMin));
		jIdx = (int)(m_yStep * (yMat.at<double>(0, i) - m_yMin));

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
	m_xRange = xMax - xMin;
	m_yRange = yMax - yMin;


	double xPadding = m_xRange/10, yPadding = m_yRange/10;

	m_xRange = m_xRange + 2*xPadding;
	m_yRange = m_yRange + 2*yPadding;


	// define relation between the index map and the values
	 m_xStep = (m_figSize.width-1) / m_xRange;
	 m_yStep = (m_figSize.height-1) / m_yRange;



	m_xMin = xMin - xPadding;
	m_yMin = yMin - yPadding;
	m_xMax = xMax;
	m_yMax = yMax;
	// displaying some stats for debugging and troubleshooting
	//std::cout << "xMin: " << m_xMin << " xMax: " << m_xMax << std::endl;
	//std::cout << std::endl; 

	//std::cout << "yMin: " << m_yMin << " yMax: " << m_yMax << std::endl;
	//std::cout << std::endl;

	//std::cout << "xRange: " << m_xRange << std::endl;
	//std::cout << "yRamge: " << m_yRange << std::endl;

	//std::cout << "m_xStep: " << m_xStep << std::endl;
	//std::cout << "m_yStep: " << m_yStep << std::endl;



	// draw the origin
	//for(int i = 0; i < m_display.cols; i++)
	//{
	//	int iIdx, jIdx;
	//	//iIdx = (int)(m_xStep * (i - xMin.at<float>(0,0)));
	//	jIdx = (int)(m_yStep * (0 - yMin));
	//	m_display.at<cv::Vec3b>(jIdx,i) = C_BLACK;
	//}

	//for(int j = 0; j < m_display.rows; j++)
	//{
	//	int iIdx, jIdx;
	//	iIdx = (int)(m_xStep * (1 - xMin));
	//	//jIdx = (int)(m_yStep * (j - yMin.at<float>(0,0)));
	//	if(iIdx < m_display.cols && iIdx > 0)
	//		m_display.at<cv::Vec3b>(j,iIdx) = C_BLACK;
	//}


	for(int i = 0; i < xMat.cols; i++)
	{
		if(xMat.at<double>(0, i) > m_xMin  && xMat.at<double>(0,i) < m_xMax  && yMat.at<double>(0, i) > m_yMin  && yMat.at<double>(0,i) < m_yMax)
		{
			int iIdx, jIdx;

			iIdx = (int)(m_xStep * (xMat.at<double>(0, i) - m_xMin));
			jIdx = (int)(m_yStep * (yMat.at<double>(0, i) - m_yMin));

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
	m_xRange = xMax - xMin;
	m_yRange = yMax - yMin ;

	double xPadding = m_xRange/10, yPadding = m_yRange/10;

	m_xRange = m_xRange + 2*xPadding;
	m_yRange = m_yRange + 2*yPadding;


	// define relation between the index map and the values
	 m_xStep = (m_figSize.width-1) / m_xRange;
	 m_yStep = (m_figSize.height-1) / m_yRange;



	m_xMin = xMin - xPadding;
	m_yMin = yMin - yPadding;

	m_xMax = xMax;
	m_yMax = yMax;
	// displaying some stats for debugging and troubleshooting
	//std::cout << "xMin: " << m_xMin << " xMax: " << m_xMax << std::endl;
	//std::cout << std::endl; 

	//std::cout << "yMin: " << m_yMin << " yMax: " << m_yMax << std::endl;
	//std::cout << std::endl;

	//std::cout << "xRange: " << m_xRange << std::endl;
	//std::cout << "yRamge: " << m_yRange << std::endl;

	//std::cout << "m_xStep: " << m_xStep << std::endl;
	//std::cout << "m_yStep: " << m_yStep << std::endl;



	//// draw the origin
	//for(int i = 0; i < m_display.cols; i++)
	//{
	//	int iIdx, jIdx;
	//	//iIdx = (int)(m_xStep * (i - xMin.at<float>(0,0)));
	//	jIdx = (int)(m_yStep * (0 - yMin));
	//	m_display.at<cv::Vec3b>(jIdx,i) = C_BLACK;
	//}

	//for(int j = 0; j < m_display.rows; j++)
	//{
	//	int iIdx, jIdx;
	//	iIdx = (int)(m_xStep * (1 - xMin));
	//	//jIdx = (int)(m_yStep * (j - yMin.at<float>(0,0)));
	//	if(iIdx < m_display.cols && iIdx > 0)
	//		m_display.at<cv::Vec3b>(j,iIdx) = C_BLACK;
	//}
	////////////////////

	if(xMat > xMin  && xMat < xMax  && yMat > yMin  && yMat < yMax)
	{
		int iIdx, jIdx;

		iIdx = (int)(m_xStep * (xMat - xMin));
		jIdx = (int)(m_yStep * (yMat - yMin));

		//std::cout << "iIdx: " << iIdx << " jIdx: " << jIdx << std::endl;
		//std::cout << "x Val: " << xMat.at<float>(0,i) << " y Val: " << yMat.at<float>(0,i) << std::endl;

		cv::circle(m_display, cv::Point(iIdx, jIdx), 3, cv::Scalar(color[0],color[1],color[2]));
		m_display.at<cv::Vec3b>(jIdx,iIdx) = color;

	}
	//cv::Mat displayMat;
	//cv::flip(m_display, displayMat, 0);
	//char buffer[50];
	//sprintf(buffer, "Figure %d", m_figNum);
	//cv::imshow(buffer, displayMat);
	//cv::waitKey(1);
}

void CPlot::drawNow()
{
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
