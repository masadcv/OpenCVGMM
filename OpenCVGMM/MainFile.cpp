// for using constants like pi
#define _USE_MATH_DEFINES

// opencv libs
#include <opencv\cv.h>
#include <opencv\highgui.h>

// standard C++ libs
#include <iostream>
#include <math.h>

void updateGMM(const cv::Mat &xMat, const cv::Mat &assignScore, cv::Mat &piK, cv::Mat &muK, cv::Mat &sigmaK);
double getLogLikelihood(const cv::Mat &xMat, const cv::Mat &piK, const cv::Mat &muK, const cv::Mat &sigmaK);
void fitGMM(const cv::Mat &inData, const int &K, cv::Mat &piK, cv::Mat &muK, cv::Mat &sigmaK);
void getAssignmentScore(const cv::Mat &inData, const cv::Mat &piK, const cv::Mat &muK, const cv::Mat &sigmaK, cv::Mat &assignScore);
void getGMMOfCluster(const cv::Mat &inData, const int &K, const cv::Mat &assignMap, cv::Mat &piK, cv::Mat &muK, cv::Mat &sigmaK);
void getMeanAndCovariance(const cv::Mat &inData, cv::Mat &outMu, cv::Mat &outCov);
void randomInitialize(cv::Mat &inHashmap, int clusterNum);
double getGaussian2D(const cv::Mat &inPoint, const cv::Mat &muC, const cv::Mat &sigmaC);
void getMeanAndCovarianceWeighted(const cv::Mat &inData, const cv::Mat &inWeights, cv::Mat &outPi, cv::Mat &outMu, cv::Mat &outCov);
// dummy data generator
cv::Mat generateData(void);

int main(void)
{
	// random seed
	cv::theRNG().state = cv::getTickCount();

	cv::Mat xMat; // for containing all the observations from multiple gaussians
	xMat = generateData().clone();

	std::cout << xMat << std::endl;


	// we have the data now we need a gaussian mixture model estimator
	// suppose we only know the data limits, and the number of clusters to make

	// piK == weights for each gaussian
	// muK == mean of each gaussian
	// sigmaK == covariance matrix for each gaussian
	cv::Mat piK, muK, sigmaK;  // for returning the output for mixture model clustering

	// two clusters
	int K = 2;

	// function for fitting gaussian mixture model
	fitGMM(xMat, K, piK, muK, sigmaK);



}

// function for fitting GMM model with K clusters
void fitGMM(const cv::Mat &xMat, const int &K, cv::Mat &piK, cv::Mat &muK, cv::Mat &sigmaK)
{
	// intialize the assignment hashmap
	cv::Mat assignMap;
	assignMap = cv::Mat::zeros(1, xMat.cols, CV_64FC1);

	randomInitialize(assignMap, K);

	// temp display
	std::cout << assignMap << std::endl;

	// calculate the mean and variance of the clusters
	getGMMOfCluster(xMat, K, assignMap, piK, muK, sigmaK);

	// display cov and mean
	//std::cout << "curWeight: " << piK << std::endl;
	//std::cout << "curMean: " << muK << std::endl;
	//std::cout << "curCov: " << sigmaK << std::endl;

	// next step is to get the assignment score
	cv::Mat assignScore;
	double maxlogLH = -DBL_MAX;
	for(int i = 0; i < 1000; i++)
	{
		getAssignmentScore(xMat, piK, muK, sigmaK, assignScore);

		updateGMM(xMat, assignScore, piK, muK, sigmaK);
		double logLH = getLogLikelihood(xMat, piK, muK, sigmaK);

		if( (logLH - maxlogLH) > 0.000001 )
			maxlogLH = logLH;
		else
		{
			std::cout << "Log likelihood converged" << std::endl;
			break;
		}
		std::cout << "logLikelihood : " << getLogLikelihood(xMat, piK, muK, sigmaK) << std::endl;

	}

}

double getLogLikelihood(const cv::Mat &xMat, const cv::Mat &piK, const cv::Mat &muK, const cv::Mat &sigmaK)
{
	double retProbability = 0;
	for(int n = 0; n < xMat.cols; n++)
	{
		double innerProbability = 0;
		for(int k = 0; k < piK.cols; k++)
		{
			cv::Rect roi(k * muK.rows, 0, muK.rows, muK.rows);
			innerProbability += piK.at<double>(0, k) * getGaussian2D(xMat.col(n).clone(), muK.col(k).clone(), sigmaK(roi));
		}
		retProbability += std::log(innerProbability);
	}

	return retProbability;
}

void updateGMM(const cv::Mat &xMat, const cv::Mat &assignScore, cv::Mat &piK, cv::Mat &muK, cv::Mat &sigmaK)
{
	// calculate the new mean and covariance
	for(int k = 0; k < piK.cols; k++)
	{
		cv::Rect roi(k * muK.rows, 0, muK.rows, muK.rows);
		cv::Mat curCov = sigmaK(roi);
		cv::Mat curMean = muK.col(k).clone();
		cv::Mat curPi = piK.col(k).clone();
		//assignScore.at<double>(j, i) = piK.at<double>(0, j) * getGaussian2D(inData.col(i).clone(), muK.col(j).clone(), sigmaK(roi));
		std::cout <<std::endl;
		std::cout << "Mean before: " << curMean <<std::endl;
		std::cout << "Pi before: " << curPi <<std::endl;
		getMeanAndCovarianceWeighted(xMat, assignScore.row(k), curPi, curMean, curCov);
		std::cout << "Mean after: " << curMean <<std::endl;
		std::cout << "Pi After: " << curPi <<std::endl;
		


		// assign the Matrices back
		curMean.copyTo(muK.col(k));
		curPi.copyTo(piK.col(k));

		for(int i = 0; i < muK.rows; i++)
		{
			for(int j = 0; j < muK.rows; j++)
			{
				sigmaK.at<double>(j, i + k*muK.rows) = curCov.at<double>(j, i);

			}
		}
	}

}

// function for calculating the covariance and mean
void getMeanAndCovarianceWeighted(const cv::Mat &inData, const cv::Mat &inWeights, cv::Mat &outPi, cv::Mat &outMu, cv::Mat &outCov)
{
	// initialize the returning matrices

	// mean
	outMu = cv::Mat::zeros(inData.rows, 1, CV_64FC1);

	// covariance
	outCov = cv::Mat::zeros(inData.rows, inData.rows, CV_64FC1);
	double sumVal = 0;
	for(int i = 0; i < inData.cols; i++)
	{
		for(int j = 0; j < inData.rows; j++)
		{
			outMu.at<double>(j, 0) += (inWeights.at<double>(0, i) * inData.at<double>(j, i));
		}
		sumVal += inWeights.at<double>(0, i);
	}

	// dividing by the total number to get the average
	outMu = outMu/sumVal;
	//std::cout << std::endl;
	// now calculate the covariance
	for(int i = 0; i < inData.cols; i++)
	{
		for(int j = 0; j < inData.rows; j++)
		{
			for(int j2 = 0; j2 < inData.rows; j2++)
			{
				outCov.at<double>(j, j2) += (inWeights.at<double>(0, i) * (inData.at<double>(j, i) - outMu.at<double>(j, 0)) * (inData.at<double>(j2, i) - outMu.at<double>(j2, 0)));

			}
		}
	}

	//std::cout << std::endl;
	// dividing by the total number to get the covariance
	outCov = outCov/sumVal;

	outPi = cv::Mat::zeros(1, 1, CV_64FC1);
	outPi.at<double>(0, 0) = sumVal/inWeights.cols;

}

void getAssignmentScore(const cv::Mat &inData, const cv::Mat &piK, const cv::Mat &muK, const cv::Mat &sigmaK, cv::Mat &assignScore)
{
	// initialize the assignScore for keeping score
	assignScore = cv::Mat::zeros(piK.cols, inData.cols, CV_64FC1);

	for(int i = 0; i < inData.cols; i++)
	{
		for(int j = 0; j < assignScore.rows; j++)
		{
			cv::Rect roi(j * muK.rows, 0, muK.rows, muK.rows);
			//std::cout << sigmaK(roi) << std::endl;
			assignScore.at<double>(j, i) = piK.at<double>(0, j) * getGaussian2D(inData.col(i).clone(), muK.col(j).clone(), sigmaK(roi));
		}
	}

	// normalize the scores
	for(int i = 0; i < assignScore.cols; i++)
	{
		double curSum = 0;
		for(int j = 0; j < assignScore.rows; j++)
		{
			curSum += assignScore.at<double>(j, i);
		}

		for(int j = 0; j < assignScore.rows; j++)
		{
			assignScore.at<double>(j, i) = assignScore.at<double>(j, i)/curSum;
		}
	}


	// display the score
	//std::cout << assignScore << std::endl;
}

double getGaussian2D(const cv::Mat &inPoint, const cv::Mat &muC, const cv::Mat &sigmaC)
{
	cv::Mat invSigmaC = sigmaC.inv();
	//std::cout << "invSigmaC: " << invSigmaC <<std::endl;
	double detSigmaC = cv::determinant(sigmaC);

	cv::Mat diffPoint = (inPoint - muC);

	//std::cout << "diffPoint: " << diffPoint << std::endl;

	double A = 1/( std::pow(2*M_PI, inPoint.rows/2) * (std::pow(detSigmaC, 0.5)) );

	cv::Mat B = ( diffPoint.t() * invSigmaC * diffPoint );
	double BVal = -B.at<double>(0, 0)/2;
	//std::cout << "B: " << BVal <<std::endl;
	double retProbability;
	retProbability = A * std::exp(BVal);

	return retProbability;
}


void getGMMOfCluster(const cv::Mat &xMat, const int &K, const cv::Mat &assignMap, cv::Mat &piK, cv::Mat &muK, cv::Mat &sigmaK)
{
	// create container for weights, mean and cov
	piK = cv::Mat::zeros(1, K, CV_64FC1);
	muK = cv::Mat::zeros(xMat.rows, K, CV_64FC1);
	sigmaK = cv::Mat::zeros(xMat.rows, xMat.rows * K, CV_64FC1);

	for(int i = 0; i < K; i++)
	{
		int curCount = 0;
		for(int idx = 0; idx < assignMap.cols; idx++)
		{
			if(assignMap.at<double>(0, idx) == i)
			{
				curCount++;
			}
		}

		// make temporary container for saving all the points that are in current cluster
		cv::Mat curPoints;
		curPoints = cv::Mat::zeros(xMat.rows, curCount, CV_64FC1);
		int curIdx = 0;
		for(int idx = 0; idx < assignMap.cols; idx++)
		{
			if(assignMap.at<double>(0, idx) == i)
			{
				xMat.col(idx).copyTo(curPoints.col(curIdx++));
			}
		}

		piK.at<double>(0, i) = curCount;

		//std::cout << "curPoints: " << curPoints << std::endl;
		cv::Mat curCov, curMean;
		curCov = cv::Mat::zeros(xMat.rows, xMat.rows, CV_64FC1);
		curMean = cv::Mat::zeros(xMat.rows, 1, CV_64FC1);

		//cv::calcCovarMatrix(curPoints, curCov, curMean, CV_COVAR_COLS | CV_COVAR_NORMAL | CV_COVAR_SCALE);
		//std::cout << "curCovariance: " << curCov << std::endl;
		//std::cout << "curMean: " << curMean << std::endl;

		getMeanAndCovariance(curPoints, curMean, curCov);
		//std::cout << "curCovariance: " << curCov << std::endl;
		//std::cout << "curMean: " << curMean << std::endl;

		// copy the mean and cov to return mat
		for(int cI = 0; cI < curMean.cols; cI++)
		{
			for(int cJ = 0; cJ < curMean.rows; cJ++)
			{
				muK.at<double>(cJ, i) = curMean.at<double>(cJ, cI);
			}
		}


		for(int cI = 0; cI < curCov.cols; cI++)
		{
			for(int cJ = 0; cJ < curCov.rows; cJ++)
			{
				//std::cout << "Index j : " << cJ  << " Index i : " << cI + curCov.rows*i << std::endl;
				sigmaK.at<double>(cJ, cI + curCov.rows*i) = curCov.at<double>(cJ, cI);
			}
		}
		//std::cout << "global Covariance: " << sigmaK << std::endl;
	}
	piK = piK/xMat.cols;
}

// function for calculating the covariance and mean
void getMeanAndCovariance(const cv::Mat &inData, cv::Mat &outMu, cv::Mat &outCov)
{
	// initialize the returning matrices

	// mean
	outMu = cv::Mat::zeros(inData.rows, 1, CV_64FC1);

	// covariance
	outCov = cv::Mat::zeros(inData.rows, inData.rows, CV_64FC1);

	for(int i = 0; i < inData.cols; i++)
	{
		for(int j = 0; j < inData.rows; j++)
		{
			outMu.at<double>(j, 0) += inData.at<double>(j, i);
		}
	}

	// dividing by the total number to get the average
	outMu = outMu/inData.cols;
	//std::cout << std::endl;
	// now calculate the covariance
	for(int i = 0; i < inData.cols; i++)
	{
		for(int j = 0; j < inData.rows; j++)
		{
			for(int j2 = 0; j2 < inData.rows; j2++)
			{
				outCov.at<double>(j, j2) += (inData.at<double>(j, i) - outMu.at<double>(j, 0)) * (inData.at<double>(j2, i) - outMu.at<double>(j2, 0));

			}
		}
	}

	//std::cout << std::endl;
	// dividing by the total number to get the covariance
	outCov = outCov/inData.cols;

}

// fucntion for random initialization of hashmap
void randomInitialize(cv::Mat &inHashmap, int clusterNum)
{
	for(int i = 0; i < inHashmap.cols; i++)
	{
		inHashmap.at<double>(0, i) = std::floor(cv::theRNG().uniform(0.0, (double)clusterNum));
	}
}


// function for generating dummy data
cv::Mat generateData(void)
{
	cv::Mat outMat;

	// initialize with a number of points
	outMat = cv::Mat::zeros(1, 100, CV_64FC1);

	// fill with 1D location of gaussian

	for(int i = 0; i < outMat.cols; i++)
	{
		if( i < 50)
		{
			outMat.at<double>(0, i) = cv::theRNG().gaussian(2);
			//outMat.at<double>(1, i) = cv::theRNG().gaussian(6);
		}
		else
		{
			outMat.at<double>(0, i) = cv::theRNG().gaussian(2) + 6;
			//outMat.at<double>(1, i) = cv::theRNG().gaussian(6) + 6;
		}

	}

	return outMat;
}



