#pragma once 
#include "common.h"
#include "Plot.h"
#include "DatasetLoader.h"

double C_REGULARIZATION_VALUE = 0.01;

void fitGMM(const cv::Mat &inData, const int &K, cv::Mat &piK, cv::Mat &muK, cv::Mat &sigmaK);
void randomInitialize(cv::Mat &inHashmap, int clusterNum);
void getGMMOfCluster(const cv::Mat &inData, const int &K, const cv::Mat &assignMap, cv::Mat &piK, cv::Mat &muK, cv::Mat &sigmaK);
void getMeanAndCovariance(const cv::Mat &inData, cv::Mat &outMu, cv::Mat &outCov);
void getAssignmentScore(const cv::Mat &inData, const cv::Mat &piK, const cv::Mat &muK, const cv::Mat &sigmaK, cv::Mat &assignScore);
double getGaussian2D(const cv::Mat &inPoint, const cv::Mat &muC, const cv::Mat &sigmaC);
void updateGMM(const cv::Mat &inData, const cv::Mat &assignScore, cv::Mat &piK, cv::Mat &muK, cv::Mat &sigmaK);
void getMeanAndCovarianceWeighted(const cv::Mat &inData, const cv::Mat &inWeights, cv::Mat &outPi, cv::Mat &outMu, cv::Mat &outCov);
double getLogLikelihood(const cv::Mat &inData, const cv::Mat &piK, const cv::Mat &muK, const cv::Mat &sigmaK);
cv::Mat generateData(void);
void clusterWithGMM(const cv::Mat &inData, const cv::Mat &piK, const cv::Mat &muK, const cv::Mat &sigmaK, cv::Mat &clusterLabel);

int main(void)
{
	// random seed
	cv::theRNG().state = cv::getTickCount();
	cv::Mat xMat; // for containing all the observations from multiple gaussians

	//if(1 == 2)
	//{
	//	DatasetLoader myData;
	//	cv::Mat allData = myData.readMatlabFile("testingAngles.dat");
	//	int sizeOfData = 20;
	//	xMat = cv::Mat::zeros(2, sizeOfData, CV_64FC1);
	//	// extract a small part of the data
	//	for(int i = 0; i < sizeOfData; i++)
	//	{
	//		xMat.at<double>(0, i) = allData.at<float>(0, i);
	//		xMat.at<double>(1, i) = allData.at<float>(1, i);
	//	}
	//}
	//else
	//{
		// generate dummy data to check
		xMat = generateData().clone();	
	//}




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

	cv::Mat clusterLabel;
	clusterWithGMM(xMat, piK, muK, sigmaK, clusterLabel);

	// display the cluster results
	std::cout << " cluster labels : " << clusterLabel << std::endl;

	
}

// function for fitting GMM model with K clusters
void fitGMM(const cv::Mat &inData, const int &K, cv::Mat &piK, cv::Mat &muK, cv::Mat &sigmaK)
{
	// intialize the assignment hashmap
	cv::Mat assignMap;
	assignMap = cv::Mat::zeros(1, inData.cols, CV_64FC1);

	// random initilization of gmm
	randomInitialize(assignMap, K);

	// temp display
	//std::cout << assignMap << std::endl;

	// calculate the mean and variance of the clusters
	getGMMOfCluster(inData, K, assignMap, piK, muK, sigmaK);

	// display cov and mean
	//std::cout << "curWeight: " << piK << std::endl;
	//std::cout << "curMean: " << muK << std::endl;
	//std::cout << "curCov: " << sigmaK << std::endl;
	CPlot myPlot(1, 600, 600);
	//myPlot.plot(xMat.row(0).clone(), xMat.row(1).clone(), -10.0, 10.0, -10.0, 10.0, C_BLACK);
	//myPlot.plot(1.0, 1.0, -10.0, 10.0, -10.0, 10.0, C_RED);
	//std::cout << xMat << std::endl;
	// next step is to get the assignment score
	cv::Mat assignScore;
	double maxlogLH = -DBL_MAX;
	for(int i = 0; i < 5000; i++)
	{
		getAssignmentScore(inData, piK, muK, sigmaK, assignScore);

		updateGMM(inData, assignScore, piK, muK, sigmaK);
		cv::Mat clusterLabel;
		clusterWithGMM(inData, piK, muK, sigmaK, clusterLabel);
		for(int iIdx = 0; iIdx < inData.cols; iIdx++)
		{
			if(clusterLabel.at<double>(0, iIdx) == 0)
				myPlot.plot(inData.at<double>(0, iIdx), inData.at<double>(1, iIdx), -10.0, 10.0, -10.0, 10.0, C_RED);
			else
				myPlot.plot(inData.at<double>(0, iIdx), inData.at<double>(1, iIdx), -10.0, 10.0, -10.0, 10.0, C_GREEN);
		}

		for(int iIdx = 0; iIdx < muK.cols; iIdx++)
		{
			cv::Rect roi(iIdx * muK.rows, 0, muK.rows, muK.rows);
			if(iIdx == 0)
				myPlot.drawErrorEllipse(muK.col(iIdx), sigmaK(roi), C_RED);
			else
				myPlot.drawErrorEllipse(muK.col(iIdx), sigmaK(roi), C_GREEN);
			
		}

		myPlot.drawNow();
 		//cv::waitKey(0);

		
		double logLH = getLogLikelihood(inData, piK, muK, sigmaK);

		if( (logLH - maxlogLH) > 0.000001 )
			maxlogLH = logLH;
		else
		{
			std::cout << "Log likelihood converged" << std::endl;
			break;
		}
		std::cout << "logLikelihood : " << getLogLikelihood(inData, piK, muK, sigmaK) << std::endl;

		char buffer[50];
		sprintf(buffer, "outImage_%0.5d.png", i);
		cv::imwrite(buffer, myPlot.m_display);
		myPlot.clear();
	}
	cv::waitKey(0);

}

// fucntion for random initialization of hashmap
void randomInitialize(cv::Mat &inHashmap, int clusterNum)
{
	for(int i = 0; i < inHashmap.cols; i++)
	{
		inHashmap.at<double>(0, i) = std::floor(cv::theRNG().uniform(0.0, (double)clusterNum));
	}
}

// get GMM from initialized random clusters
void getGMMOfCluster(const cv::Mat &inData, const int &K, const cv::Mat &assignMap, cv::Mat &piK, cv::Mat &muK, cv::Mat &sigmaK)
{
	// create container for weights, mean and cov
	piK = cv::Mat::zeros(1, K, CV_64FC1);
	muK = cv::Mat::zeros(inData.rows, K, CV_64FC1);
	sigmaK = cv::Mat::zeros(inData.rows, inData.rows * K, CV_64FC1);

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
		curPoints = cv::Mat::zeros(inData.rows, curCount, CV_64FC1);
		int curIdx = 0;
		for(int idx = 0; idx < assignMap.cols; idx++)
		{
			if(assignMap.at<double>(0, idx) == i)
			{
				inData.col(idx).copyTo(curPoints.col(curIdx++));
			}
		}

		piK.at<double>(0, i) = curCount;

		//std::cout << "curPoints: " << curPoints << std::endl;
		cv::Mat curCov, curMean;
		curCov = cv::Mat::zeros(inData.rows, inData.rows, CV_64FC1);
		curMean = cv::Mat::zeros(inData.rows, 1, CV_64FC1);

		//cv::calcCovarMatrix(curPoints, curCov, curMean, CV_COVAR_COLS | CV_COVAR_NORMAL | CV_COVAR_SCALE);
		//std::cout << "curCovariance: " << curCov << std::endl;
		//std::cout << "curMean: " << curMean << std::endl;

		getMeanAndCovariance(curPoints, curMean, curCov);
		//std::cout << "curCovariance: " << curCov << std::endl;
		//std::cout << "curMean: " << curMean << std::endl;

		// regularization
        cv::Mat eyeMat = cv::Mat::eye(inData.rows, inData.rows, CV_64FC1);
        curCov = C_REGULARIZATION_VALUE * eyeMat + (1 - C_REGULARIZATION_VALUE) * curCov;

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
	piK = piK/inData.cols;
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

// get the score for assigning each data point to each gaussian cluster in GMM
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

// get probability from multi-dimensional gaussian
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

void updateGMM(const cv::Mat &inData, const cv::Mat &assignScore, cv::Mat &piK, cv::Mat &muK, cv::Mat &sigmaK)
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
		getMeanAndCovarianceWeighted(inData, assignScore.row(k), curPi, curMean, curCov);
		std::cout << "Mean after: " << curMean <<std::endl;
		std::cout << "Pi After: " << curPi <<std::endl;

		// regularization of covariance using identity
        cv::Mat eyeMat = cv::Mat::eye(muK.rows, muK.rows, CV_64FC1);
        curCov = C_REGULARIZATION_VALUE * eyeMat + (1 - C_REGULARIZATION_VALUE) * curCov;

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

// get the loglikelihood for checking convergence
double getLogLikelihood(const cv::Mat &inData, const cv::Mat &piK, const cv::Mat &muK, const cv::Mat &sigmaK)
{
	double retProbability = 0;
	for(int n = 0; n < inData.cols; n++)
	{
		double innerProbability = 0;
		for(int k = 0; k < piK.cols; k++)
		{
			cv::Rect roi(k * muK.rows, 0, muK.rows, muK.rows);
			innerProbability += piK.at<double>(0, k) * getGaussian2D(inData.col(n).clone(), muK.col(k).clone(), sigmaK(roi));
		}
		retProbability += std::log(innerProbability);
	}

	return retProbability;
}

// function for generating dummy data
cv::Mat generateData(void)
{
	cv::Mat outMat;

	// initialize with a number of points
	outMat = cv::Mat::zeros(2, 500, CV_64FC1);

	// fill with 1D location of gaussian

	for(int i = 0; i < outMat.cols; i++)
	{
		if( i < outMat.cols/2)
		{
			outMat.at<double>(0, i) = cv::theRNG().gaussian(1);
			outMat.at<double>(1, i) = cv::theRNG().gaussian(1);
			//outMat.at<double>(1, i) = cv::theRNG().gaussian(6);
		}
		else
		{
			 outMat.at<double>(0, i) = cv::theRNG().gaussian(1) + 6;
			 outMat.at<double>(1, i) = cv::theRNG().gaussian(1) + 6;
			//outMat.at<double>(0, i) = outMat.at<double>(0, i-50) + cv::theRNG().gaussian(2);
			//outMat.at<double>(1, i) = outMat.at<double>(1, i-50) + cv::theRNG().gaussian(2);
			//outMat.at<double>(1, i) = cv::theRNG().gaussian(6) + 6;
		}

	}

	return outMat;
}

// function for assigning cluster labels
void clusterWithGMM(const cv::Mat &inData, const cv::Mat &piK, const cv::Mat &muK, const cv::Mat &sigmaK, cv::Mat &clusterLabel)
{
	// get the assignment score for each data point
	cv::Mat assignScore;
	getAssignmentScore(inData, piK, muK, sigmaK, assignScore);

	clusterLabel = cv::Mat::zeros(1, assignScore.cols, CV_64FC1);

	//  take arg max over each data point score from each mixture
	for(int i = 0; i < inData.cols; i++)
	{
		double maxIdx = -1;
		double maxIdxVal = -DBL_MAX;
	
		for(int k = 0; k < piK.cols; k++)
		{
			if( maxIdxVal < assignScore.at<double>(k, i) )
			{
				maxIdx = k;
				maxIdxVal = assignScore.at<double>(k, i);
			}
		}

		clusterLabel.at<double>(0, i) = maxIdx;
	}
}
