#pragma once
#include"ConvolutionNeural.h"
#include"DeepNeural.h"


class network {
public:
	network(int ConNum, ConvolutionFliter ConList[], int Con, neural Fulist[]);
	matrix query(matrix input);
	matrix train(matrix target);
	//void AllDataTrain(int TrainNum, int SigNum, vector<matrix> AllData[], matrix OutData[], double TrainPart);
	bool SetParm(double num);
private:
	ConvolutionLayer CNN;
	FullyConnectedLayer DNN;
};