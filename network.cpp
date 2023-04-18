#include"network.h"




network::network(int ConNum, ConvolutionFliter ConList[], int FulNum, neural Fulist[]) {
	ConvolutionLayer a(ConNum, ConList);
	FullyConnectedLayer b(FulNum, Fulist);
	this->CNN = a;
	this->DNN = b;
}


matrix network::query(matrix input) {
	matrix CNNOutput = this->CNN.query(input);
	//CNNOutput.show();
	matrix DNNOutput = this->DNN.query(CNNOutput);
	return DNNOutput;
}

//ÑµÁ·Ç°ÇëÏÈquery
matrix network::train(matrix target) {
	matrix DNNError = this->DNN.train(target);
	matrix CNNError = this->CNN.train(DNNError);
	return CNNError;
}

/*
void network::AllDataTrain(int TrainNum, int SigNum, vector<matrix> AllData[], matrix OutData[], double TrainPart) {
	int Train = TrainNum * TrainPart;
	int Test = TrainNum - Train;
	for(int i=0;i<SigNum)
}
*/

bool network::SetParm(double num) {
	this->CNN.SetdKernel(num);
	this->DNN.SetWio(num);
	return true;
}