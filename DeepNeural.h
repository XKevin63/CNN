#pragma once
#include"matrix.h"



class neural {
public:
	neural(int inputnodes, int hiddennodes, double learningrate);
	matrix query(matrix input);
	matrix train(matrix error);
	bool Setdwio(double num);
	matrix wio;
	matrix dwio;
	matrix OutBeforActive;
	matrix OutAfterActive;
	matrix input;
	int inputnodes, hiddennodes, outputnodes;
private:
	double learningrate;
	matrix bias;
	matrix db;
};


class softmax {
public:
	softmax(int inputnodes);
	matrix query(matrix input);
	matrix train(matrix target);
	matrix intput;
	matrix output;
private:
	int inputnodes;
};



class FullyConnectedLayer {
public:
	FullyConnectedLayer(int LayerNum, neural NeuralList[]);
	FullyConnectedLayer();
	matrix query(matrix input);
	matrix train(matrix target);
	bool SetWio(double num);
private:
	vector<neural> BpPart;
	softmax SoftPart{ 0 };
};