#pragma once
#include"matrix.h"

//����Ұ
class fliter {
public:
	fliter(int chanel, int row, int column, int stride, double learningrate);
	matrix query(vector<matrix> input);
	vector<matrix> train(matrix dL_dz);
	bool SetdKernel(double num);
	matrix OutBeforReLU;
private:
	int chanel, row, column, stride;
	double learningrate;
	vector<matrix> kernel;
	vector<matrix> dk;
	vector<matrix> input;
	double bias;
	double dbias;
};


//�����
class ConvolutionFliter {
public:
	ConvolutionFliter(int FliterNum, int chanel, int row, int column, int stride, double learningrate, int PoolRow, int PoolCol);
	vector<matrix> query(vector<matrix> input);
	vector<matrix> train(vector<matrix> error);
	//�ػ�
	matrix pooling(matrix target, vector<matrix>* max_row, vector<matrix>* max_col);
	//���ػ�
	matrix ReversePool(matrix input, int num);
	bool SetdKernel(double num);
	int FliterNum;
private:
	vector<fliter> FliterPart;
	int PoolRow, PoolCol;
	vector<matrix> max_row;
	vector<matrix> max_col;
};




//�������
class ConvolutionLayer {
public:
	ConvolutionLayer(int num, ConvolutionFliter ConFlilist[]);
	ConvolutionLayer();
	matrix query(matrix input);
	matrix train(matrix error);
	bool SetdKernel(double num);
private:
	vector<ConvolutionFliter> ConFliPart;
	bool FlodKey = false;
	int row, column;
};
