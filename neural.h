/*
实现神经网络的搭建
采用误差反向传播
*/
#pragma once
#include<iostream>
#include"matrix.h"

using namespace std;

//一层的基础神经网络
class neural {
public:
	neural(int inputnodes, int outputnodes, double learingrate);
	~neural();
	matrix train(double input_list[], double error_list[]);
	matrix train(matrix input_mat, matrix error_mat);
	matrix train(double input_list[], matrix error_mat);
	matrix train(matrix input_mat, double error_list[]);
	matrix query(double inputs[]);
	matrix query(matrix input_mat);
	void show_wio();
	
	matrix wio{ 0,0 };
	matrix out{ 0,0 };
private:
	int inputnodes=0;
	int outputnodes=0;
	double learingrate=0;
	
	
};




//多层深度神经网络，输入层个数，隐藏层个数，输出层个数，神经网络深度，学习率
class DeepNeural {
public:
	DeepNeural(int inputnodes, int hiddennodes, int outputnodes, int hiddennum, double learingrate);
	DeepNeural();
	~DeepNeural();
	matrix train(double input_list[], double output_list[]);
	matrix train(matrix input, matrix error);
	matrix query(matrix input_mat);
	matrix query(double input_list[]);

private:
	int inputnodes = 0;
	int outputnodes = 0;
	int hiddennodes = 0;
	int hiddennum = 0;
	double learingrate = 0;
	vector<neural> body;
};


//卷积部分的卷积单元,输入为二维矩阵, 激活函数为ReLU
class con_neural {
public:
	con_neural(int InRow, int InColumn, int KeRow, int KeColumn, int stride, double learingrate,int mod, bool pooling, int poolrow, int poolcol);
	~con_neural();
	matrix pooling(matrix target);
	//卷积加激活后的输出
	matrix query(vector<matrix> a);
	//训练
	vector<matrix> train(matrix error);

	vector<matrix> OutPut;

private:
	int InRow, InColumn = 0;
	int stride = 0;
	double learingrate = 0;
	int con_row, con_column = 0;
	int poolrow, poolcol = 0;
	int mod = 0;
	bool pool = true;
	matrix kernel{ 0,0 };
	matrix OneMat{ 0,0 };
	vector<matrix> max_row;
	vector<matrix> max_col;
	vector<matrix> input;
};



//单位卷积层
class ConvolutionNeural {
public:
	//EachConNum顺序储存每个卷积层的卷积核个数
	ConvolutionNeural(int EachConNum[],double LearningRate, bool is_pooling);
	~ConvolutionNeural();
	vector<matrix> query(matrix input);
	vector<matrix> query(vector<matrix> input);
	vector<matrix> trian(vector<matrix> error);
	vector<con_neural> EachConvolu;


};


//完整卷积神经网络(不包含全连接层)
class CNN {
public:
	CNN(int num, int EachConNum[][9], double LearningRate[], bool is_pooling[]);
	CNN();
	~CNN();
	vector<matrix> query(matrix input);
	void train(matrix input, matrix error);
private:
	//vector<vector<ConvolutionNeural>> CoNe;
	//vector<vector<matrix>> EachOutMat;
	//int CoNeNum = 0;
	vector<ConvolutionNeural> Con;

};


//加上全连接层的卷积神经网络
class ConDeeNetwork {
public:
	ConDeeNetwork(CNN CNNPart, DeepNeural DNNPart);
	~ConDeeNetwork();
	matrix query(matrix input);
	void train(vector<matrix> TrainingData, int TrainNum, matrix OutData);
private:
	CNN CNNPart;
	DeepNeural DNNPart;
	matrix CNNOut;
};

