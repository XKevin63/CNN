
/*实现矩阵的加法、减法、乘法*/
#pragma once
#include<iostream>
#include<vector>
#include<fstream>

using namespace std;
static int MAX_POOLING = 1;
static int VEG_POOLING = 0;

class matrix {
public:
	matrix(int row, int column);
	matrix(int row, int column, double inital);
	matrix();
	~matrix();
	//提取对应的行矩阵
	matrix GetRow(int StartRow, int EndRow);
	void show();
	//记录行数和列数
	int row;
	int column;
	//在容器中记录数据
	vector<vector<double>>  data;
	void set_1d(double list[]);
};

//设置矩阵加减乘运算
matrix add_mat(matrix a, matrix b);
matrix add_mat(double bias, matrix a);
vector<matrix> add_mat(vector<matrix> a, vector<matrix> b);
matrix sub_mat(matrix a, matrix b);

matrix mul_mat(matrix a, matrix b);
matrix mul(matrix a, matrix b);
matrix mul(double a, matrix b);
matrix divide(matrix a, matrix b);
matrix sub(double a, matrix b);
matrix To_Matrix(double a[], int num);

//激活函数
double sigmoid1(double x);
double ReLU(double x);

matrix sigmoid_mat(matrix a);
matrix ReLU_mat(matrix a);
matrix transposition(matrix a);

//卷积运算
matrix convolution_mat(matrix target, matrix kernel, int stride);

//获得数据集
vector<matrix> DataGet(int num, int row, int column, string route);

matrix DataGet1Col(int row, int column, string route);
matrix MergeVectorMatrix(vector<matrix> data);

//进行卷积层数据的误差分配
vector<matrix> ErrorDistribute(vector<matrix> OutMat, matrix OneMat, matrix InError);
matrix ErrorCNNDist(matrix Error, matrix kernel, int stride);
//归一化
double NormalizationMat(matrix* a);

matrix FlodMat(vector<matrix> input);
vector<matrix> ReveseFlodMat(int row, int column, matrix input);