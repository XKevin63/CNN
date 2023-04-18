
/*ʵ�־���ļӷ����������˷�*/
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
	//��ȡ��Ӧ���о���
	matrix GetRow(int StartRow, int EndRow);
	void show();
	//��¼����������
	int row;
	int column;
	//�������м�¼����
	vector<vector<double>>  data;
	void set_1d(double list[]);
};

//���þ���Ӽ�������
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

//�����
double sigmoid1(double x);
double ReLU(double x);

matrix sigmoid_mat(matrix a);
matrix ReLU_mat(matrix a);
matrix transposition(matrix a);

//�������
matrix convolution_mat(matrix target, matrix kernel, int stride);

//������ݼ�
vector<matrix> DataGet(int num, int row, int column, string route);

matrix DataGet1Col(int row, int column, string route);
matrix MergeVectorMatrix(vector<matrix> data);

//���о�������ݵ�������
vector<matrix> ErrorDistribute(vector<matrix> OutMat, matrix OneMat, matrix InError);
matrix ErrorCNNDist(matrix Error, matrix kernel, int stride);
//��һ��
double NormalizationMat(matrix* a);

matrix FlodMat(vector<matrix> input);
vector<matrix> ReveseFlodMat(int row, int column, matrix input);