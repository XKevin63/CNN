#include "matrix.h"
#include<cstdlib>
#include<ctime>
#include<iomanip>


using namespace std;


matrix::matrix() {
}
//初始化矩阵，生成随机矩阵
matrix::matrix(int row, int column) {
	this->column = column;
	this->row = row;
	for (int i = 0; i < row; i++) {
		vector<double> col(column, 1);
		for (int t = 0; t < column; t++) {
			//创建[-0.5,0.5)的随机数
			col[t] = double((rand() % 10000)) / 10000 + 0.1;
		}
		this->data.push_back(col);
	}
}

//初始化固定值
matrix::matrix(int row, int column, double initial) {
	this->column = column;
	this->row = row;
	for (int i = 0; i < row; i++) {
		vector<double> col(column, 1);
		for (int t = 0; t < column; t++) {
			col[t] = initial;
		}
		this->data.push_back(col);
	}
}


matrix::~matrix() {
}

//展示矩阵
void matrix::show() {
	for (int i = 0; i < row; i++) {
		cout << setw(10) << "| ";
		for (int j = 0; j < column; j++) {
			cout << setw(10) << this->data[i][j] << " ";
		}
		cout << setw(10) << "|" << endl;
	}
	cout << endl;
}

//加减乘法
matrix add_mat(matrix a, matrix b) {
	if (a.row == b.row and a.column == b.column) {
		matrix r(a.row, a.column);
		for (int i = 0; i < r.row; i++) {
			for (int j = 0; j < r.column; j++) {
				r.data[i][j] = a.data[i][j] + b.data[i][j];
			}
		}
		return r;

	}
	else {
		cout << "add_mat error" << endl;
		exit(0);
	}
}

vector<matrix> add_mat(vector<matrix> a, vector<matrix> b) {
	if (a.size() != b.size()) {
		cout << "num error" << endl;
		exit(0);
	}
	vector<matrix> out;
	matrix c = add_mat(a[0], b[0]);
	out.push_back(c);
	for (int i = 1; i < a.size(); i++) {
		c = add_mat(a[i], b[i]);
		out.push_back(c);
	}
	return out;
}

matrix sub_mat(matrix a, matrix b) {
	if (a.row == b.row and a.column == b.column) {
		matrix r(a.row, a.column);
		for (int i = 0; i < r.row; i++) {
			for (int j = 0; j < r.column; j++) {
				r.data[i][j] = a.data[i][j] - b.data[i][j];
			}
		}
		return r;

	}
	else {
		cout << "sub_mat error" << endl;
		exit(0);
	}
}

matrix mul_mat(matrix a, matrix b) {
	if (a.column != b.row) {
		cout << " mul_mat error" << endl;
		exit(0);
	}
	matrix r(a.row, b.column);
	for (int ri = 0; ri < r.row; ri++) {
		for (int rj = 0; rj < r.column; rj++) {
			double sum = 0;
			for (int ai = 0; ai < a.column; ai++) {
				sum += double(a.data[ri][ai] * b.data[ai][rj]);
			}
			r.data[ri][rj] = sum;
		}
	}
	return r;
}


//设置列向量数据
void matrix::set_1d(double list[]) {
	for (int i = 0; i < row; i++) {
		this->data[i][0] = list[i];
	}
}


//激活函数
matrix sigmoid_mat(matrix a) {
	if (a.column != 1) {
		cout << "error" << endl;
		exit(0);
	}
	matrix b(a.row, 1);
	for (int i = 0; i < a.row; i++) {
		b.data[i][0] = sigmoid1(a.data[i][0]);
	}
	return b;
}
matrix ReLU_mat(matrix a) {
	matrix b(a.row, a.column);
	for (int i = 0; i < a.row; i++) {
		for (int j = 0; j < a.column; j++) {
			b.data[i][j] = ReLU(a.data[i][j]);
		}
	}
	return b;
}



//转置
matrix transposition(matrix a) {
	matrix b(a.column, a.row);
	for (int i = 0; i < b.row; i++) {
		for (int j = 0; j < b.column; j++) {
			b.data[i][j] = a.data[j][i];
		}
	}
	return b;
}


double sigmoid1(double x) {
	return (1 / (1 + exp(-x)));
}
double ReLU(double x) {
	if (x <= 0) 
		return 0;
	else
		return x;
}

//矩阵按对应位置相乘
matrix mul(matrix a, matrix b) {
	if (a.row != b.row or a.column != b.column) {
		cout << "error_mul" << endl;
		exit(0);
	}
	matrix c(a.row, a.column);
	for (int i = 0; i < a.row; i++) {
		for (int j = 0; j < a.column; j++) {
			c.data[i][j] = a.data[i][j] * b.data[i][j];
		}
	}
	return c;
}

matrix mul(double a, matrix b) {
	matrix c(b.row, b.column);
	for (int i = 0; i < b.row; i++) {
		for (int j = 0; j < b.column; j++) {
			c.data[i][j] = b.data[i][j] * a;
		}
	}
	return c;
}

//矩阵对应位置相除
matrix divide(matrix a, matrix b) {
	if (a.column != b.column || a.row != b.row) {
		cout << "divide error" << endl;
		exit(0);
	}
	matrix c(a.row, a.column);
	for (int i = 0; i < b.row; i++) {
		for (int j = 0; j < b.column; j++) {
			c.data[i][j] = a.data[i][j] / b.data[i][j];
		}
	}
	return c;
}


matrix sub(double a, matrix b) {
	matrix c(b.row, b.column);
	for (int i = 0; i < b.row; i++) {
		for (int j = 0; j < b.column; j++) {
			c.data[i][j] = a - b.data[i][j];
		}
	}
	return c;
}

//卷积，stride为步长
matrix convolution_mat(matrix target, matrix kernel, int stride) {
	if (target.column < kernel.column or target.row < kernel.row or (target.column - kernel.column) % stride != 0 or (target.row - kernel.row) % stride != 0 ) {
		cout << "convolution error" << target.column << kernel.column << endl;
		exit(0);
	}
	int column = (target.column - kernel.column) / stride + 1;
	int row = (target.row - kernel.row) / stride + 1;
	matrix out_mat(row, column);
	//卷积
	for (int irow = 0; irow < row; irow++) {
		for (int icol = 0; icol < column; icol++) {
			double sum = 0;
			for (int i = 0; i < kernel.row; i++) {
				for (int j = 0; j < kernel.column; j++) {
					sum += target.data[irow * stride + i][icol * stride + j] * kernel.data[i][j];
				}
			}
			out_mat.data[irow][icol] = sum;
		}
	}
	return out_mat;
}


vector<matrix> DataGet(int num, int row, int column, string route) {
	ifstream data;
	data.open(route, ios::in);
	if (!data.is_open()) {
		cout << "文件打开失败" << endl;
		exit(0);
	}
	vector<matrix> out;
	for (int j = 0; j < num; j++) {
		matrix traindata(row, column, 0);
		for (int i = 0; i < row; i++) {
			double d;
			char c;
			data >> d;
			traindata.data[i][0] = d;
			for (int j = 1; j < column; j++) {
				data >> c >> d;
				traindata.data[i][j] = d;
			}
		}
		out.push_back(traindata);
	}
	data.close();
	return out;
}

matrix DataGet1Col(int row, int column, string route) {
	ifstream data;
	data.open(route, ios::in);
	matrix traindata(row * column, 1, 0);
	if (!data.is_open()) {
		cout << "文件打开失败" << endl;
		exit(0);
	}
	for (int i = 0; i < row; i++) {
		double d;
		char c;
		data >> d;
		traindata.data[i*column][0] = d;
		for (int j = 1; j < column; j++) {
			data >> c >> d;
			traindata.data[i*column+j][0] = d;
		}
	}
	data.close();
	return traindata;
}



//提取从StartRow行到EndRow行的矩阵
matrix matrix::GetRow(int StartRow, int EndRow) {
	matrix out(EndRow + 1 - StartRow, this->column);
	for (int Row = StartRow - 1; Row < EndRow; Row++) {
		for (int Col = 0; Col < out.column; Col++) {
			out.data[Row - StartRow + 1][Col] = this->data[Row][Col];
		}
	}
	return out;
}


//将矩阵向量合并为一个矩阵，按一列排序
matrix MergeVectorMatrix(vector<matrix> data) {
	int OutRow = 0;
	int OutCol = data[0].column;
	for (int i = 0; i < data.size(); i++) {
		OutRow += data[i].row;
	}
	matrix out(OutRow, OutCol);
	int cout = 0;
	for (int i = 0; i < data.size(); i++) {
		for (int j = 0; j < data[i].row; j++) {
			for (int t = 0; t < OutCol; t++) {
				out.data[cout][t] = data[i].data[j][t];
			}
			cout++;
		}
	}
	return out;
	
}

matrix To_Matrix(double a[], int num) {
	matrix out(num, 1);
	for (int i = 0; i < num; i++) {
		out.data[i][0] = a[i];
	}
	return out;
}



//误差分配
vector<matrix> ErrorDistribute(vector<matrix> OutMat, matrix OneMat, matrix InError) {
	vector<matrix> out;
	matrix Intermediate = divide(InError, OneMat);
	for (int i = 0; i < OutMat.size(); i++) {
		matrix In = mul(Intermediate, OutMat[i]);
		out.push_back(In);
	}
	return out;
}

//误差反向传播分配,error补0一圈再与kernel旋转180度卷积
matrix ErrorCNNDist(matrix Error, matrix kernel, int stride) {
	//将卷积核旋转180度
	matrix FinalKernel(kernel.row, kernel.column);
	for (int i = 0; i < kernel.row; i++) 
		for (int j = 0; j < kernel.column; j++) 
			FinalKernel.data[i][j] = kernel.data[kernel.row - i - 1][kernel.column - j - 1];
	//为Error补0
	matrix FinalError(Error.row + (kernel.row - 1) * 2, Error.column + (kernel.column - 1) * 2, 0);
	for (int i = kernel.row - 1; i < kernel.row + Error.row - 1; i++)
		for (int j = kernel.column - 1; j < kernel.column + Error.column - 1; j++) {
			FinalError.data[i][j] = Error.data[i - kernel.row + 1][j - kernel.column + 1];
		}
	//进行卷积
	matrix out = convolution_mat(FinalError, FinalKernel, stride);
	return out;
}