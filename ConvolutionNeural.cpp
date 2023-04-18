#include"ConvolutionNeural.h"


//感受野
fliter::fliter(int chanel, int row, int column, int stride, double learningrate) {
	this->chanel = chanel;
	this->row = row;
	this->column = column;
	this->stride = stride;
	this->learningrate = learningrate;
	for (int i = 0; i < chanel; i++) {
		matrix kernel(row, column);
		matrix dk(row, column, 0);
		this->kernel.push_back(kernel);
		this->dk.push_back(dk);
	}
	this->bias = 0.5;
	this->dbias = 0;
}

matrix fliter::query(vector<matrix> input) {
	//检查数据是否合法
	if (input.size() != this->chanel) {
		cout << "fliter input error" << endl;
		exit(0);
	}
	this->input = input;
	//输出查询，不要池化，要激活
	matrix out = convolution_mat(input[0], this->kernel[0], this->stride);
	for (int ChanelNum = 1; ChanelNum < this->chanel; ChanelNum++) {
		matrix res = convolution_mat(input[ChanelNum], this->kernel[ChanelNum], this->stride);
		out = add_mat(out, res);
	}
	out = add_mat(this->bias, out);
	this->OutBeforReLU = out;
	out = ReLU_mat(out);
	return out;
}

vector<matrix> fliter::train(matrix dL_dz) {
	//检验数据合法
	if (dL_dz.row != this->OutBeforReLU.row || dL_dz.column != this->OutBeforReLU.column) {
		cout << "fliter train error" << endl;
		exit(0);
	}
	matrix error = dL_dz;
	//进行激活求导
	
	for (int i = 0; i < dL_dz.row; i++) 
		for (int j = 0; j < dL_dz.column; j++) 
			if (this->OutBeforReLU.data[i][j] <= 0) 
				error.data[i][j] = 0;
	
	vector<matrix> out;
	vector<matrix> dkerror;
	for (int ChanelNum = 0; ChanelNum < this->chanel; ChanelNum++) {
		//求上层误差
		matrix dout = ErrorCNNDist(error, this->kernel[ChanelNum], this->stride);
		out.push_back(dout);
		//求梯度
		//卷积核
		matrix dk = convolution_mat(this->input[ChanelNum], error, this->stride);
		dkerror.push_back(dk);
		//dk = mul(this->learningrate, dk);
		//dk.show();
		//int w;
		//cin >> w;
		//this->kernel[ChanelNum] = sub_mat(this->kernel[ChanelNum], dk);
	}
	//偏置
	double db = 0;
	for (int i = 0; i < error.row; i++)
		for (int j = 0; j < error.column; j++)
			db += error.data[i][j];
	//this->bias -= (db * this->learningrate);
	this->dbias += db;
	this->dk = add_mat(this->dk, dkerror);
	return out;
}

bool fliter::SetdKernel(double num) {
	vector<matrix> orgin;
	for (int i = 0; i < this->chanel; i++) {
		matrix dk = mul(this->learningrate / num, this->dk[i]);
		this->kernel[i] = sub_mat(this->kernel[i], dk);
		matrix a(this->row, this->column, 0);
		orgin.push_back(a);
	}
	this->bias -= this->learningrate * (this->dbias / num);
	this->dbias = 0;
	this->dk = orgin;
	return true;
}

//单独卷积层
ConvolutionFliter::ConvolutionFliter(int FliterNum, int chanel, int row, int column, int stride, double learningrate, int PoolRow, int PoolCol) {
	for (int i = 0; i < FliterNum; i++) {
		fliter ea(chanel, row, column, stride, learningrate);
		this->FliterPart.push_back(ea);
	}
	this->PoolCol = PoolCol;
	this->PoolRow = PoolRow;
	this->FliterNum = FliterNum;
}

//池化
matrix ConvolutionFliter::pooling(matrix target, vector<matrix>* mrow, vector<matrix>* mcol) {
	int row = this->PoolRow;
	int col = this->PoolCol;
	//检验数据
	if (target.row % row != 0 or target.column % col != 0) {
		std::cout << "pooling error" << endl;
		exit(0);
	}
	//池化
	matrix out(target.row / row, target.column / col);
	matrix max_row(target.row / row, target.column / col);
	matrix max_col(target.row / row, target.column / col);
	for (int i = 0; i < out.row; i++)
		for (int j = 0; j < out.column; j++) {
			double max = target.data[i * row][j * col];
			max_row.data[i][j] = i * row;
			max_col.data[i][j] = j * col;
			for (int ir = 0; ir < row; ir++) 
				for (int ic = 0; ic < col; ic++) 
					if (target.data[i * row + ir][j * col + ic] > max) {
						max = target.data[i * row + ir][j * col + ic];
						max_row.data[i][j] = i * row + ir;
						max_col.data[i][j] = j * col + ic;
					}
			out.data[i][j] = max;
		}
	mrow->push_back(max_row);
	mcol->push_back(max_col);
	return out;
}


vector<matrix> ConvolutionFliter::query(vector<matrix> input) {
	vector<matrix> max_row;
	vector<matrix> max_col;
	vector<matrix> out;
	for (int i = 0; i < this->FliterPart.size(); i++) {
		matrix fliterout = this->FliterPart[i].query(input);
		fliterout = pooling(fliterout, &max_row, &max_col);
		out.push_back(fliterout);
	}
	this->max_col = max_col;
	this->max_row = max_row;
	return out;
}

matrix ConvolutionFliter::ReversePool(matrix input, int i) {
	matrix max_row = this->max_row[i];
	matrix max_col = this->max_col[i];
	matrix error_befor_pool(this->FliterPart[i].OutBeforReLU.row, this->FliterPart[i].OutBeforReLU.column, 0);
	for (int w = 0; w < input.row; w++)
		for (int j = 0; j < input.column; j++) 
			error_befor_pool.data[max_row.data[w][j]][max_col.data[w][j]] = input.data[w][j];
	return error_befor_pool;
}


vector<matrix> ConvolutionFliter::train(vector<matrix> error) {
	//检查数据
	if (error.size() != this->FliterPart.size()) {
		cout << "ConvolutionFliter train error" << endl;
		exit(0);
	}
	//进行反池化
	matrix dL_dZ = ReversePool(error[0], 0);
	vector<matrix> out = this->FliterPart[0].train(dL_dZ);
	for (int i = 1; i < error.size(); i++) {
		dL_dZ = ReversePool(error[i], i);
		out = add_mat(out, this->FliterPart[i].train(dL_dZ));
	}
	return out;
}


bool ConvolutionFliter::SetdKernel(double num) {
	for (int i = 0; i < this->FliterPart.size(); i++) {
		this->FliterPart[i].SetdKernel(num);
	}
	return true;
}


//卷积部分
ConvolutionLayer::ConvolutionLayer(int num, ConvolutionFliter ConFlilist[]) {
	for (int i = 0; i < num; i++) 
		this->ConFliPart.push_back(ConFlilist[i]);
}
ConvolutionLayer::ConvolutionLayer() {
}

matrix ConvolutionLayer::query(matrix input) {
	vector<matrix> input_mat;
	input_mat.push_back(input);
	vector<matrix> out = this->ConFliPart[0].query(input_mat);
	for (int i = 1; i < this->ConFliPart.size();i++) 
		out = this->ConFliPart[i].query(out);
	matrix result;
	if (out[0].column != 1) {
		this->FlodKey = true;
		this->row = out[0].row;
		this->column = out[0].column;
		result = FlodMat(out);
	}
	else {
		result = MergeVectorMatrix(out);
		this->FlodKey = false;
	}
	return result;
}

matrix ConvolutionLayer::train(matrix error) {
	if (this->FlodKey == true) {
		vector<matrix> hidden = ReveseFlodMat(this->row, this->column, error);
		error = MergeVectorMatrix(hidden);
	}
	vector<matrix> InputError;
	int ErrorNum = this->ConFliPart[this->ConFliPart.size() - 1].FliterNum;
	for (int i = 0; i < ErrorNum; i++) {
		matrix hid = error.GetRow(i * (error.row / ErrorNum) + 1, (i + 1) * (error.row / ErrorNum));
		InputError.push_back(hid);
	}
	for (int i = this->ConFliPart.size() - 1; i >= 0; i--) 
		InputError = this->ConFliPart[i].train(InputError);
	error = MergeVectorMatrix(InputError);
	return error;
}

bool ConvolutionLayer::SetdKernel(double num) {
	for (int i = 0; i < this->ConFliPart.size(); i++)
		this->ConFliPart[i].SetdKernel(num);
	return true;
}