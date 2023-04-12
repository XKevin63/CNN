#include"neural.h"
#include"tools.h"



using namespace std;




/*
		������Ԫ
*/
void neural::show_wio() {
	this->wio.show();
}


neural::neural(int inputnodes, int outputnodes, double learingrate) {
	this->inputnodes = inputnodes;
	this->outputnodes = outputnodes;
	this->learingrate = learingrate;
	matrix wio(outputnodes, inputnodes);
	this->wio = wio;
	matrix out(outputnodes, 1);
	this->out = out;
}

neural::~neural() {
}

matrix neural::train(double input_list[], matrix error_mat) {
	matrix input_mat(inputnodes, 1);
	input_mat.set_1d(input_list);

	//�������������
	matrix final_inputs = mul_mat(this->wio, input_mat);
	matrix final_outputs = sigmoid_mat(final_inputs);
	//final_outputs.show();

	//������һ�����
	matrix upon_errors = mul_mat(transposition(this->wio), error_mat);
	//���²���
	matrix dwio = mul(learingrate, mul_mat(mul(error_mat, mul(final_outputs, sub(1.0, final_outputs))), transposition(input_mat)));
	wio = add_mat(wio, dwio);
	return upon_errors;
}
matrix neural::train(matrix input_mat, double error_list[]) {

	matrix error_mat(outputnodes, 1);
	error_mat.set_1d(error_list);

	//�������������
	matrix final_inputs = mul_mat(this->wio, input_mat);
	matrix final_outputs = sigmoid_mat(final_inputs);
	//final_outputs.show();

	//������һ�����
	matrix upon_errors = mul_mat(transposition(this->wio), error_mat);
	//���²���
	matrix dwio = mul(learingrate, mul_mat(mul(error_mat, mul(final_outputs, sub(1.0, final_outputs))), transposition(input_mat)));
	wio = add_mat(wio, dwio);
	return upon_errors;
}

matrix neural::train(double input_list[], double error_list[]) {

	matrix input_mat(inputnodes, 1);
	input_mat.set_1d(input_list);
	matrix error_mat(outputnodes, 1);
	error_mat.set_1d(error_list);

	//�������������
	matrix final_inputs = mul_mat(this->wio, input_mat);
	matrix final_outputs = sigmoid_mat(final_inputs);
	//final_outputs.show();

	//������һ�����
	matrix upon_errors = mul_mat(transposition(this->wio), error_mat);
	//���²���
	matrix dwio = mul(learingrate, mul_mat(mul(error_mat, mul(final_outputs, sub(1.0, final_outputs))), transposition(input_mat)));
	wio = add_mat(wio, dwio);
	return upon_errors;
}

matrix neural::train(matrix input_mat, matrix error_mat) {
	/*
	error_mat.show();
	input_mat.show();
	wio.show();
	for (int i = 0; i < wio.row; i++) {
		for (int j = 0; j < wio.column; j++) {
			if (std::isnan(wio.data[i][j])) {
				
				exit(0);
			}
		}
	}
	*/
	//�������������
	matrix final_inputs = mul_mat(this->wio, input_mat);
	matrix final_outputs = sigmoid_mat(final_inputs);
	//final_outputs.show();

	//������һ�����
	matrix upon_errors = mul_mat(transposition(this->wio), error_mat);
	//���²���
	matrix dwio = mul(learingrate, mul_mat(mul(error_mat, mul(final_outputs, sub(1.0, final_outputs))), transposition(input_mat)));
	
	wio = add_mat(wio, dwio);
	return upon_errors;
}

//��ѯ���
matrix neural::query(double inputs[]) {
	matrix input_mat(inputnodes, 1);
	input_mat.set_1d(inputs);
	//�������������
	matrix final_inputs = mul_mat(this->wio, input_mat);
	matrix final_outputs = sigmoid_mat(final_inputs);
	this->out = final_outputs;
	return final_outputs;
}

matrix neural::query(matrix input_mat) {
	matrix final_inputs = mul_mat(this->wio, input_mat);
	matrix final_outputs = sigmoid_mat(final_inputs);
	this->out = final_outputs;
	return final_outputs;
}



/*
		��������粿��
*/
DeepNeural::DeepNeural(int inputnodes, int hiddennodes, int outputnodes, int hiddennum, double learingrate) {
	this->inputnodes = inputnodes;
	this->hiddennodes = hiddennodes;
	this->outputnodes = outputnodes;
	this->hiddennum = hiddennum;
	this->learingrate = learingrate;
	neural first(inputnodes, hiddennodes, learingrate);
	neural final(hiddennodes, outputnodes, learingrate);
	this->body.push_back(first);
	for (int i = 0; i < (hiddennum - 2); i++) {
		neural hidden(hiddennodes, hiddennodes, learingrate);
		this->body.push_back(hidden);
	}
	this->body.push_back(final);
}
DeepNeural::DeepNeural() {
}
DeepNeural::~DeepNeural() {
}

matrix DeepNeural::train(double input_list[], double output_list[]) {
	matrix input_mat(inputnodes, 1);
	input_mat.set_1d(input_list);
	matrix output_mat(outputnodes, 1);
	output_mat.set_1d(output_list);

	matrix final_output = query(input_mat);
	matrix final_error = sub_mat(output_mat, final_output);
	matrix error = final_error;
	//final_error.show();
	for (int w = hiddennum; w > 1; w--) {
		error = body[w - 1].train(body[w - 2].out, error);
	}
	matrix out = body[0].train(input_mat, error);
	return out;
}

matrix DeepNeural::train(matrix input, matrix error) {
	matrix final_output = query(input);
	double a = 0;
	for (int i = 0; i < error.row; i++) {
		a = a + error.data[i][0] * error.data[i][0];
	}
	a = sqrt(a);
	cout << "��ǰ��" << a << endl;
	for (int w = hiddennum; w > 1; w--) {
		error = body[w - 1].train(body[w - 2].out, error);
	}
	matrix out = body[0].train(input, error);
	return out;
}

matrix DeepNeural::query(matrix input_mat) {

	matrix output_1 = body[0].query(input_mat);
	for (int i = 1; i < hiddennum - 1; i++) {
		output_1 = body[i].query(output_1);
	}
	output_1 = body[hiddennum - 1].query(output_1);
	return output_1;
	
}

matrix DeepNeural::query(double input_list[]) {

	matrix input_mat(inputnodes, 1);
	input_mat.set_1d(input_list);
	matrix output_1 = body[0].query(input_mat);
	for (int i = 1; i < hiddennum - 1; i++) {
		output_1 = body[i].query(output_1);
	}
	output_1 = body[hiddennum - 1].query(output_1);
	return output_1;

}


/*
��������粿��
*/

con_neural::con_neural(int InRow, int InColumn, int KeRow, int KeColumn, int stride, double learingrate, int mod, bool pooling, int poolrow, int poolcol) {
	this->InColumn = InColumn;
	this->InRow = InRow;
	this->stride = stride;
	this->learingrate = learingrate;
	this->pool = pooling;
	matrix t(KeRow, KeColumn);
	this->kernel = t;
	this->mod = mod;
	this->poolrow = poolrow;
	this->poolcol = poolcol;
}


con_neural::~con_neural() {
}

//�ػ�
matrix con_neural::pooling(matrix target) {
	int row = this->poolrow;
	int col = this->poolcol;
	if (target.row % row != 0 or target.column % col != 0) {
		std::cout << "pooling error" << endl;
		exit(0);
	}
	//ģʽ1���ػ���ģʽ0ƽ���ػ�
	matrix out(target.row / row, target.column / col);
	matrix max_row(target.row / row, target.column / col);
	matrix max_col(target.row / row, target.column / col);
	for (int i = 0; i < out.row; i++) {
		for (int j = 0; j < out.column; j++) {
			double avg = 0;
			double max = 0;
			int coor[2] = { 0,0 };
			for (int ir = 0; ir < row; ir++) {
				for (int ic = 0; ic < col; ic++) {
					avg += target.data[i * row + ir][j * col + ic];
					if (target.data[i * row + ir][j * col + ic] > max) {
						max = target.data[i * row + ir][j * col + ic];
						max_row.data[i][j] = i * row + ir;
						max_col.data[i][j] = j * col + ic;
					}
				}
			}
			if (mod == 1) {
				out.data[i][j] = max;
			}
			else if (mod == 0) {
				out.data[i][j] = avg / (row * col);
			}
		}
	}
	this->max_row.push_back(max_row);
	this->max_col.push_back(max_col);
	return out;
}


//��ѯ���,�����гػ�����û��
matrix con_neural::query(vector<matrix> input) {
	vector<matrix> OutMat;
	this->input = input;
	for (int i = 0; i < input.size(); i++) {
		matrix out = convolution_mat(input[i], this->kernel, this->stride);
		out = ReLU_mat(out);
		if (!pool) {
			OutMat.push_back(out);
			continue;
		}
		//cout <<"out:" << out.row << "||" << out.column << endl;
		this->con_row = out.row;
		this->con_column = out.column;
		out = pooling(out);
		OutMat.push_back(out);
	}
	matrix OneMat = OutMat[0];
	for (int i = 1; i < OutMat.size(); i++) {
		OneMat = add_mat(OneMat, OutMat[i]);
	}
	this->OutPut = OutMat;
	this->OneMat = OneMat;
	return OneMat;
}

//ѵ��
vector<matrix> con_neural::train(matrix error) {
	vector<matrix> error_mat = ErrorDistribute(this->OutPut, this->OneMat, error);
	//�����гػ����Ϳ�ʼά������
	vector<matrix> error_out_pool;
	if (pool) {
		if (this->mod == 1) {
			for (int ErrorNum = 0; ErrorNum < error_mat.size(); ErrorNum++) {
				matrix error_midden(this->con_row, this->con_column, 0);
				for (int i = 0; i < error.row; i++) {
					for (int j = 0; j < error.column; j++) {
						error_midden.data[max_row[ErrorNum].data[i][j]][max_col[ErrorNum].data[i][j]] = error_mat[ErrorNum].data[i][j];
					}
				}
				error_out_pool.push_back(error_midden);
			}
		}
		else if (this->mod == 0) {
			exit(0);
		}
	}
	else {
		vector<matrix> error_out_pool = error_mat;
	}
	//error_mat��ʾ�ػ����������
	vector<matrix> bef_error;
	//�����ϲ�������
	/*
	for (int i = 0; i < error_mat.row; i++) {
		for (int j = 0; j < error.column; j++) {
			er = mul(error_mat.data[i][j], this->kernel);
			for (int ki = 0; ki < this->kernel.row; ki++) {
				for (int kj = 0; kj < this->kernel.column; kj++) {
					bef_error.data[i * stride + ki][j * stride + kj] += er.data[ki][kj];
				}
			}
		}
	}
	*/
	for (int i = 0; i < error_out_pool.size(); i++) {
		matrix e = ErrorCNNDist(error_out_pool[i], this->kernel, this->stride);
		bef_error.push_back(e);
	}


	//���о�����ѵ��
	matrix dkernel(this->kernel.row, this->kernel.column, 0);
	for (int i = 0; i < error_out_pool.size(); i++) {
		error_out_pool[i].show();
		input[i].show();
		matrix ww = convolution_mat( this->input[i], error_out_pool[i], this->stride);
		dkernel = add_mat(dkernel, ww);
	}
	dkernel = mul(learingrate, dkernel);
	this->kernel = add_mat(this->kernel, dkernel);
	return bef_error;
}




//�������������,num��ʾ�м�������㣬EachConNum˳�򴢴�ÿ�������ľ���˸���
/*
ConvolutionNeural::ConvolutionNeural(int cnnum, int EachConNum[][10], bool is_pooling[]) {
	this->ConNum = cnnum;
	for (int i = 0; i < cnnum; i++) {
		vector<con_neural> Each;
		for (int j = 0; j < EachConNum[i][0]; j++) {
			con_neural a(EachConNum[i][1], EachConNum[i][2], EachConNum[i][3], EachConNum[i][4], EachConNum[i][5], EachConNum[i][6], EachConNum[i][7], is_pooling[i], EachConNum[i][8], EachConNum[i][9]);
			Each.push_back(a);
		}
		this->neural.push_back(Each);
	}
}
*/


ConvolutionNeural::ConvolutionNeural(int EachConNum[], double LearningRate, bool is_pooling) {
	for (int i = 0; i < EachConNum[0]; i++) {
		con_neural a(EachConNum[1], EachConNum[2], EachConNum[3], EachConNum[4], EachConNum[5], LearningRate, EachConNum[6], is_pooling, EachConNum[7], EachConNum[8]);
		this->EachConvolu.push_back(a);
	}
}


ConvolutionNeural::~ConvolutionNeural() {
}


//�����˳�����е�n�;���
vector<matrix> ConvolutionNeural::query(matrix input) {
	vector<matrix> out;
	vector<matrix> InputMat;
	InputMat.push_back(input);
	for (int i = 0; i < this->EachConvolu.size(); i++) {
		matrix a = this->EachConvolu[i].query(InputMat);
		out.push_back(a);
	}
	return out;
}

vector<matrix> ConvolutionNeural::query(vector<matrix> input) {
	vector<matrix> out;
	for (int i = 0; i < this->EachConvolu.size(); i++) {
		matrix a = this->EachConvolu[i].query(input);
		out.push_back(a);
	}
	return out;
}

//��λ������ѵ��
vector<matrix> ConvolutionNeural::trian(vector<matrix> error) {
	vector<matrix> out = this->EachConvolu[0].train(error[0]);
	for (int i = 1; i < this->EachConvolu.size(); i++) {
		vector<matrix> a = this->EachConvolu[i].train(error[i]);
		out = add_mat(out, a);
	}
	return out;
}



/*������������磨����ȫ���ӣ�*/
CNN::CNN(int num, int EachConNum[][9], double LearningRate[], bool is_pooling[]) {
	//this->CoNeNum = num;
	vector<ConvolutionNeural> a;
	ConvolutionNeural t(EachConNum[0], LearningRate[0], is_pooling[0]);
	a.push_back(t);
	//this->CoNe.push_back(a);
	//int outnum = EachConNum[0][0];
	for (int i = 1; i < num; i++) {
		ConvolutionNeural b(EachConNum[i], LearningRate[i], is_pooling[i]);
		a.push_back(b);
		//this->CoNe.push_back(a);
		//outnum = outnum * EachConNum[i][0];
	}
	this->Con = a;
}
CNN::CNN() {
}
CNN::~CNN() {
}

vector<matrix> CNN::query(matrix input) {
	//vector<vector<matrix>> EachOutMat;
	vector<matrix> out = this->Con[0].query(input);
	//EachOutMat.push_back(out);
	for (int i = 1; i < this->Con.size(); i++) {
		out = this->Con[i].query(out);
		/*
		vector<matrix> outhidden;
		for (int j = 0; j < out.size(); j++) {
			vector<matrix> hidden = this->CoNe[i][j].query(out[j]);
			for (int t = 0; t < hidden.size(); t++) {
				outhidden.push_back(hidden[t]);
			}
		}
		out = outhidden;
		EachOutMat.push_back(out);
		*/
	}
	//this->EachOutMat = EachOutMat;
	return out;
}


void CNN::train(matrix input, matrix error) {
	//�Ƚ������ݷָ�
	this->query(input);
	vector<matrix> InputError;
	int ErrorNum = this->Con[this->Con.size() - 1].EachConvolu.size();
	for (int i = 0; i < ErrorNum; i++) {
		matrix hid = error.GetRow(i * (error.row / ErrorNum) + 1, (i+1) * (error.row / ErrorNum));
		InputError.push_back(hid);
	}
	for (int i = this->Con.size() - 1; i >= 0; i--) {
		InputError = this->Con[i].trian(InputError);
	}
	/*
	vector<vector<matrix>> error_hid;
	//���һ��ĵ�һ����������˸���
	int EachConNNum = this->CoNe[this->CoNeNum - 1][0].EachConvoluNumber;
	//�������ľ����ʽ
	int OutRow = this->CoNe[this->CoNeNum - 1][0].EachConvolu[0].output.row;
	int OutCol = this->CoNe[this->CoNeNum - 1][0].EachConvolu[0].output.column;
	//��error.row / OutRow���������,��EachConNNum�õ����һ���������
	for (int i = 0; i < error.row / OutRow / EachConNNum; i++) {
		vector<matrix> err;
		for (int j = 0; j < EachConNNum; j++) {
			matrix hid = error.GetRow(i * OutRow * EachConNNum + j * OutRow + 1, i * OutRow * EachConNNum + j * OutRow + OutRow);
			err.push_back(hid);
		}
		error_hid.push_back(err);
	}
	//��ʼѵ�������򴫲�
	//�����һ�㿪ʼ
	for (int i = this->CoNeNum - 1; i > 0; i--) {
		cout << "�����ѵ����" << i << "��:" << endl;
		vector<vector<matrix>> error_h;
		//for (int t = 0; t < this->EachOutMat[i - 1].size(); t++) {
		vector<matrix> error_mat;
		int count = this->CoNe[i - 1][0].EachConvoluNumber;
		for (int t = 0; t < error_hid.size(); t++) {
			cout << "��" << t << "�������Ԫ��" << endl;
			matrix hi = this->CoNe[i][t].trian(this->EachOutMat[i - 1][t], error_hid[t]);
			error_mat.push_back(hi);
			count--;
			if (count == 0) {
				count = this->CoNe[i - 1][0].EachConvoluNumber;
				error_h.push_back(error_mat);
				vector<matrix> empty_error;
				error_mat = empty_error;
			}
		}
		//}
		error_hid = error_h;
	}
	matrix Eror_in = this->CoNe[0][0].trian(input, error_hid[0]);
	*/
}


/*����ȫ���Ӳ�ľ��������*/
ConDeeNetwork::ConDeeNetwork(CNN CNNPart, DeepNeural DNNPart) {
	this->CNNPart = CNNPart;
	this->DNNPart = DNNPart;
}
ConDeeNetwork::~ConDeeNetwork() {
}

matrix ConDeeNetwork::query(matrix input) {
	vector<matrix> CNNOut = CNNPart.query(input);
	matrix out = MergeVectorMatrix(CNNOut);
	this->CNNOut = out;
	matrix Final_out = DNNPart.query(out);
	return Final_out;
}

void ConDeeNetwork::train(vector<matrix> TrainingData, int TrainNum, matrix OutData) {
	for (int i = 0; i < TrainNum; i++) {
		matrix Final_out = this->query(TrainingData[i]);
		matrix Final_error = sub_mat(OutData, Final_out);
		matrix CNNError = DNNPart.train(this->CNNOut, Final_error);
		CNNPart.train(TrainingData[0], CNNError);
	}
}

