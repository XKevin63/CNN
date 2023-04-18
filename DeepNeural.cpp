#include"DeepNeural.h"
#include<math.h>


/*ȫ�����񾭵�Ԫ*/
neural::neural(int inputnodes, int hiddennodes, double learningrate) {
	this->inputnodes = inputnodes;
	this->hiddennodes = hiddennodes;
	this->learningrate = learningrate;
	matrix wio(hiddennodes, inputnodes);
	this->wio = wio;
	matrix bi(hiddennodes, 1, 0.5);
	matrix db(hiddennodes, 1, 0);
	this->bias = bi;
	this->db = db;
	matrix dwio(hiddennodes, inputnodes, 0);
	this->dwio = dwio;
}

matrix neural::query(matrix input) {
	this->input = input;
	matrix out = mul_mat(this->wio, input);
	out = add_mat(this->bias, out);
	this->OutBeforActive = out;
	out = ReLU_mat(out);
	this->OutAfterActive = out;
	return out;
}

matrix neural::train(matrix error) {
	//��ƫ���뼤�����,����С��0,����0
	matrix zero_error = error;
	
	for (int i = 0; i < this->OutBeforActive.row; i++) 
		if (this->OutBeforActive.data[i][0] <= 0) 
			zero_error.data[i][0] = 0;
	
	//�������
	matrix out = mul_mat(transposition(this->wio), zero_error);
	//���ѵ���ݶ�
	//cout << "���" << endl;
	//error.show();
	//this->OutBeforActive.show();
	//matrix dwad = mul_mat(zero_error, transposition(this->input));
	//dwad.show();
	//ƫ��
	//matrix db = mul(this->learningrate, zero_error);
	//this->bias = sub_mat(this->bias, db);
	this->db = add_mat(this->db, zero_error);
	//Ȩ��
	matrix dw = mul_mat(zero_error, transposition(this->input));
	//dw.show();
	//this->wio = sub_mat(this->wio, dw);
	this->dwio = add_mat(this->dwio, dw);
	//�������ǰһ�㼤����ƫ��
	return out;
}

bool neural::Setdwio(double num) {
	this->wio = sub_mat(this->wio, mul(this->learningrate / num, this->dwio));
	matrix dwio(hiddennodes, inputnodes, 0);
	this->dwio = dwio;
	this->bias = sub_mat(this->bias, mul(this->learningrate / num, this->db));
	matrix db(hiddennodes, 1, 0);
	this->db = db;
	return true;
}



/*softmax�����*/
softmax::softmax(int inputnodes) {
	this->inputnodes = inputnodes;
}

matrix softmax::query(matrix input) {
	this->intput = input;
	if (input.column != 1) {
		cout << "softmax error" << '\n';
		exit(0);
	}
	matrix out(input.row, 1);
	double sum = 0;
	for (int i = 0; i < input.row; i++) {
		out.data[i][0] = exp(input.data[i][0]);
		sum += out.data[i][0];
	}
	for (int i = 0; i < input.row; i++) {
		out.data[i][0] = out.data[i][0] / sum;
	}
	this->output = out;
	return out;
}


//��������Լ���������ƫ��
matrix softmax::train(matrix target) {
	//ʹ�ý����غ���
	//�����Ƿ�ֻ��һ��1
	//double sum = target.data[0][0] * log(this->output.data[0][0]);
	double t = target.data[0][0];
	int key = 0;
	for (int i = 1; i < target.row; i++) {
		if (target.data[i][0] == 1) {
			key = i;
		}
		//sum -= target.data[i][0]*log(this->output.data[i][0]);
		t -= target.data[i][0];
	}
	//cout << "���Ϊ��" << sum << '\n';
	//if (t * t != 1) {
		//cout << "softmax error" << endl;
		//exit(0);
	//}
	//������
	matrix out(target.row, target.column);
	for (int i = 0; i < target.row; i++) {
		if (i == key)
			out.data[i][0] = this->output.data[i][0] - 1;
		else
			out.data[i][0] = this->output.data[i][0];
	}
	return out;
}




/*ȫ���Ӳ���*/
FullyConnectedLayer::FullyConnectedLayer(int LayerNum, neural NeuralList[]) {
	for (int i = 0; i < LayerNum; i++) 
		this->BpPart.push_back(NeuralList[i]);
	this->SoftPart = softmax(NeuralList[LayerNum - 1].hiddennodes);
}
FullyConnectedLayer::FullyConnectedLayer() {
}

matrix FullyConnectedLayer::query(matrix input) {
	matrix out = this->BpPart[0].query(input);
	for (int i = 1; i < this->BpPart.size(); i++) {
		out = this->BpPart[i].query(out);
	}
	out = this->SoftPart.query(out);
	return out;
}

matrix FullyConnectedLayer::train(matrix target) {
	matrix dL_dz = this->SoftPart.train(target);
	for (int i = this->BpPart.size() - 1; i >= 0; i--) 
		dL_dz = this->BpPart[i].train(dL_dz);
	return dL_dz;
}


bool FullyConnectedLayer::SetWio(double num) {
	for (int i = 0; i < this->BpPart.size(); i++) {
		this->BpPart[i].Setdwio(num);
	}
	return true;
}