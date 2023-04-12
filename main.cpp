#include"neural.h"
#include"tools.h"
#include"matrix.h"

const string TrainDataRoute[] = {
	"D:\\project\\C++\\arduino\\arduino_data\\arduino_data\\eat.csv",
	"D:\\project\\C++\\arduino\\arduino_data\\arduino_data\\good.csv",
	"D:\\project\\C++\\arduino\\arduino_data\\arduino_data\\think.csv",
	"D:\\project\\C++\\arduino\\arduino_data\\arduino_data\\night.csv",
	"D:\\project\\C++\\arduino\\arduino_data\\arduino_data\\come.csv",
	"D:\\project\\C++\\arduino\\arduino_data\\arduino_data\\bye.csv",
};
double TrainDataOut[][3] = {
	{1,0,0},
	{0,1,0},
	{0,0,1},
	{1,1,0},
	{1,0,1},
	{1,1,1},
};


int main() {
	/*
	ofstream fout("b.txt"); //文件输出流对象
	streambuf* pOld = cout.rdbuf(fout.rdbuf());
	*/

	srand(time(0));
	//读取所有训练数据，每个文件3个数据
	vector<matrix> AllData[] = {
		DataGet(3, 40, 12, TrainDataRoute[0]),
		DataGet(3, 40, 12, TrainDataRoute[1]),
		DataGet(3, 40, 12, TrainDataRoute[2]),
		DataGet(3, 40, 12, TrainDataRoute[3]),
		DataGet(3, 40, 12, TrainDataRoute[4]),
		DataGet(3, 40, 12, TrainDataRoute[5]),
	};
	matrix OutData[] = {
		To_Matrix(TrainDataOut[0],3),
		To_Matrix(TrainDataOut[1],3),
		To_Matrix(TrainDataOut[2],3),
		To_Matrix(TrainDataOut[3],3),
		To_Matrix(TrainDataOut[4],3),
		To_Matrix(TrainDataOut[5],3),
	};


	
	//卷积神经网络参数，第一个整数为该层卷积核个数
	int EachNum[][9] = {
			{5, 40, 12, 3, 3, 1, 1, 2, 2},
			{7, 19, 5, 4, 2, 1, 1, 4, 4},
	};
	double LearningRate[] = { 0.05,0.05 };
	//是否池化
	bool is_pool[] = {
		true,
		true,
	};


	CNN CNNPart(2, EachNum, LearningRate, is_pool);
	DeepNeural DNNPart(28, 45, 3, 2, 0.1);
	ConDeeNetwork network(CNNPart, DNNPart);
	int TraningTimes = 300;

	int count = 0;
	for (int i = 0; i < TraningTimes; i++) {
		count = PrintProgress(i, TraningTimes, count);
		network.train(AllData[0], 2, OutData[0]);
		network.train(AllData[1], 2, OutData[1]);
		network.train(AllData[2], 2, OutData[2]);
		network.train(AllData[3], 2, OutData[3]);
		network.train(AllData[4], 2, OutData[4]);
		network.train(AllData[5], 2, OutData[5]);
	}
	for (int i = 0; i < 6; i++) {
		cout << "第" << i << "组" << endl;
		cout << "期望输出矩阵：" << endl;
		OutData[i].show();
		cout << "网络输出矩阵：" << endl;
		network.query(AllData[i][2]);
	}
	

	return 0;
}
