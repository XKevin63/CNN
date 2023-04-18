#include"network.h"


const string TrainDataRoute[] = {
    "D:\\project\\C++\\NeuralNetwork\\data\\eat.csv",
    "D:\\project\\C++\\NeuralNetwork\\data\\good.csv",
    "D:\\project\\C++\\NeuralNetwork\\data\\think.csv",
    "D:\\project\\C++\\NeuralNetwork\\data\\night.csv",
    "D:\\project\\C++\\NeuralNetwork\\data\\come.csv",
    "D:\\project\\C++\\NeuralNetwork\\data\\bye.csv",
};
double TrainDataOut[][6] = {
    {1,0,0,0,0,0},
    {0,1,0,0,0,0},
    {0,0,1,0,0,0},
    {0,0,0,1,0,0},
    {0,0,0,0,1,0},
    {0,0,0,0,0,1},
};


int main() {
#if 1
    srand(time(0));
    //读取训练数据
    int AllNum = 20;
    vector<matrix> AllData[] = {
        DataGet(AllNum, 20, 12, TrainDataRoute[0]),
        DataGet(AllNum, 20, 12, TrainDataRoute[1]),
        DataGet(AllNum, 20, 12, TrainDataRoute[2]),
        DataGet(AllNum, 20, 12, TrainDataRoute[3]),
        DataGet(AllNum, 20, 12, TrainDataRoute[4]),
        DataGet(AllNum, 20, 12, TrainDataRoute[5]),
    };
    matrix OutData[] = {
        To_Matrix(TrainDataOut[0],6),
        To_Matrix(TrainDataOut[1],6),
        To_Matrix(TrainDataOut[2],6),
        To_Matrix(TrainDataOut[3],6),
        To_Matrix(TrainDataOut[4],6),
        To_Matrix(TrainDataOut[5],6),
    };

    //设置神经网络
    ConvolutionFliter CNNPart[] = {
        ConvolutionFliter(5, 1, 3, 3, 1, 0.005, 2, 2),
        ConvolutionFliter(5, 5, 2, 2, 1, 0.005, 2, 2),
    };
    neural DNNPart[] = {
        neural(40, 20, 0.005),
        neural(20, 6, 0.005),
    };
    network work(2, CNNPart, 2, DNNPart);

    //进行神经网络
    int time = 800;
    std::cout << "训练前:" << endl;
    work.query(AllData[0][0]).show();
    work.query(AllData[1][0]).show();
    work.query(AllData[2][0]).show();
    system("pause");
    int TrNum = 15;
    bool trainkey = true;
    for (int i = 0; i < time && trainkey == true; i++) {
        double SumError = 0;
        double cont = 0;
        //trainkey = false;
        std::cout << i + 1 << "/" << time << endl;
        for (int k = 0; k < 6; k++) {
            for (int j = 0; j < TrNum; j++) {
                matrix out = work.query(AllData[k][j]);
                double sum = log(out.data[k][0]);
                SumError += sum;
                work.train(OutData[k]);
                cont++;
            }
        }
        work.SetParm(cont);
        cout << "误差:" << SumError/cont << endl;
        double Trinnn = 0;
        for (int k = 0; k < 6; k++) {
            for (int j = 0; j < TrNum; j++) {
                matrix out = work.query(AllData[k][j]);
                if (out.data[k][0] >= 0.9)
                    Trinnn++;
            }
        }
        cout << "训练集正确率：" << Trinnn / (6 * TrNum) << endl;
        Trinnn = 0;
        for (int k = 0; k < 6; k++) {
            for (int j = TrNum; j < AllNum; j++) {
                matrix out = work.query(AllData[k][j]);
                if (out.data[k][0] >= 0.9)
                    Trinnn++;
            }
        }
        cout << "训练集正确率：" << Trinnn / (6 * (AllNum - TrNum)) << endl;
    }
#endif
}
