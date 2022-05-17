
#include <cstdio>
#include <cstdlib>

#include "../../../Src/LNeuralNetwork.h"

/// @brief 打印矩阵
void MatrixPrint(IN const LNNMatrix& dataMatrix)
{
    printf("Matrix Row: %u  Col: %u\n", dataMatrix.RowLen, dataMatrix.ColumnLen);
    for (unsigned int i = 0; i < dataMatrix.RowLen; i++)
    {
        for (unsigned int j = 0; j < dataMatrix.ColumnLen; j++)
        {
            printf("%.2f  ", dataMatrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main()
{
    LBPNetworkPogology pogology;
    pogology.InputNumber = 2;
    pogology.HiddenLayerNumber = 1;
    pogology.OutputNumber = 1;
    pogology.NeuronsOfHiddenLayer = 2;

    // 初始化神经网络并且设置学习速率
    LBPNetwork bpNetwork(pogology);

    // 定义输入矩阵
    double inputList[8] =
    {
        1.0f, 0.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 1.0f,
    };

    LNNMatrix input(4, 2, inputList);

    // 定义目标输出矩阵
    LNNMatrix targetOutput(4, 1);
    targetOutput[0][0] = 1.0f;
    targetOutput[1][0] = 1.0f;
    targetOutput[2][0] = 0.0f;
    targetOutput[3][0] = 0.0f;

    // 训练1000次
    for (int i = 0; i < 1000; i++)
    {
        bpNetwork.Train(input, targetOutput, 2.1f);
    }

    // 观察训练后的结果
    LNNMatrix output;
    bpNetwork.Active(input, &output);

    MatrixPrint(output);

    system("pause");
}