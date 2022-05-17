
#include "../../../Src/LRegression.h"
#include "../../../Src/LCSVIo.h"
#include "../../../Src/LPreProcess.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>

/// @brief 打印矩阵
void MatrixPrint(IN const LDataMatrix& dataMatrix)
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

/// @brief 测试线性回归
void TestLinearRegression()
{
    // 加载糖尿病数据集
    LCSVParser dataCSVParser(L"../../../DataSet/diabetes_data.csv");
    dataCSVParser.SetDelimiter(L' ');
    LDataMatrix xMatrix;
    dataCSVParser.LoadAllData(xMatrix);
    LCSVParser targetCSVParser(L"../../../DataSet/diabetes_target.csv");
    LDataMatrix yVector;
    targetCSVParser.LoadAllData(yVector);


    // 定义线性回归对象
    LLinearRegression linearReg;

    printf("Linear Regression Model Train:\n");
    // 训练模型
    // 计算每一次训练后的得分
    for (unsigned int i = 0; i < 500; i++)
    {
        linearReg.TrainModel(xMatrix, yVector, 0.004f);
        double score = linearReg.Score(xMatrix, yVector);
        printf("Time: %u  Score: %.4f\n", i, score);
    }
}


int main()
{
    TestLinearRegression();

    return 0;
}