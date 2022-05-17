#include "../../../Src/LDecisionTree.h"
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

/// @brief 测试树回归
void TestDecisionTreeRegression()
{
    // 加载糖尿病数据集
    LCSVParser dataCSVParser(L"../../../DataSet/diabetes_data.csv");
    dataCSVParser.SetDelimiter(L' ');
    LDataMatrix xMatrix;
    dataCSVParser.LoadAllData(xMatrix);
    LCSVParser targetCSVParser(L"../../../DataSet/diabetes_target.csv");
    LDataMatrix yVector;
    targetCSVParser.LoadAllData(yVector);

    DoubleMatrixShuffle(0, xMatrix, yVector);

    // 将样本集拆分为训练集和验证集, 20%作为验证集
    unsigned int verifySampleCount = (unsigned int)(xMatrix.RowLen * 0.2);
    LDTMatrix verifyXMatrix;
    LDTMatrix trainXMatrix;
    xMatrix.SubMatrix(0, verifySampleCount, 0, xMatrix.ColumnLen, verifyXMatrix);
    xMatrix.SubMatrix(verifySampleCount, xMatrix.RowLen - verifySampleCount, 0, xMatrix.ColumnLen, trainXMatrix);
    LDTMatrix verifyYVector;
    LDTMatrix trainYVector;
    yVector.SubMatrix(0, verifySampleCount, 0, yVector.ColumnLen, verifyYVector);
    yVector.SubMatrix(verifySampleCount, yVector.RowLen - verifySampleCount, 0, yVector.ColumnLen, trainYVector);

    // 定义特征值分布向量
    double featureN[10] =
    {
        DT_FEATURE_CONTINUUM,
        DT_FEATURE_CONTINUUM,
        DT_FEATURE_CONTINUUM,
        DT_FEATURE_CONTINUUM,
        DT_FEATURE_CONTINUUM,
        DT_FEATURE_CONTINUUM,
        DT_FEATURE_CONTINUUM,
        DT_FEATURE_CONTINUUM,
        DT_FEATURE_CONTINUUM,
        DT_FEATURE_CONTINUUM
    };
    LDTMatrix nVector(1, 10, featureN);


    // 定义线性回归对象
    LDecisionTreeRegression treeReggression;

    printf("Regression Tree Model Train:\n");
    treeReggression.TrainModel(trainXMatrix, nVector, trainYVector);
    treeReggression.PrintTree();

    double score = treeReggression.Score(verifyXMatrix, verifyYVector);
    printf("Model Score: %.2f\n", score);
}


int main()
{
    TestDecisionTreeRegression();

    system("pause");

    return 0;
}
