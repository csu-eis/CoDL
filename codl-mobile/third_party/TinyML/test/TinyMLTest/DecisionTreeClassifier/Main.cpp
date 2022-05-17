
#include "../../../Src/LDecisionTree.h"
#include "../../../Src/LCSVIo.h"
#include "../../../Src/LPreProcess.h"

#include <cstdio>
#include <cstdlib>


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

int main()
{
    // 加载数据集
    LCSVParser csvParser(L"../../../DataSet/iris.csv");
    csvParser.SetSkipHeader(true);
    LDataMatrix dataMatrix;
    csvParser.LoadAllData(dataMatrix);

    // 打乱数据集
    DoubleMatrixShuffle(0, dataMatrix);

    // 将数据集拆分为训练集和测试集, 测试集占总集合的20%
    unsigned int testSize = (unsigned int)(dataMatrix.RowLen * 0.2);
    LDTMatrix testData;
    LDTMatrix trainData;
    dataMatrix.SplitRow(testSize, testData, trainData);

    // 将训练集拆分为训练样本集合和标签集
    LDTMatrix trainXMatrix;
    LDTMatrix trainYVector;
    trainData.SplitCloumn(trainData.ColumnLen - 1, trainXMatrix, trainYVector);

    // 将测试集拆分为测试样本集合和标签集
    LDTMatrix testXMatrix;
    LDTMatrix testYVector;
    testData.SplitCloumn(testData.ColumnLen - 1, testXMatrix, testYVector);


    // 定义特征值分布向量
    double featureN[4] =
    {
        DT_FEATURE_CONTINUUM,
        DT_FEATURE_CONTINUUM,
        DT_FEATURE_CONTINUUM,
        DT_FEATURE_CONTINUUM
    };
    LDTMatrix nVector(1, 4, featureN);

    // 使用训练集训练模型
    LDecisionTreeClassifier clf;
    clf.TrainModel(trainXMatrix, nVector, trainYVector);
    clf.PrintTree();

    // 使用测试集计算模型得分
    double score = clf.Score(testXMatrix, testYVector);

    printf("Decision Tree Classifier Score: %.2f\n", score);

    system("pause");

    return 0;
}