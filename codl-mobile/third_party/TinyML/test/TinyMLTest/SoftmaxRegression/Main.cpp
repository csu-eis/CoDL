

#include "../../../include/LRegression.h"
#include "../../../include/LCSVIo.h"
#include "../../../include/LPreProcess.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>

/// @brief ��ӡ����
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

/// @brief ����SoftMax�ع�
void TestSoftmaxRegression()
{
    // �����β�����ݼ�
    LCSVParser csvParser("../../../DataSet/iris.csv");
    csvParser.SetSkipHeader(true);
    LDataMatrix dataMatrix;
    csvParser.LoadAllData(dataMatrix);

    // ���ݽ�������
    LUIntMatrix colVec(1, 4);
    colVec[0][0] = 0;
    colVec[0][1] = 1;
    colVec[0][2] = 2;
    colVec[0][3] = 3;
    LMinMaxScaler scaler(0.0, 1.0);
    scaler.FitTransform(colVec, dataMatrix);

    // �������ݼ�
    DoubleMatrixShuffle(0, dataMatrix);

    // �����ݼ����Ϊѵ�����Ͳ��Լ�, ���Լ�ռ�ܼ��ϵ�20%
    unsigned int testSize = (unsigned int)(dataMatrix.RowLen * 0.2);
    LDataMatrix trainData;
    LDataMatrix testData;
    dataMatrix.SubMatrix(0, testSize, 0, dataMatrix.ColumnLen, testData);
    dataMatrix.SubMatrix(testSize, dataMatrix.RowLen - testSize, 0, dataMatrix.ColumnLen, trainData);

    // ��ѵ�������Ϊѵ���������Ϻͱ�ǩ��
    LRegressionMatrix trainXMatrix;
    LRegressionMatrix trainYVector;
    trainData.SubMatrix(0, trainData.RowLen, 0, trainData.ColumnLen - 1, trainXMatrix);
    trainData.SubMatrix(0, trainData.RowLen, trainData.ColumnLen - 1, 1, trainYVector);
    // ��ѵ����ǩ�����ӻ�
    LRegressionMatrix trainYMatrix(trainYVector.RowLen, 3, REGRESSION_ZERO);
    for (unsigned int i = 0; i < trainYVector.RowLen; i++)
    {
        unsigned int label = (int)trainYVector[i][0];
        trainYMatrix[i][label] = REGRESSION_ONE;
    }

    // �����Լ����Ϊ�����������Ϻͱ�ǩ��
    LRegressionMatrix testXMatrix;
    LRegressionMatrix testYVector;
    testData.SubMatrix(0, testData.RowLen, 0, testData.ColumnLen - 1, testXMatrix);
    testData.SubMatrix(0, testData.RowLen, testData.ColumnLen - 1, 1, testYVector);
    // �����Ա�ǩ�����ӻ�
    LRegressionMatrix testYMatrix(testYVector.RowLen, 3, REGRESSION_ZERO);
    for (unsigned int i = 0; i < testYVector.RowLen; i++)
    {
        unsigned int label = (int)testYVector[i][0];
        testYMatrix[i][label] = REGRESSION_ONE;
    }

    printf("Softmax Regressio Train Model:");
    // ʹ��ѵ����ѵ��ģ��
    LSoftmaxRegression clf;
    for (int i = 0; i < 150; i++)
    {
        clf.TrainModel(trainXMatrix, trainYMatrix, 0.1);

        // ʹ�ò��Լ�����ģ�͵÷�
        double score = clf.Score(testXMatrix, testYMatrix);
        printf("Time: %u Score: %.2f\n", i, score);
    }

}

int main()
{
    TestSoftmaxRegression();

    system("pause");

    return 0;
}