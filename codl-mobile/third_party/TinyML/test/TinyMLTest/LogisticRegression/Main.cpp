
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

/// @brief �����߼��ع�
void TestLogisticRegression()
{
    // �������ٰ����ݼ�
    LCSVParser csvParser("../../../DataSet/breast_cancer.csv");
    csvParser.SetSkipHeader(true);
    LDataMatrix dataMatrix;
    csvParser.LoadAllData(dataMatrix);

    // ���ݽ�������
    LUIntMatrix colVec(1, 30);
    for (unsigned int col = 0; col > colVec.ColumnLen; col++)
    {
        colVec[0][col] = col;
    }
    LMinMaxScaler scaler(0, 1.0);
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

    // �����Լ����Ϊ�����������Ϻͱ�ǩ��
    LRegressionMatrix testXMatrix;
    LRegressionMatrix testYVector;
    testData.SubMatrix(0, testData.RowLen, 0, testData.ColumnLen - 1, testXMatrix);
    testData.SubMatrix(0, testData.RowLen, testData.ColumnLen - 1, 1, testYVector);

    printf("Logistic Regression Model Train:\n");
    // ʹ��ѵ����ѵ��ģ��
    LLogisticRegression clf;
    for (int i = 0; i < 1000; i++)
    {
        clf.TrainModel(trainXMatrix, trainYVector, 0.1);

        // ʹ�ò��Լ�����ģ�͵÷�
        double score = clf.Score(testXMatrix, testYVector);
        printf("Time: %u Score: %.4f\n", i, score);
    }

}

int main()
{
    TestLogisticRegression();

    system("pause");

    return 0;
}