
#include "include/LRegression.h"
#include "include/LCSVIo.h"
#include "include/LPreProcess.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <chrono>

void MatrixPrint(IN const LDataMatrix& dataMatrix)
{
    printf("Matrix Row: %u  Col: %u\n", dataMatrix.RowLen, dataMatrix.ColumnLen);
    for (unsigned int i = 0; i < dataMatrix.RowLen; i++)
    {
        for (unsigned int j = 0; j < dataMatrix.ColumnLen; j++)
        {
            printf("%.6f  ", dataMatrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void TestLinearRegression(const unsigned int epoches)
{
    LCSVParser dataCSVParser("train_x.csv");
    dataCSVParser.SetDelimiter(',');
    dataCSVParser.SetSkipHeader(true);
    LDataMatrix trainX;
    dataCSVParser.LoadAllData(trainX);

    LCSVParser targetCSVParser("train_y.csv");
    targetCSVParser.SetDelimiter(',');
    targetCSVParser.SetSkipHeader(true);
    LDataMatrix trainY;
    targetCSVParser.LoadAllData(trainY);

    // data must be transformed to [0, 1] to converge
    LMinMaxScaler scaler(0.0, 1.0);
    LUIntMatrix colVec(1, trainX.ColumnLen);
    for (int i = 0; i < trainX.ColumnLen; i++) {
        colVec[0][i] = i;
    }
    scaler.FitTransform(colVec, trainX);
    // MatrixPrint(trainX);

    LLinearRegression linearReg;

    printf("Linear Regression Model Train:\n");
    auto start = std::chrono::system_clock::now();
    for (unsigned int i = 0; i < epoches; i++)
    {
        linearReg.TrainModel(trainX, trainY, 0.0001f);
    }
    auto end = std::chrono::system_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    const double time_cost = ((double) diff.count()) / 1000.0;
    double score = linearReg.Score(trainX, trainY);
    printf("num_features %d, epoches %u, time_cost %.2f ms, score %.4f\n",
            trainX.ColumnLen, epoches, time_cost, score);

    // LCSVParser testParser("test_x.csv");
    // testParser.SetDelimiter(',');
    // testParser.SetSkipHeader(true);
    // LDataMatrix testX;
    // dataCSVParser.LoadAllData(testX);

    // LCSVParser testParser2("test_y.csv");
    // testParser2.SetDelimiter(',');
    // testParser2.SetSkipHeader(true);
    // LDataMatrix testY;
    // testParser2.LoadAllData(testY);

    // scaler.FitTransform(colVec, testX);
    // double score = linearReg.Score(testX, testY);
    // printf("test score: %.4f\n", score);
}

int main(int argc, char *argv[])
{
    unsigned int epoches = 1;
    if (argc - 1 >= 1) {
        epoches = atoi(argv[1]);
    }

    TestLinearRegression(epoches);

    return 0;
}
