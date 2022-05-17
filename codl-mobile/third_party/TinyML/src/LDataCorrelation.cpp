
#include "LDataCorrelation.h"

#include <math.h>


float LEuclideanDistance::Calculate(IN const LDCVector& dataVecA, IN const LDCVector& dataVecB)
{
    if (dataVecA.RowLen != dataVecB.RowLen)
        return 0;

    if (dataVecA.RowLen != 1)
        return 0;

    if (dataVecA.ColumnLen != dataVecB.ColumnLen)
        return 0;

    float sqrSum = 0.0f;
    for (unsigned int i = 0; i < dataVecA.ColumnLen; i++)
    {
        float dif = dataVecA[0][i] - dataVecB[0][i];
        sqrSum += dif * dif;
    }

    return 1.0f/(1.0f + sqrtf(sqrSum));
}

float LPearsonCorrelation::Calculate(IN const LDCVector& dataVecA, IN const LDCVector& dataVecB)
{
    if (dataVecA.RowLen != dataVecB.RowLen)
        return 0;

    if (dataVecA.RowLen != 1)
        return 0;

    if (dataVecA.ColumnLen != dataVecB.ColumnLen)
        return 0;

    unsigned int length = dataVecA.ColumnLen;
    if (length < 1)
        return 0;

    float sumA = 0.0f;
    float sumB = 0.0f;
    float sqrSumA = 0.0f;
    float sqrSumB = 0.0f;
    float proSum = 0.0f;
    for (unsigned int i = 0; i < length; i++)
    {
        const float& a = dataVecA[0][i];
        const float& b = dataVecB[0][i];

        sumA += a;
        sumB += b;

        sqrSumA += a * a;
        sqrSumB += b * b;

        proSum += a * b;
    }

    // 计算皮尔逊相关系数
    float num = proSum - (sumA * sumB/length);
    float den = sqrtf((sqrSumA - sumA * sumA/length) * (sqrSumB - sumB * sumB/length));

    if (den == 0)
        return 0;

    return num/den;
}

float LTanimotoCoefficient::Calculate(IN const LDCVector& dataVecA, IN const LDCVector& dataVecB)
{
    if (dataVecA.RowLen != dataVecB.RowLen)
        return 0;

    if (dataVecA.RowLen != 1)
        return 0;

    if (dataVecA.ColumnLen != dataVecB.ColumnLen)
        return 0;

    int coutA = 0;
    int countB = 0;
    int countAB = 0;
    for (unsigned int i = 0; i < dataVecA.ColumnLen; i++)
    {
        if (dataVecA[0][i] == m_sameValue)
            coutA++;
        if (dataVecB[0][i] == m_sameValue)
            countB++;
        if (dataVecA[0][i] == m_sameValue && dataVecB[0][i] == m_sameValue)
            countAB++;

    }

    if (countAB == 0)
        return 0.0f;

    return (float)countAB/(float)(coutA + countB - countAB);
}

LTanimotoCoefficient::LTanimotoCoefficient()
{
    m_sameValue = 1.0f;
}

LTanimotoCoefficient::~LTanimotoCoefficient()
{

}

void LTanimotoCoefficient::SetSameValue(IN float sameValue)
{
    m_sameValue = sameValue;
}