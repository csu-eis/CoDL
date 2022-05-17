
#include "../include/LPreProcess.h"

#include <cstdlib>
#include <vector>
using std::vector;

/// @brief 列值范围
struct CColumnRange
{
    unsigned int Idx; ///< 列索引
    double Min;       ///< 最小值
    double Max;       ///< 最大值
    double Dis;       ///< 最大最小值差, 应该大于0.0
};


/// @brief 最小最大缩放器
class CMinMaxScaler
{
public:
    /// @brief 构造函数
    CMinMaxScaler(IN double min, IN double max)
    {
        m_targetMin = min;
        m_targetMax = max;
        m_targetDis = max - min;
    }

    /// @brief 析构函数
    ~CMinMaxScaler()
    {

    }

    /// @brief 训练并进行转换
    bool FitTransform(IN const LUIntMatrix colVec, INOUT LDoubleMatrix& matrix)
    {
        // 检查最大最小值设置是否有问题
        if (m_targetDis <= 0.0)
            return false;

        // 检查参数
        if (colVec.ColumnLen < 1)
            return false;
        if (colVec.RowLen != 1)
            return false;
        if (matrix.ColumnLen < 1)
            return false;
        if (matrix.RowLen < 2)
            return false;

        m_rangeList.clear();
        m_columnLen = matrix.ColumnLen;

        // 针对每一个需要转换的列
        for (unsigned int i = 0; i < colVec.ColumnLen; i++)
        {
            // 数据有误
            unsigned int colIdx = colVec[0][i];
            if (colIdx >= matrix.ColumnLen)
            {
                m_rangeList.clear();
                return false;
            }

            // 找到每一列的最大最小值
            CColumnRange colRange;
            colRange.Idx = colIdx;
            colRange.Min = matrix[0][colIdx];
            colRange.Max = matrix[0][colIdx];

            for (unsigned int row = 0; row < matrix.RowLen; row++)
            {
                if (matrix[row][colIdx] < colRange.Min)
                    colRange.Min = matrix[row][colIdx];
                if (matrix[row][colIdx] > colRange.Max)
                    colRange.Max = matrix[row][colIdx];
            }
            colRange.Dis = colRange.Max - colRange.Min;

            // 数据有误, 该列值都相同
            if (colRange.Dis <= 0.0)
            {
                m_rangeList.clear();
                return false;
            }

            m_rangeList.push_back(colRange);

        }

        return this->Transform(matrix);

    }

    /// @brief 进行转换
    bool Transform(INOUT LDoubleMatrix& matrix)
    {
        // 检查最大最小值设置是否有问题
        if (m_targetDis <= 0.0)
            return false;

        // 检查有无训练好缩放器
        if (m_rangeList.size() < 1)
            return false;

        // 检查参数
        if (matrix.ColumnLen != m_columnLen)
            return false;
        if (matrix.RowLen < 1)
            return false;

        // 针对每一个需要缩放的列
        for (auto iter = m_rangeList.begin(); iter != m_rangeList.end(); iter++)
        {
            // 针对每一个需要缩放的行
            for (unsigned int row = 0; row < matrix.RowLen; row++)
            {
                double prop = (matrix[row][iter->Idx] - iter->Min) / iter->Dis;

                double scaled = prop * m_targetDis + m_targetMin;

                matrix[row][iter->Idx] = scaled;

            }
        }

        return true;
    }

private:
    double m_targetMin; ///< 目标最小值
    double m_targetMax; ///< 目标最大值
    double m_targetDis; ///< 目标最大最小值差, 应该大于0.0

    unsigned int m_columnLen; ///< 数据列长度

    vector<CColumnRange> m_rangeList; ///< 每列值范围列表
};

LMinMaxScaler::LMinMaxScaler(IN double min, IN double max)
{
    m_pScaler = nullptr;
    m_pScaler = new CMinMaxScaler(min, max);
}

LMinMaxScaler::~LMinMaxScaler()
{
    if (nullptr != m_pScaler)
    {
        delete m_pScaler;
        m_pScaler = nullptr;
    }
}

bool LMinMaxScaler::FitTransform(IN const LUIntMatrix colVec, INOUT LDoubleMatrix& matrix)
{
    return m_pScaler->FitTransform(colVec, matrix);
}

bool LMinMaxScaler::Transform(INOUT LDoubleMatrix& matrix)
{
    return m_pScaler->Transform(matrix);
}

/// @brief 产生随机整数
/// @param[in] min 随机整数的最小值(包含该值)
/// @param[in] max 随机整数的最大值(包含该值)
/// @return 随机整数
static int RandInt(int min, int max)
{
    if (min > max)
    {
        int t = max; max = min; min = t;
    }

    return rand() % (max - min + 1) + min;
}

void DoubleMatrixShuffle(IN unsigned int seed, INOUT LDoubleMatrix& dataMatrix)
{
    // 设置随机种子
    srand(seed);

    /// 每次从未处理的数据中随机取出一个数字，然后把该数字放在数组的尾部，即数组尾部存放的是已经处理过的数字
    for (int i = 0; i < (int)dataMatrix.RowLen; i++)
    {
        int k = RandInt(0, i);

        for (int j = 0; j < (int)dataMatrix.ColumnLen; j++)
        {
            double t = dataMatrix[k][j];
            dataMatrix[k][j] = dataMatrix[i][j];
            dataMatrix[i][j] = t;
        }

    }
}

void DoubleMatrixShuffle(IN unsigned int seed, INOUT LDoubleMatrix& dataMatrixA, INOUT LDoubleMatrix& dataMatrixB)
{
    if (dataMatrixA.RowLen != dataMatrixB.RowLen)
        return;


    // 设置随机种子
    srand(seed);

    /// 每次从未处理的数据中随机取出一个数字，然后把该数字放在数组的尾部，即数组尾部存放的是已经处理过的数字
    for (int i = 0; i < (int)dataMatrixA.RowLen; i++)
    {
        int k = RandInt(0, i);

        for (int j = 0; j < (int)dataMatrixA.ColumnLen; j++)
        {
            double t = dataMatrixA[k][j];
            dataMatrixA[k][j] = dataMatrixA[i][j];
            dataMatrixA[i][j] = t;
        }

        for (int j = 0; j < (int)dataMatrixB.ColumnLen; j++)
        {

            double t = dataMatrixB[k][j];
            dataMatrixB[k][j] = dataMatrixB[i][j];
            dataMatrixB[i][j] = t;
        }
    }
}