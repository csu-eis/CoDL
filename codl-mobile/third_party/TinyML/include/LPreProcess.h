/// @file LPreProcess.h
/// @brief 本文件中声明了一些数据预处理类
/// 
/// Detail:
/// @author Jie Liu Email:coderjie@outlook.com
/// @version   
/// @date 2018/01/31

#ifndef _PREPROCESS_H_
#define _PREPROCESS_H_

#include "LMatrix.h"

typedef LMatrix<double> LDoubleMatrix;     ///< 浮点数矩阵
typedef LMatrix<unsigned int> LUIntMatrix; ///< 无符号整数矩阵



class CMinMaxScaler;

/// @brief 最小最大缩放器
class LMinMaxScaler
{
public:
    /// @brief 构造函数
    LMinMaxScaler(IN double min, IN double max);

    /// @brief 析构函数
    ~LMinMaxScaler();

    /// @brief 训练并进行转换
    /// @param[in] colVec 需要转换的列索引向量(行向量)
    /// @param[inout] matrix 需要转换的矩阵
    /// @return 成功返回true, 失败返回false(参数错误的情况下会返回失败)
    bool FitTransform(IN const LUIntMatrix colVec, INOUT LDoubleMatrix& matrix);

    /// @brief 进行转换
    /// @param[inout] matrix 需要转换的矩阵
    /// @return 成功返回true, 失败返回false(参数错误的情况下会返回失败)
    bool Transform(INOUT LDoubleMatrix& matrix);

private:
    CMinMaxScaler* m_pScaler; ///< 缩放器实现对象

};


/// @brief 对矩阵进行洗牌(随机打乱各个行)
/// @param[in] seed 随机数种子
/// @param[inout] dataMatrix 需要洗牌的矩阵
void DoubleMatrixShuffle(IN unsigned int seed, INOUT LDoubleMatrix& dataMatrix);

/// @brief 同时两个矩阵进行洗牌(随机打乱各个行)
/// 要求两个矩阵具有相同的行数, 洗牌后两个矩阵行对应关系没变
/// @param[in] seed 随机数种子
/// @param[inout] dataMatrix 需要洗牌的矩阵
void DoubleMatrixShuffle(IN unsigned int seed, INOUT LDoubleMatrix& dataMatrixA, INOUT LDoubleMatrix& dataMatrixB);


#endif
