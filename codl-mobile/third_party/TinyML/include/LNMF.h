
/// @file LNMF.h
/// @brief 非负矩阵因式分解
/// 将一个非负矩阵分解为两个非负矩阵的乘, V = W * H
/// V为原始矩阵, W为基矩阵, H为系数矩阵
/// 基矩阵的列数和系数矩阵的行数为R
/// @author Burnell Liu Email:burnell_liu@outlook.com
/// @version   
/// @date 2015:12:17
/// @sample

/*  使用NMF的示例代码如下

// 定义原始矩阵
float dataList[4] = 
{
    1.0f, 2.0f,
    3.0f, 4.0f
};
LNMFMatrix V(2, 2, dataList);

// 定义原始问题, 设置R为2, 迭代次数为50
LNMFProblem problem(V, 2, 50);

// 进行因式分解
LNMFMatrix W;
LNMFMatrix H;

LNMF nmf;
nmf.Factoring(problem, &W, &H);
*/

#ifndef _LNMF_H_
#define _LNMF_H_

#include "LDataStruct/LMatrix.h"

typedef LMatrix<float> LNMFMatrix; ///< NMF矩阵

/// @brief NMF问题结构
struct LNMFProblem
{
    /// @brief 构造函数
    /// @param[in] v 原始矩阵, 原始矩阵中不能有负数
    /// @param[in] r 基矩阵的列数(系数矩阵的行数)
    /// @param[in] iterCount 迭代次数
    LNMFProblem(IN const LNMFMatrix& v, IN unsigned int r, IN unsigned int iterCount)
        : V(v), R(r), IterCount(iterCount)
    {

    }

    const LNMFMatrix& V; ///< 原始矩阵
    const unsigned int R; ///< 基矩阵的列数(系数矩阵的行数)
    const unsigned int IterCount; ///< 迭代次数
};

/// @brief 非负矩阵因式分解
class LNMF
{
public:
    /// @brief 构造函数
    LNMF();

    /// @brief 析构函数
    ~LNMF();

    /// @brief 因式分解
    /// @param[in] problem 原始问题, 原始矩阵中不能有负数
    /// @param[out] pW 存储分解后的基矩阵, 不能为0
    /// @param[out] pH 存储分解后的系数矩阵, 不能为0
    /// @return 成功返回true, 失败返回false, 参数有误会失败
    bool Factoring(IN const LNMFProblem& problem, OUT LNMFMatrix* pW, OUT LNMFMatrix* pH);
};

#endif