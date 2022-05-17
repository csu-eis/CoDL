/// @file LSVM.h
/// @brief  支持向量机
/// 
/// Detail:
/// @author Burnell_Liu  
/// @version   
/// @date 6:7:2015


#ifndef _LSVM_H_
#define _LSVM_H_

#include "LDataStruct/LMatrix.h"

#ifndef IN
#define IN
#endif

#ifndef INOUT
#define INOUT
#endif

#ifndef OUT
#define OUT
#endif

typedef LMatrix<float> LSVMMatrix;


/// @brief 核函数接口
class ISVMKernelFunc
{
public:
    /// @brief 析构函数
    virtual ~ISVMKernelFunc() = 0 {}

    /// @brief 转换
    /// 要求向量A和B的长度相同, 并且都是行向量
    /// @param[in] vectorA 向量A(行向量)
    /// @param[in] vectorB 向量B(行向量)
    /// @return 返回向量A, B映射在高纬空间上的向量的内积
    virtual float Translate(IN const LSVMMatrix& vectorA, IN const LSVMMatrix& vectorB) = 0;
};

/// @brief 径向基核函数
/// 默认gamma值为0.1
class LSVMKRBF : public ISVMKernelFunc
{
public:
    /// @brief 构造函数
    /// gamma选的很小则可以将任意的数据映射为线性可分, 但是会产生过拟合的问题
    /// gamma选的很大则高次特征的权重衰减的很快
    /// @param[in] gamma 参数
    explicit LSVMKRBF(IN float gamma);

    /// @brief 析构函数
    ~LSVMKRBF();

    /// @brief 转换
    /// 要求向量A和B的长度相同, 并且都是行向量
    /// @param[in] vectorA 向量A(行向量)
    /// @param[in] vectorB 向量B(行向量)
    /// @return 返回向量A, B映射在高纬空间上的向量内积
    virtual float Translate(IN const LSVMMatrix& vectorA, IN const LSVMMatrix& vectorB);

private:
    float m_gamma; ///< gamma参数

    /*
    以下变量被设为成员变量为优化程序效率目的
    */
    LSVMMatrix m_deltaRow;
    LSVMMatrix m_deltaRowT;
    LSVMMatrix m_k;
};

/// @brief SVM参数结构
struct LSVMParam
{
    float C; ///< 常数C, C越大表明模型对离群点越敏感(即越大的C表明越不希望有离群点)
    unsigned int MaxIterCount; ///< 最大遍历次数
    ISVMKernelFunc* PKernelFunc; ///< 核函数接口指针, 如果为空则不使用核函数
};

/// @brief SVM原始问题结构
struct LSVMProblem
{
    /// @brief 构造函数
    /// @param[in] sampleMatrix 样本矩阵, 每一行为一个样本, 每行中的值为样本的特征值
    /// @param[in] classVector 类别向量(列向量), 行数为样本矩阵的行数, 列数为1, 只能为-1.0f或1.0f
    LSVMProblem(IN const LSVMMatrix& sampleMatrix, IN const LSVMMatrix& classVector)
        : XMatrix(sampleMatrix), YVector(classVector)
    {
    }

    const LSVMMatrix& XMatrix; ///< 样本矩阵
    const LSVMMatrix& YVector; ///< 类别向量(列向量)
};

/// @brief SVM结果
struct LSVMResult
{
    unsigned int IterCount; ///< 遍历次数
    unsigned int SupportVectorNum; ///< 支持向量数
};


class CSVM;

/// @brief 支持向量机
class LSVM
{
public:
    /// @brief 构造函数
    explicit LSVM(IN const LSVMParam& param);

    /// @brief 析构函数
    ~LSVM();

    /// @brief 训练模型
    /// problem参数中的样本矩阵的行数要和标签向量的行数相同
    /// 该函数可能很耗时, 视问题的复杂度而定
    /// @param[in] problem 原始问题
    /// @param[result] result 存储结果 
    /// @return 成功返回true, 失败返回false(参数错误的情况下会返回失败)
    bool TrainModel(IN const LSVMProblem& problem, OUT LSVMResult& result);

    /// @brief 使用训练好的模型进行预测
    /// 请保证需要预测的样本的特征长度和训练样本的特征长度相同
    /// @param[in] sampleSet 需要预测的样本集
    /// @param[out] yVector 存储预测的结果向量(列向量), 值为-1.0 or 1.0
    /// @return 成功返回true, 失败返回false(模型未训练或参数错误的情况下会返回失败)
    bool Predict(IN const LSVMMatrix& sampleSet, OUT LSVMMatrix& yVector);

private:
    CSVM* m_pSVM; ///< SVM实现对象

private:
    // 禁止拷贝构造函数和赋值操作符
    LSVM(const LSVM&);
    LSVM& operator = (const LSVM&);
};

#endif
