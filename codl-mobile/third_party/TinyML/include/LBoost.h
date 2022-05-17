/// @file LBoost.h
/// @brief 本头文件中声明了一些提升算法
/// 
/// Detail:
/// 提升方法的核心思想是针对同一个训练集训练不同的分类器(弱分类器), 然后把这些弱分类器组合起来, 构成一个更强的最终分类器(强分类器)
/// 提升树: 有监督学习, 判别模型, 二元分类, 提升树的弱分类器为决策桩
/// @author Burnell_Liu  Email:burnell_liu@outlook.com
/// @version   
/// @date 22:7:2015
/// @sample

/* 使用提升树的示例代码如下:

    //定义样本矩阵和标签向量
    //以下定义的数据集, 只用一个单层决策树是无法处理的
    //多个单层决策树组成提升树就可解决如下问题
    float sampleList[5 * 2] =
    {
        1.0f, 2.0f,
        2.0f, 1.0f,
        2.0f, 0.0f,
        1.5f, 0.5f,
        1.0f, 1.0f
    };

    float labelList[5 * 1] = 
    {
        LBOOST_MOON,
        LBOOST_MOON,
        LBOOST_MOON,
        LBOOST_SUN,
        LBOOST_SUN
    };


    LBoostMatrix sampleMatrix(5, 2, sampleList);
    LBoostMatrix labelVector(5, 1, labelList);
    LBoostProblem problem(sampleMatrix, labelVector);


    // 定义提升树, 使用样本集进行并且训练
    LBoostTree boostTree;
    boostTree.TrainModel(problem);

    // 计算训练错误率
    LBoostMatrix resultVector;
    boostTree.Predict(sampleMatrix, &resultVector);
    unsigned errorCount = 0;
    for (unsigned int i = 0; i < resultVector.RowLen; i++)
    {
        if (resultVector[i][0] != labelVector[i][0])
            errorCount++;
    }

    float errorRate = (float)errorCount/resultVector.RowLen;
*/


#ifndef _LBOOST_H_
#define _LBOOST_H_

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

/// @brief 二元分类中的一类: 月(阴), 取太极两仪之意
#ifndef LBOOST_MOON
#define LBOOST_MOON -1.0f
#endif

/// @brief 二元分类中的一类: 日(阳), 取太极两仪之意
#ifndef LBOOST_SUN
#define LBOOST_SUN 1.0f
#endif

typedef LMatrix<float> LBoostMatrix;

/// @brief Boost原始问题结构
struct LBoostProblem
{
    /// @brief 构造函数
    /// @param[in] sampleMatrix 样本矩阵, 每一行为一个样本, 每行中的值为样本的特征值
    /// @param[in] classVector 类别向量(列向量), 行数为样本矩阵的行数, 列数为1, 只能为BOOST_MOON或BOOST_SUN
    LBoostProblem(IN const LBoostMatrix& sampleMatrix, IN const LBoostMatrix& classVector)
        : XMatrix(sampleMatrix), YVector(classVector)
    {
    }

    const LBoostMatrix& XMatrix; ///< 样本矩阵
    const LBoostMatrix& YVector; ///< 标签向量(列向量)
};

class CBoostTree;

/// @brief 提升树
class LBoostTree
{
public:
    /// @brief 构造函数
    LBoostTree();

    /// @brief 析构函数
    ~LBoostTree();

    /// @brief 设置最大弱分类器数量, 默认值为40
    /// @param[in] num 弱分类器数量
    void SetMaxClassifierNum(IN unsigned int num);

    /// @brief 训练模型
    /// @param[in] problem 原始问题
    /// @return 返回true表示训练成功, 返回false表示参数数据错误
    bool TrainModel(IN const LBoostProblem& problem);

    /// @brief 使用训练好的模型进行预测(单样本预测)
    /// 请保证需要预测的样本的特征长度和训练样本的特征长度相同
    /// @param[in] sample 需要预测的样本
    /// @return 返回预测结果: BOOST_SUN or BOOST_MOON, 返回0.0表示出错(需要预测的样本出错或者模型没有训练好)
    float Predict(IN const LBoostMatrix& sample);

    /// @brief 使用训练好的模型进行预测(多样本预测)
    /// 请保证需要预测的样本的特征长度和训练样本的特征长度相同
    /// @param[in] sampleMatrix 需要预测的样本矩阵
    /// @param[out] pClassisVector 存储预测结果的向量
    /// @return 返回true表示成功, 返回false表示出错(需要预测的样本出错或者模型没有训练好)
    bool Predict(IN const LBoostMatrix& sampleMatrix, OUT LBoostMatrix* pClassisVector);

private:
    CBoostTree* m_pBoostTree; ///< 提升树实现对象

private:
    // 禁止拷贝构造函数和赋值操作符
    LBoostTree(const LBoostTree&);
    LBoostTree& operator = (const LBoostTree&);
};


#endif