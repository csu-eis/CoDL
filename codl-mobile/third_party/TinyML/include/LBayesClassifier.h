
/// @file LBayesClassifier.h
/// @brief 贝叶斯分类器
/// 
/// Detail: 
/// 贝叶斯分类器为: 生成模型, 有监督分类, 多元分类
/// 本文件实现的贝叶斯分类器的样本特征值数据可为连续型数据或离散型数据, 但不可混合使用, 即样本数据不可同时出现离散型数据和连续型数据
/// 连续型数据如: 人的身高, 离散型数据如: 文本分类中的单词数量
/// 贝叶斯分类器主要依据贝叶斯公式: P(Y|X) = P(Y, X)/P(X) = P(X, Y)/P(X) = P(X|Y) * P(Y)/P(X)
/// @author Burnell Liu Email:burnell_liu@outlook.com
/// @version   
/// @date 2015:12:17
/// @sample


/*  使用贝叶斯分类器的示例代码如下:

// 定义样本矩阵
// 每一行代表一个人
// 第一列为身高(英尺), 第二列为体重(磅), 第3列为脚掌(英寸)
int dataList[24] = 
{
    600, 180, 12,
    592, 190, 11,
    558, 170, 12,
    592, 165, 10,
    500, 100, 6,
    550, 150, 8,
    542, 130, 7,
    575, 150, 9
};
LBayesMatrix sampleMatrix(8, 3, dataList);

// 定义样本的类别向量
// 1: 男
// 0: 女
int classList[8] = 
{
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0
};
LBayesMatrix classVector(8, 1, classList);

// 定义贝叶斯原始问题, 指明样本数据为连续值
LBayesProblem problem(sampleMatrix, classVector, BAYES_FEATURE_CONTINUS);

// 定义贝叶斯分类器, 并且训练
LBayesClassifier classifier;
classifier.TrainModel(problem);


// 输入新样本, 并预测新样本的类别
LBayesMatrix newSample(1, 3);
newSample[0][0] = 600; newSample[0][1] = 130; newSample[0][2] = 8;

int predictValue = -1; 
classifier.Predict(newSample, &predictValue);

*/

#ifndef _LBAYESCLASSIFIER_H_
#define _LBAYESCLASSIFIER_H_

#include "LDataStruct/LMatrix.h"

typedef LMatrix<int> LBayesMatrix; ///< 贝叶斯矩阵

/// @brief 贝叶斯特征值分布
enum LBayesFeatureDistribution
{
    BAYES_FEATURE_DISCRETE = 1, ///< 离散
    BAYES_FEATURE_CONTINUS = 2, ///< 连续
};

/// @brief 贝叶斯原始问题结构
/// 特征数据为非离散(连续)表示值服从高斯分布
struct LBayesProblem
{
    /// @brief 构造函数
    /// @param[in] sampleMatrix 样本矩阵, 每一行为一个样本, 每行中的值为样本的特征值
    /// @param[in] classVector 类别向量(列向量), 行数为样本矩阵的行数, 列数为1
    /// @param[in] dataType 特征值分布
    LBayesProblem(
        IN const LBayesMatrix& sampleMatrix, 
        IN const LBayesMatrix& classVector, 
        IN LBayesFeatureDistribution distribution)
        : XMatrix(sampleMatrix), YVector(classVector), FeatureDistribution(distribution)
    {
    }

    const LBayesMatrix& XMatrix; ///< 样本矩阵
    const LBayesMatrix& YVector; ///< 类别向量(列向量)
    const LBayesFeatureDistribution FeatureDistribution; ///< 贝叶斯特征值分布
};


class CBayesClassifier;

/// @brief 贝叶斯分类器接口类
class LBayesClassifier
{
public:
    /// @brief 构造函数
    LBayesClassifier();

    /// @brief 析构函数
    ~LBayesClassifier();

    /// @brief 训练模型
    /// @param[in] problem 贝叶斯问题
    /// @return 成功返回true, 失败返回false, 参数错误的情况下会返回false
    bool TrainModel(IN const LBayesProblem& problem);

    /// @brief 使用训练好的模型进行预测
    /// 请保证需要预测的样本的特征长度和训练样本的特征长度相同
    /// @param[in] sample 需要预测的样本
    /// @param[out] pClassValue 存储预测结果, 不能为0
    /// @return 成功预测返回true, 失败返回false, 参数错误或模型未训练的情况下会返回false
    bool Predict(IN const LBayesMatrix& sample, OUT int* pClassValue);


private:
    CBayesClassifier* m_pBayesClassifier; ///< 贝叶斯分类器实现对象

private:
    // 禁止拷贝构造函数和赋值操作符
    LBayesClassifier(const LBayesClassifier&);
    LBayesClassifier& operator = (const LBayesClassifier&);
};


#endif