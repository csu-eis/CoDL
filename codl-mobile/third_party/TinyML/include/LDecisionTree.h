/// @file LDecisionTree.h
/// @brief  决策树头文件
/// 
/// Detail:该文件声明了分类树和回归树
/// @author Jie Liu Email:coderjie@outlook.com
/// @version   
/// @date 2018/05/23

#ifndef _LDECISIONTREE_H_
#define _LDECISIONTREE_H_

#include "LMatrix.h"


typedef LMatrix<double> LDTMatrix;     ///< 决策树矩阵

#ifndef DT_FEATURE_DISCRETE
#define DT_FEATURE_DISCRETE  0.0       ///< 特征值为离散分布
#endif

#ifndef DT_FEATURE_CONTINUUM
#define DT_FEATURE_CONTINUUM 1.0       ///< 特征值为连续分布
#endif


class CDecisionTree;

/// @brief 分类树
class LDecisionTreeClassifier
{
public:
    /// @brief 构造函数
    LDecisionTreeClassifier();

    /// @brief 析构函数
    ~LDecisionTreeClassifier();

    /// @brief 训练模型
    /// 每使用一次该方法, 则生成一个新的模型
    /// @param[in] xMatrix 样本矩阵, 每一行代表一个样本, 每一列代表样本的一个特征
    /// @param[in] nVector 样本特征分布向量(行向量), 每一列代表一个特征的分布, 值只能为DT_FEATURE_DISCRETE和DT_FEATURE_CONTINUUM
    /// @param[in] yVector 样本标签向量(列向量), 每一行代表一个样本, 标签值应为离散值, 不同的值代表不同的类别
    /// @return 成功返回true, 失败返回false(参数错误的情况下会返回失败)
    bool TrainModel(IN const LDTMatrix& xMatrix, IN const LDTMatrix& nVector, IN const LDTMatrix& yVector);

    /// @brief 使用训练好的模型预测数据
    /// @param[in] xMatrix 需要预测的样本矩阵
    /// @param[out] yVector 存储预测的标签向量(列向量)
    /// @return 成功返回true, 失败返回false(模型未训练或参数错误的情况下会返回失败)
    bool Predict(IN const LDTMatrix& xMatrix, OUT LDTMatrix& yVector) const;

    /// @brief 计算模型得分
    /// @param[in] xMatrix 样本矩阵
    /// @param[in] yVector 标签向量(列向量)
    /// @return 得分 值为0.0~1.0, 模型未训练或者参数有误返回-1.0
    double Score(IN const LDTMatrix& xMatrix, IN const LDTMatrix& yVector) const;

    /// @brief 打印树, 用于调试
    void PrintTree() const;

private:
    CDecisionTree* m_pClassifier; ///< 分类树实现对象
};


/// @brief 回归树
class LDecisionTreeRegression
{
public:
    /// @brief 构造函数
    LDecisionTreeRegression();

    /// @brief 析构函数
    ~LDecisionTreeRegression();

    /// @brief 训练模型
    /// 每使用一次该方法, 则生成一个新的模型
    /// 请保持样本数据足够混乱, 因为回归树会连续抽取一部分样本作为验证集进行剪枝操作
    /// @param[in] xMatrix 样本矩阵, 每一行代表一个样本, 每一列代表样本的一个特征
    /// @param[in] nVector 样本特征分布向量(行向量), 每一列代表一个特征的分布, 值只能为DT_FEATURE_DISCRETE和DT_FEATURE_CONTINUUM
    /// @param[in] yVector 样本目标向量(列向量), 每一行代表一个样本
    /// @return 成功返回true, 失败返回false(参数错误的情况下会返回失败)
    bool TrainModel(IN const LDTMatrix& xMatrix, IN const LDTMatrix& nVector, IN const LDTMatrix& yVector);

    /// @brief 使用训练好的模型预测数据
    /// @param[in] xMatrix 需要预测的样本矩阵
    /// @param[out] yVector 存储预测的结果向量(列向量)
    /// @return 成功返回true, 失败返回false(模型未训练或参数错误的情况下会返回失败)
    bool Predict(IN const LDTMatrix& xMatrix, OUT LDTMatrix& yVector) const;

    /// @brief 计算模型得分
    /// @param[in] xMatrix 样本矩阵
    /// @param[in] yVector 目标向量(列向量)
    /// 相关指数R ^ 2, 该值最大值为1, 该值越接近1, 表示回归的效果越好, 如果有错误则返回2.0
    double Score(IN const LDTMatrix& xMatrix, IN const LDTMatrix& yVector) const;

    /// @brief 打印树, 用于调试
    void PrintTree() const;

private:
    CDecisionTree* m_pRegressor; ///< 回归树实现对象
};



#endif