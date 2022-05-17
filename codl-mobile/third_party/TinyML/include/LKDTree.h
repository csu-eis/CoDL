/// @file LKDTree.h
/// @brief k-d树
/// 
/// Detail: k-d树(k-dimensional树的简称), 是一种分割k维数据空间的数据结构
/// 主要应用于多维空间关键数据的搜索（如：范围搜索和最近邻搜索）。
/// @author Burnell_Liu  
/// @version   
/// @date 31:7:2015

#ifndef _LKDTREE_H_
#define _LKDTREE_H_

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

typedef LMatrix<float> LKDTreeMatrix; ///< KD树矩阵
typedef LMatrix<int> LKDTreeList; ///< 列表(行向量)

class CKDTree;

/// @brief KD树
class LKDTree
{
public:
    /// @brief 构造函数
    LKDTree();

    /// @brief 析构函数
    ~LKDTree();

    /// @brief 构造KD树
    /// @param[in] dataSet 数据集, 要求数据集多行多列
    void BuildTree(IN const LKDTreeMatrix& dataSet);

    /// @brief 在数据集中搜索与指定数据最邻近的数据索引
    /// @param[in] data 源数据(行向量)
    /// @return 成功返回最邻近的数据索引, 失败返回-1
    int SearchNearestNeighbor(IN const LKDTreeMatrix& data);

    /// @brief 在数据集中搜索与指定数据最邻近的K个数据索引
    /// @param[in] data 源数据(行向量)
    /// @param[in] k 需要搜索的最邻近的个数(k要求大于0的整数)
    /// @param[out] indexList 存储最邻近数据索引的列表(行向量, 1 * k), 从近到远
    /// @return 成功返回true, 失败返回false
    bool SearchKNearestNeighbors(IN const LKDTreeMatrix& data, IN unsigned int k, OUT LKDTreeList& indexList);

private:
    CKDTree* m_pKDTree; ///< KD树实现对象
};


#endif