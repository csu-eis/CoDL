
/// Author:Burnell_Liu Email:674288799@qq.com Date: 2014/08/07
/// Description: 数据聚类
/// 
/// 聚类算法的目标是采集数据, 然后从中找到不同的群组
/// Others: 
/// Function List: 
///
/// History: 
///  1. Date, Author 
///  Modification
///

#ifndef _LDATACLUSTER_H_
#define _LDATACLUSTER_H_

#include "LDataStruct/include/LArray.h"

#include "LMachineLearning/LDataCorretation.h"

#ifndef IN
#define IN
#endif

#ifndef INOUT
#define INOUT
#endif

#ifndef OUT
#define OUT
#endif

typedef LArray<float> LDCDataList; // 聚类数据列表
typedef LArray<LDCDataList> LDCDataMatrix; // 聚类数据矩阵


/// @数据相似度评价方法
enum LDATA_SIMILAR_METHOD
{
    EUCLIDEAN_DISTANCE = 0, ///< 欧几里得距离值评价
    PEARSON_CORRETATIO ///< 皮尔逊相关度评价
};

/// @brief 二叉聚类树结点
struct LBiClusterTNode
{
    LBiClusterTNode()
    {
        Id = 0;
        Distance = 0.0;
        PLChild = 0;
        PRChild = 0;
    }

    int Id; ///< 标识符
    float Distance; ///< 两个孩子的数据距离值(相关度越高, 距离值越小)
    LDCDataList DataList; ///< 数据列表

    LBiClusterTNode* PLChild; ///< 左孩子
    LBiClusterTNode* PRChild; ///< 右孩子
};

/// @brief 二叉聚类树访问者接口
class LBiClustarTreeVisitor
{
public:
    /// @brief 访问聚类树根结点
    /// @param[in] pNode 根结点
    virtual void Visit(IN const LBiClusterTNode* pNode) const = 0;
};

/// @brief 二叉聚类树(分级聚类)
///
/// 分级聚类通过不断的将最为相似的群组两两合并,
/// 来构造出一个群组的层次结构(树形结构, 叶子结点为原始群组)
class LBiClusterTree
{
public:
    LBiClusterTree();
    ~LBiClusterTree();

    /// @brief 设置数据相似度评价方法
    ///
    /// 请在Init()方法前使用该方法, 默认使用皮尔逊相关度评价
    void SetDataSimilerMethod(IN LDATA_SIMILAR_METHOD similarMethod);

    /// @brief 初始化
    void Init();

    /// @brief 对数据进行聚类
    ///
    /// 聚类后可使用Receive()方法来访问聚类后的结果
    /// @param[in] dataMatrix 数据矩阵(要求数据每行长度相等)
    void Cluster(IN const LDCDataMatrix& dataMatrix);

    /// @brief 接受一个访问者
    /// @param[in] visitor 访问者
    void Receive(IN const LBiClustarTreeVisitor& visitor);

private:
    /// @brief 清理二叉聚类树
    /// @param[in] pNode 树根结点指针
    void Clear(IN LBiClusterTNode*& pNode);

private:
    LDATA_SIMILAR_METHOD m_dataSimilarMethod; ///< 数据相似度评价方法
    LDataSimilar* m_pDataSimilar; ///< 数据相似度接口指针
    LBiClusterTNode* m_pRootNode; ///< 根结点
};


typedef LArray<int> LDCResultList; // 聚类结果列表
typedef LArray<LDCResultList> LDCResultMatrix; // 聚类结果矩阵

/// @brief K均值聚类
class LKMeansCluster
{
public:
    LKMeansCluster();
    ~LKMeansCluster();

    /// @brief 设置数据相似度评价方法
    ///
    /// 请在Init()方法前使用该方法, 默认使用皮尔逊相关度评价
    void SetDataSimilerMethod(IN LDATA_SIMILAR_METHOD similarMethod);

    /// @brief 设置K值
    ///
    /// 请在Init()方法前使用该方法, 默认K值为2
    void SetK(IN int k);

    /// @brief 初始化
    void Init();

    /// @brief 对数据进行聚类
    ///
    /// 聚类后可使用Receive()方法来访问聚类后的结果
    /// @param[in] dataMatrix 数据矩阵(要求数据每行长度相等)
    void Cluster(IN const LDCDataMatrix& dataMatrix, OUT LDCResultMatrix& resultMatrix);

private:
    float RandFloat();

private:
    int m_k; ///< K值
    LDCResultMatrix m_resultMatrix; ///< 聚类结果矩阵
    LDataSimilar* m_pDataSimilar; ///< 数据相似度接口指针
    LDATA_SIMILAR_METHOD m_dataSimilarMethod; ///< 数据相似度评价方法
};



#endif

