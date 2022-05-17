

#include "LDecisionTree.h"

#include <vector>
using std::vector;
#include <string>
using std::string;
#include <set>
using std::set;
#include <map>
using std::map;

#ifndef CLASSIFIER_TREE
#define CLASSIFIER_TREE 0
#endif
#ifndef REGRESSION_TREE
#define REGRESSION_TREE 1
#endif


/// @brief 决策树树节点
struct CDecisionTreeNode
{
    unsigned int CheckColumn;           ///< 需要检验的列索引, 叶子结点该值无意义
    double CheckValue;                  ///< 检验值, 为了使结果为true, 当前列必须匹配的值(如果是离散值则必须相等才为true, 如果是连续值则大于等于为true), 叶子结点该值无意义
    double FeatureDis;                  ///< 特征分布, 可以为DT_FEATURE_DISCRETE或DT_FEATURE_CONTINUUM, 叶子结点该值无意义

    CDecisionTreeNode* PTrueChildren;   ///< 条件为true的分支结点, 叶子结点该值为nullptr
    CDecisionTreeNode* PFalseChildren;  ///< 条件为false的分支结点, 叶子结点该值为nullptr

    double TargetValue;                 ///< 该结点代表的目标值
    double LossValue;                   ///< 该结点的损失值
};

/// @brief 复制决策树
/// @param[in] pNode 需要复制的树结点
/// @return 复制后的新结点
static CDecisionTreeNode* DecisionTreeCopy(IN const CDecisionTreeNode* pNode)
{
    if (pNode == nullptr)
        return nullptr;

    CDecisionTreeNode* pNewNode = new CDecisionTreeNode();
    (*pNewNode) = (*pNode);

    pNewNode->PTrueChildren = nullptr;
    pNewNode->PFalseChildren = nullptr;

    pNewNode->PTrueChildren = DecisionTreeCopy(pNode->PTrueChildren);
    pNewNode->PFalseChildren = DecisionTreeCopy(pNode->PFalseChildren);

    return pNewNode;

}

/// @brief 决策树损失值
/// @param[in] pNode 需要计算损失值的结点
/// @param[out] lossValue 存储损失值
/// @param[out] leafCount 存储叶子结点数量
static void DecisionTreeLossValue(IN CDecisionTreeNode* pNode, OUT double& lossValue, OUT unsigned int& leafCount)
{
    if (pNode == nullptr)
        return;

    if (pNode->PTrueChildren != nullptr)
        DecisionTreeLossValue(pNode->PTrueChildren, lossValue, leafCount);
    if (pNode->PFalseChildren != nullptr)
        DecisionTreeLossValue(pNode->PFalseChildren, lossValue, leafCount);

    if (pNode->PTrueChildren == nullptr &&
        pNode->PFalseChildren == nullptr)
    {
        lossValue += pNode->LossValue;
        leafCount += 1;
    }

}

/// @brief 后序遍历内部结点
/// @param[in] pNode 需要遍历的结点
/// @param[out] nodeList 存储内部结点
static void DecisionTreeInsideNodes(IN CDecisionTreeNode* pNode, OUT vector<CDecisionTreeNode*>& nodeList)
{
    if (pNode == nullptr)
        return;

    if (pNode->PTrueChildren != nullptr)
        DecisionTreeInsideNodes(pNode->PTrueChildren, nodeList);
    if (pNode->PFalseChildren != nullptr)
        DecisionTreeInsideNodes(pNode->PFalseChildren, nodeList);

    if (pNode->PTrueChildren != nullptr &&
        pNode->PFalseChildren != nullptr)
        nodeList.push_back(pNode);

}

/// @brief 决策树树预测
/// @param[in] pNode 预测结点
/// @param[in] xMatrix 需要预测的样本矩阵
/// @param[in] idx 需要预测的样本索引
/// @param[out] yVector 存储预测结果
static void DecisionTreePredicty(
    IN CDecisionTreeNode* pNode,
    IN const LDTMatrix& xMatrix,
    IN unsigned int idx,
    OUT LDTMatrix& yVector)
{
    if (pNode == nullptr)
        return;

    // 叶子结点
    if (pNode->PTrueChildren == nullptr &&
        pNode->PFalseChildren == nullptr)
    {
        yVector[idx][0] = pNode->TargetValue;
        return;
    }

    // 分支结点
    double currentValue = xMatrix[idx][pNode->CheckColumn];
    double checkVallue = pNode->CheckValue;
    if (pNode->FeatureDis == DT_FEATURE_DISCRETE)
    {
        if (currentValue == checkVallue)
            DecisionTreePredicty(pNode->PTrueChildren, xMatrix, idx, yVector);
        else
            DecisionTreePredicty(pNode->PFalseChildren, xMatrix, idx, yVector);
    }
    else if (pNode->FeatureDis == DT_FEATURE_CONTINUUM)
    {
        if (currentValue >= checkVallue)
            DecisionTreePredicty(pNode->PTrueChildren, xMatrix, idx, yVector);
        else
            DecisionTreePredicty(pNode->PFalseChildren, xMatrix, idx, yVector);
    }
}

/// @brief 删除决策树
/// @param[in] pNode 需要删除的结点
static void DecisionTreeDelete(IN CDecisionTreeNode* pNode)
{
    if (pNode == nullptr)
        return;

    if (pNode->PTrueChildren != nullptr)
        DecisionTreeDelete(pNode->PTrueChildren);
    if (pNode->PFalseChildren != nullptr)
        DecisionTreeDelete(pNode->PFalseChildren);


    delete pNode;
}

/// @brief 打印决策树树
static void DecisionTreePrint(IN const CDecisionTreeNode* pNode, IN string space)
{
    if (pNode == nullptr)
        return;

    if (pNode->PTrueChildren == nullptr &&
        pNode->PFalseChildren == nullptr)
    {
        printf("{TargetValue: %.2f LossValue: %.2f}\n", pNode->TargetValue, pNode->LossValue);
        return;
    }

    printf(" %d : %.2f ?\n", pNode->CheckColumn, pNode->CheckValue);

    printf("%sTrue->  ", space.c_str());
    DecisionTreePrint(pNode->PTrueChildren, space + "  ");
    printf("%sFalse->  ", space.c_str());
    DecisionTreePrint(pNode->PFalseChildren, space + "  ");
}


/// @brief 决策树
class CDecisionTree
{
public:
    /// @brief 构造函数
    explicit CDecisionTree(int treeType)
    {
        m_pXMatrix = nullptr;
        m_pNVector = nullptr;
        m_pYVector = nullptr;

        m_pRootNode = nullptr;

        m_treeType = treeType;
        
    }

    /// @brief 析构函数
    ~CDecisionTree()
    {
        if (m_pRootNode != nullptr)
        {
            DecisionTreeDelete(m_pRootNode);
            m_pRootNode = nullptr;
        }
    }

    /// @brief 训练模型
    bool TrainModel(IN const LDTMatrix& xMatrix, IN const LDTMatrix& nVector, IN const LDTMatrix& yVector)
    {
        // 检查参数
        if (xMatrix.RowLen < 1)
            return false;
        if (xMatrix.ColumnLen < 1)
            return false;
        if (nVector.RowLen != 1)
            return false;
        if (nVector.ColumnLen != xMatrix.ColumnLen)
            return false;
        if (yVector.ColumnLen != 1)
            return false;
        if (yVector.RowLen != xMatrix.RowLen)
            return false;
        for (unsigned int i = 0; i < nVector.ColumnLen; i++)
        {
            if (nVector[0][i] != DT_FEATURE_DISCRETE && nVector[0][i] != DT_FEATURE_CONTINUUM)
                return false;
        }

        // 如果已经训练过, 则删除树
        if (m_pRootNode != nullptr)
        {
            DecisionTreeDelete(m_pRootNode);
            m_pRootNode = nullptr;
        }

        // 将样本集拆分为训练集和验证集, 30%作为验证集
        unsigned int verifySampleCount = (unsigned int)(xMatrix.RowLen * 0.3);
        LDTMatrix verifyXMatrix;
        LDTMatrix trainXMatrix;
        xMatrix.SubMatrix(0, verifySampleCount, 0, xMatrix.ColumnLen, verifyXMatrix);
        xMatrix.SubMatrix(verifySampleCount, xMatrix.RowLen - verifySampleCount, 0, xMatrix.ColumnLen, trainXMatrix);
        LDTMatrix verifyYVector;
        LDTMatrix trainYVector;
        yVector.SubMatrix(0, verifySampleCount, 0, yVector.ColumnLen, verifyYVector);
        yVector.SubMatrix(verifySampleCount, yVector.RowLen - verifySampleCount, 0, yVector.ColumnLen, trainYVector);

        m_pXMatrix = &trainXMatrix;
        m_pYVector = &trainYVector;
        m_pNVector = &nVector;
        m_featureNum = xMatrix.ColumnLen;

        // 提取样本标签
        vector<unsigned int> xIdxList;
        xIdxList.reserve(m_pXMatrix->RowLen);
        for (unsigned int i = 0; i < m_pXMatrix->RowLen; i++)
        {
            xIdxList.push_back(i);
        }

        // 建一个尽量大的树
        double lossValue = 0.0;
        double targetValue = 0.0;
        this->CalculateLossValue(xIdxList, targetValue, lossValue);
        CDecisionTreeNode* pTree = RecursionBuildTree(xIdxList, targetValue, lossValue);

        // 剪枝后的树列表
        vector<CDecisionTreeNode*> treeList;

        while (true)
        {
            // 复制树并且保存起来
            CDecisionTreeNode* pNewTree = DecisionTreeCopy(pTree);
            treeList.push_back(pNewTree);

            // 如果树被剪枝只剩根结点则退出
            if (pTree->PTrueChildren == nullptr &&
                pTree->PFalseChildren == nullptr)
            {
                DecisionTreeDelete(pTree);
                pTree = nullptr;
                break;
            }

            // 获取树所有内部结点
            vector<CDecisionTreeNode*> nodeList;
            DecisionTreeInsideNodes(pTree, nodeList);

            // 针对每个结点计算对该结点进行剪枝后整个树损失函数减少的程度
            // 选择树损失函数减少的最小的结点进行剪枝
            double minAlpha = -1.0;
            CDecisionTreeNode* pPruneNode = nullptr;
            for (auto iter = nodeList.begin(); iter != nodeList.end(); iter++)
            {
                lossValue = 0.0;
                unsigned int leafCount = 0;
                DecisionTreeLossValue(*iter, lossValue, leafCount);
                double alpha = ((*iter)->LossValue - lossValue) / (leafCount - 1);
                
                if (minAlpha == -1.0)
                {
                    minAlpha = alpha;
                    pPruneNode = *iter;
                    continue;
                }

                if (alpha < minAlpha)
                {
                    minAlpha = alpha;
                    pPruneNode = *iter;
                }
            }

            // 进行剪枝
            DecisionTreeDelete(pPruneNode->PTrueChildren);
            DecisionTreeDelete(pPruneNode->PFalseChildren);
            pPruneNode->PTrueChildren = nullptr;
            pPruneNode->PFalseChildren = nullptr;
        }

        // 使用验证集在被剪枝的树中选择一个最优的
        double minErrorSum = -1.0;
        CDecisionTreeNode* pBestTree = nullptr;
        for (auto iter = treeList.begin(); iter != treeList.end(); iter++)
        {
            double errorSum = 0.0;
            LDTMatrix predictY(verifyYVector.RowLen, 1, 0.0);
            for (unsigned int i = 0; i < verifyXMatrix.RowLen; i++)
            {
                DecisionTreePredicty(*iter, verifyXMatrix, i, predictY);

                if (m_treeType == REGRESSION_TREE)
                {
                    double dif = verifyYVector[i][0] - predictY[i][0];
                    errorSum += dif * dif;
                }
                else if (m_treeType == CLASSIFIER_TREE)
                {
                    if (verifyYVector[i][0] != predictY[i][0])
                        errorSum += 1;
                }

            }

            if (minErrorSum == -1.0)
            {
                minErrorSum = errorSum;
                pBestTree = *iter;
                continue;
            }

            if (errorSum < minErrorSum)
            {
                // 删除被抛弃的树
                DecisionTreeDelete(pBestTree);
                pBestTree = nullptr;

                minErrorSum = errorSum;
                pBestTree = *iter;
            }
            else
            {
                // 删除被抛弃的树
                DecisionTreeDelete(*iter);
            }

            *iter = nullptr;
            
        }

        m_pRootNode = pBestTree;

        m_pXMatrix = nullptr;
        m_pYVector = nullptr;
        m_pNVector = nullptr;

        return true;
    }

    /// @brief 使用训练好的模型预测数据
    bool Predict(IN const LDTMatrix& xMatrix, OUT LDTMatrix& yVector) const
    {
        // 检查参数
        if (nullptr == m_pRootNode)
            return false;
        if (xMatrix.RowLen < 1)
            return false;
        if (xMatrix.ColumnLen != m_featureNum)
            return false;

        yVector.Reset(xMatrix.RowLen, 1, 0.0);

        for (unsigned int i = 0; i < xMatrix.RowLen; i++)
        {
            DecisionTreePredicty(m_pRootNode, xMatrix, i, yVector);
        }

        return true;
    }

    /// @brief 计算模型得分
    double Score(IN const LDTMatrix& xMatrix, IN const LDTMatrix& yVector) const
    {
        if (m_treeType == REGRESSION_TREE)
            return this->ScoreRSquare(xMatrix, yVector);
        else if (m_treeType == CLASSIFIER_TREE)
            return this->ScoreAccuracy(xMatrix, yVector);

        return 0.0;
    }



    /// @brief 打印树, 用于调试
    void PrintTree() const
    {
        printf("Regression Tree: \n");
        DecisionTreePrint(m_pRootNode, "  ");
        printf("\n");
    }

private:
    /// @brief 递归构造决策树
    /// @param[in] xIdxList 样本索引列表
    /// @return 决策树节点
    CDecisionTreeNode* RecursionBuildTree(IN const vector<unsigned int>& xIdxList, double targetValue, double lossValue)
    {
        CDecisionTreeNode* pNode = new CDecisionTreeNode();

        // 存储当前结点的目标值和损失值
        pNode->TargetValue = targetValue;
        pNode->LossValue = lossValue;

        // 如果当前损失值为0.0则生成叶子结点
        if (lossValue == 0.0f)
        {
            pNode->PTrueChildren = nullptr;
            pNode->PFalseChildren = nullptr;
            return pNode;
        }

        double minLossValue = lossValue;     // 最小损失值
        unsigned int bestCheckCol = 0;       // 最佳检查列
        double bestCheckValue;               // 最佳检查值
        double bestColFeaturDis;             // 最佳列特征分布
        vector<unsigned int> xBestTrueList;  // 最佳true分支样本索引列表
        vector<unsigned int> xBestFalseList; // 最佳false分支样本索引列表
        double bestTrueTargetValue;
        double bestTrueLossValue;
        double bestFalseTargetValue;
        double bestFalseLossValue;

         // 针对每个列
        for (unsigned int col = 0; col < m_pXMatrix->ColumnLen; col++)
        {
            set<double> columnValueSet; // 列中不重复的值集合

             // 当前列中生成一个由不同值构成的序列
            for (unsigned int i = 0; i < xIdxList.size(); i++)
            {
                unsigned int idx = xIdxList[i];

                columnValueSet.insert((*m_pXMatrix)[idx][col]);
            }

            // 针对一个列中的每个不同值
            for (auto iter = columnValueSet.begin(); iter != columnValueSet.end(); iter++)
            {
                double checkValue = *iter;       // 检查值
                vector<unsigned int> xTrueList;  // true分支样本索引列表
                vector<unsigned int> xFalseList; // false分支样本索引列表
                this->DevideSample(xIdxList, col, checkValue, xTrueList, xFalseList);
                if (xTrueList.size() == 0 ||
                    xFalseList.size() == 0)
                    continue;

                double trueMean;
                double trueLossValue;
                double falseMean;
                double falseLossValue;
                this->CalculateLossValue(xTrueList, trueMean, trueLossValue);
                this->CalculateLossValue(xFalseList, falseMean, falseLossValue);
                double sumLossValue = trueLossValue + falseLossValue;

                if (sumLossValue < minLossValue)
                {
                    minLossValue = sumLossValue;
                    bestCheckCol = col;
                    bestCheckValue = checkValue;
                    bestColFeaturDis = (*m_pNVector)[0][col];
                    xBestTrueList = xTrueList;
                    xBestFalseList = xFalseList;

                    bestTrueTargetValue = trueMean;
                    bestTrueLossValue = trueLossValue;
                    bestFalseTargetValue = falseMean;
                    bestFalseLossValue = falseLossValue;
                }
            }
        }

        CDecisionTreeNode* pTrueChildren = this->RecursionBuildTree(xBestTrueList, bestTrueTargetValue, bestTrueLossValue);
        CDecisionTreeNode* pFalseChildren = this->RecursionBuildTree(xBestFalseList, bestFalseTargetValue, bestFalseLossValue);
        pNode->PTrueChildren = pTrueChildren;
        pNode->PFalseChildren = pFalseChildren;
        pNode->CheckColumn = bestCheckCol;
        pNode->CheckValue = bestCheckValue;
        pNode->FeatureDis = bestColFeaturDis;

        return pNode;

    }

    /// @brief 拆分样本集
    /// @param[in] xIdxList 样本索引列表
    /// @param[in] column 拆分依据的列
    /// @param[in] checkValue 拆分依据的列的检查值
    /// @param[out] xTrueList 检查结果为true的样本索引列表
    /// @param[out] xFalseList 检查结果为false的样本索引列表
    void DevideSample(
        IN const vector<unsigned int>& xIdxList,
        IN unsigned int column,
        IN double checkValue,
        OUT vector<unsigned int>& xTrueList,
        OUT vector<unsigned int>& xFalseList)
    {
        xTrueList.clear();
        xFalseList.clear();

        if ((*m_pNVector)[0][column] == DT_FEATURE_DISCRETE)
        {
            for (unsigned int i = 0; i < xIdxList.size(); i++)
            {
                unsigned int idx = xIdxList[i];

                if ((*m_pXMatrix)[idx][column] == checkValue)
                    xTrueList.push_back(idx);
                else
                    xFalseList.push_back(idx);

            }

        }

        if ((*m_pNVector)[0][column] == DT_FEATURE_CONTINUUM)
        {
            for (unsigned int i = 0; i < xIdxList.size(); i++)
            {
                unsigned int idx = xIdxList[i];

                if ((*m_pXMatrix)[idx][column] >= checkValue)
                    xTrueList.push_back(idx);
                else
                    xFalseList.push_back(idx);

            }
        }
    }

    /// @brief 计算样本损失值
    /// @param[in] xIdxList 样本索引列表
    /// @param[out] targetValue 目标值
    /// @param[out] lossValue 损失值
    /// @return 样本损失值
    void CalculateLossValue(IN const vector<unsigned int>& xIdxList, OUT double& targetValue, OUT double& lossValue) const
    {
        if (m_treeType == CLASSIFIER_TREE)
            this->CalculateLossValueGini(xIdxList, targetValue, lossValue);
        else if (m_treeType == REGRESSION_TREE)
            this->CalculateLossValueSquare(xIdxList, targetValue, lossValue);
    }

    /// @brief 计算样本损失值(二乘法)
    /// @param[in] xIdxList 样本索引列表
    /// @param[out] targetMean 目标均值
    /// @param[out] lossValue 损失值
    /// @return 样本损失值
    void CalculateLossValueSquare(IN const vector<unsigned int>& xIdxList, OUT double& targetMean, OUT double& lossValue) const
    {
        targetMean = 0.0;
        lossValue = 0.0;

        for (auto iter = xIdxList.begin(); iter != xIdxList.end(); iter++)
        {
            unsigned int idx = *iter;
            targetMean += (*m_pYVector)[idx][0];
        }

        targetMean = targetMean / (double)xIdxList.size();

        for (auto iter = xIdxList.begin(); iter != xIdxList.end(); iter++)
        {
            unsigned int idx = *iter;
            double dif = (*m_pYVector)[idx][0] - targetMean;
            lossValue += dif * dif;
        }
    }

     /// @brief 计算损失值(基尼指数)
     /// @param[in] xIdxList 样本索引列表
     /// @param[out] targetLabel 目标标签
     /// @param[out] lossValue 损失值
     /// @return 计算损失值(基尼指数)
     void CalculateLossValueGini(IN const vector<unsigned int>& xIdxList, OUT double& targetLabel, OUT double& lossValue) const
     {
         map<double, int> labelMap;
     
         for (unsigned int i = 0; i < xIdxList.size(); i++)
         {
             unsigned int idx = xIdxList[i];
             ++labelMap[(*m_pYVector)[idx][0]];
         }
     
         double gini = 1.0;
         int maxLabelCount = 0;
         for (auto iter = labelMap.begin(); iter != labelMap.end(); iter++)
         {
             double prob = (double)(iter->second) / (double)(xIdxList.size());
             gini -= prob * prob;
     
             if (iter->second > maxLabelCount)
             {
                 maxLabelCount = iter->second;
                 targetLabel = iter->first;
             }
         }
     
         lossValue = gini * xIdxList.size();
     }

    /// @brief 计算模型得分(相关指数R^2)
    double ScoreRSquare(IN const LDTMatrix& xMatrix, IN const LDTMatrix& yVector) const
    {
        // 检查参数
        if (nullptr == m_pRootNode)
            return 2.0;
        if (xMatrix.RowLen < 1)
            return 2.0;
        if (xMatrix.ColumnLen != m_featureNum)
            return 2.0;
        if (yVector.ColumnLen != 1)
            return 2.0;
        if (yVector.RowLen != xMatrix.RowLen)
            return 2.0;

        LDTMatrix predictY;
        this->Predict(xMatrix, predictY);

        double sumY = 0.0;
        for (unsigned int i = 0; i < yVector.RowLen; i++)
        {
            sumY += yVector[i][0];
        }
        double meanY = sumY / (double)yVector.RowLen;

        double lossValue = 0.0;
        double denominator = 0.0;
        for (unsigned int i = 0; i < yVector.RowLen; i++)
        {
            double dis = yVector[i][0] - predictY[i][0];
            lossValue += dis * dis;

            dis = yVector[i][0] - meanY;
            denominator += dis * dis;
        }

        double score = 1.0 - lossValue / denominator;

        return score;
    }

     /// @brief 计算模型得分(正确率)
     double ScoreAccuracy(IN const LDTMatrix& xMatrix, IN const LDTMatrix& yVector) const
     {
         // 检查参数
         if (nullptr == m_pRootNode)
             return -1.0;
         if (xMatrix.RowLen < 1)
             return -1.0;
         if (xMatrix.ColumnLen != m_featureNum)
             return -1.0;
         if (yVector.ColumnLen != 1)
             return -1.0;
         if (yVector.RowLen != xMatrix.RowLen)
             return -1.0;
     
         LDTMatrix predictVector;
         this->Predict(xMatrix, predictVector);
         if (predictVector.RowLen != yVector.RowLen)
             return -1.0;
     
         double trueCount = 0.0;
         for (unsigned int row = 0; row < yVector.RowLen; row++)
         {
             if (predictVector[row][0] == yVector[row][0])
                 trueCount += 1.0;
         }
     
         return trueCount / (double)yVector.RowLen;
     }

private:

    const LDTMatrix* m_pXMatrix;   ///< 样本矩阵, 训练时所用临时变量
    const LDTMatrix* m_pYVector;   ///< 标签向量(列向量), 训练时所用临时变量
    const LDTMatrix* m_pNVector;   ///< 特征分布向量(行向量), 训练时所用临时变量

    unsigned int m_featureNum;      ///< 特征数

    int m_treeType;                 ///< 树类型
    CDecisionTreeNode* m_pRootNode; ///< 决策树根结点

};

LDecisionTreeClassifier::LDecisionTreeClassifier()
{
    m_pClassifier = nullptr;
    m_pClassifier = new CDecisionTree(CLASSIFIER_TREE);
}

LDecisionTreeClassifier::~LDecisionTreeClassifier()
{
    if (m_pClassifier != nullptr)
    {
        delete m_pClassifier;
        m_pClassifier = nullptr;
    }
}

bool LDecisionTreeClassifier::TrainModel(IN const LDTMatrix& xMatrix, IN const LDTMatrix& nVector, IN const LDTMatrix& yVector)
{
    return m_pClassifier->TrainModel(xMatrix, nVector, yVector);
}

bool LDecisionTreeClassifier::Predict(IN const LDTMatrix& xMatrix, OUT LDTMatrix& yVector) const
{
    return m_pClassifier->Predict(xMatrix, yVector);
}

double LDecisionTreeClassifier::Score(IN const LDTMatrix& xMatrix, IN const LDTMatrix& yVector) const
{
    return m_pClassifier->Score(xMatrix, yVector);
}

void LDecisionTreeClassifier::PrintTree() const
{
    return m_pClassifier->PrintTree();
}

LDecisionTreeRegression::LDecisionTreeRegression()
{
    m_pRegressor = nullptr;
    m_pRegressor = new CDecisionTree(REGRESSION_TREE);
}

LDecisionTreeRegression::~LDecisionTreeRegression()
{
    if (m_pRegressor != nullptr)
    {
        delete m_pRegressor;
        m_pRegressor = nullptr;
    }
}

bool LDecisionTreeRegression::TrainModel(IN const LDTMatrix& xMatrix, IN const LDTMatrix& nVector, IN const LDTMatrix& yVector)
{
    return m_pRegressor->TrainModel(xMatrix, nVector, yVector);
}

bool LDecisionTreeRegression::Predict(IN const LDTMatrix& xMatrix, OUT LDTMatrix& yVector) const
{
    return m_pRegressor->Predict(xMatrix, yVector);
}

double LDecisionTreeRegression::Score(IN const LDTMatrix& xMatrix, IN const LDTMatrix& yVector) const
{
    return m_pRegressor->Score(xMatrix, yVector);
}

void LDecisionTreeRegression::PrintTree() const
{
    return m_pRegressor->PrintTree();
}

