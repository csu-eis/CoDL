
#include <cmath>

#include <vector>
using std::vector;

#include "LKDTree.h"
#include "LDataStruct/LOrderedList.h"



/// @brief KD树节点
struct LKDTreeNode
{
    enum
    {
        UNDEFINE_SPLIT = -1 // 表示未定义的分割序号
    };
    int Split; ///< 垂直于分割超面的方向轴序号(如果值为UNDEFINE_SPLIT, 表示该节点为叶子节点)
    unsigned int DataIndex; ///< 节点数据的索引
    LKDTreeNode* Parent; ///< 父节点
    LKDTreeNode* LeftChildren; ///< 左孩子节点
    LKDTreeNode* RightChildren; ///< 右孩子 节点
};

/// @brief KD树节点与目标点的距离
struct LKDTreeNodeDistance
{
    unsigned int DataIndex; ///< 数据索引
    float Distance; ///< 距离值

    bool operator < (IN const LKDTreeNodeDistance& B) const
    {
        if (this->Distance < B.Distance)
            return true;
        else
            return false;
    }
};

typedef vector<LKDTreeNode*> LKDTreeNodeList; ///< 树节点列表

/// @brief KD树
class CKDTree
{
public:
    /// @brief 构造函数
    CKDTree()
    {
        this->m_pRootNode = 0;
    }

    /// @brief 析构函数
    ~CKDTree()
    {
        this->ClearTree(m_pRootNode);
    }

    /// @brief 构造KD树
    void BuildTree(IN const LKDTreeMatrix& dataSet)
    {
        // 清理树
        if (this->m_pRootNode != 0)
            this->ClearTree(this->m_pRootNode);

        // 检查数据集
        if (dataSet.RowLen < 1 || dataSet.ColumnLen < 1)
            return;

        // 复制数据集
        this->m_dataSet = dataSet;

        // 递归构建树
        this->m_pRootNode = new LKDTreeNode();
        vector<unsigned int> dataIndexList(dataSet.RowLen);
        for (unsigned int i = 0; i < dataIndexList.size(); i++)
        {
            dataIndexList[i] = i;
        }
        this->CreateTree(0, this->m_pRootNode, dataIndexList);
    }

    /// @brief 在数据集中搜索与指定数据最邻近的数据索引
    int SearchNearestNeighbor(IN const LKDTreeMatrix& data)
    {
        LKDTreeList indexList;
        bool bRet = this->SearchKNearestNeighbors(data, 1, indexList);
        if (!bRet)
        {
            return -1;
        }

        return indexList[0][0];
    }

    /// @brief 在数据集中搜索与指定数据最邻近的K个数据索引
    bool SearchKNearestNeighbors(IN const LKDTreeMatrix& data, IN unsigned int k, OUT LKDTreeList& indexList)
    {
        // 检查参数
        if (data.RowLen != 1 || data.ColumnLen != m_dataSet.ColumnLen)
            return false;
        if (k < 1)
            return false;

        LKDTreeNodeList searchPath;
        this->SearchTree(data, searchPath);
        if (searchPath.size() < 1)
            return false;

        LOrderedList<LKDTreeNodeDistance> nearestDistanceList;

        // 在搜索路径中查找是否存在比当前最近点还近的点
        for (int i = (int)searchPath.size() - 1; i >= 0; i--)
        {
            LKDTreeNode* node = searchPath[i];

            if (nearestDistanceList.Size() < k)
            {
                LKDTreeNodeList nodeList;
                this->TraverseTree(node, nodeList);
                if (nodeList.size() >= k)
                {
                    LKDTreeNodeDistance nodeDistance;
                    for (unsigned int n = 0; n < nodeList.size(); n++)
                    {
                        nodeDistance.DataIndex = nodeList[n]->DataIndex;
                        nodeDistance.Distance = this->CalculateDistance(data, nodeDistance.DataIndex);
                        nearestDistanceList.Insert(nodeDistance);
                    }
                }
                continue;
            }

            while (nearestDistanceList.Size() > k)
            {
                nearestDistanceList.PopBack();
            }


            LKDTreeNodeDistance currentNearestDistanceMax = nearestDistanceList.End()->Data;
            // 以目标点和当前最近点的距离为半径作的圆如果和分割超面相交则搜索分割面的另一边区域
            float dif = data[0][node->Split] - m_dataSet[node->DataIndex][node->Split];
            dif = abs(dif);
            if (dif >= currentNearestDistanceMax.Distance)
            {
                continue;
            }

            LKDTreeNodeList nodeList;
            nodeList.push_back(node);
            if (m_dataSet[currentNearestDistanceMax.DataIndex][node->Split] < m_dataSet[node->DataIndex][node->Split])
            {
                this->TraverseTree(node->RightChildren, nodeList);
            }
            else if (m_dataSet[currentNearestDistanceMax.DataIndex][node->Split] > m_dataSet[node->DataIndex][node->Split])
            {
                this->TraverseTree(node->LeftChildren, nodeList);
            }
            else
            {
                this->TraverseTree(node->RightChildren, nodeList);
                this->TraverseTree(node->LeftChildren, nodeList);
            }

            for (unsigned int n = 0; n < nodeList.size(); n++)
            {
                float distance = this->CalculateDistance(data, nodeList[n]->DataIndex);
                LKDTreeNodeDistance nodeDistance;
                if (distance < currentNearestDistanceMax.Distance)
                {
                    nodeDistance.Distance = distance;
                    nodeDistance.DataIndex = nodeList[n]->DataIndex;
                    nearestDistanceList.Insert(nodeDistance);
                    nearestDistanceList.PopBack();
                    currentNearestDistanceMax = nearestDistanceList.End()->Data;
                }
            }


        }

        if (nearestDistanceList.Size() < k)
            return false;


        indexList.Reset(1, k, -1);

        const LOrderedListNode<LKDTreeNodeDistance>* pCurrentNode = nearestDistanceList.Begin();

        int col = 0;
        while (pCurrentNode)
        {
            indexList[0][col] = pCurrentNode->Data.DataIndex;
            col++;

            pCurrentNode = pCurrentNode->PNext;
        }

        return true;
    }

private:
    /// @brief 构造KD树
    /// @param[in] pParent 父节点
    /// @param[in] pNode 当前节点
    /// @param[in] dataIndexList 数据索引列表
    void CreateTree(
        IN LKDTreeNode* pParent,
        IN LKDTreeNode* pNode,
        IN const vector<unsigned int>& dataIndexList)
    {
        if (pNode == 0)
            return;

        pNode->Parent = pParent;

        // 只剩一个数据, 则为叶子节点
        if (dataIndexList.size() == 1)
        {
            pNode->DataIndex = dataIndexList[0];
            pNode->Split = LKDTreeNode::UNDEFINE_SPLIT;
            pNode->LeftChildren = 0;
            pNode->RightChildren = 0;

            return;
        }

        unsigned int bestColIndex = 0; // 标记最大方差的维度索引
        unsigned int midDataIndex = 0; // 标记最佳分割点的索引

        this->FindMaxVarianceColumn(dataIndexList, bestColIndex);

        this->FindMidValueOnColumn(dataIndexList, bestColIndex, midDataIndex);

        pNode->Split = bestColIndex;
        pNode->DataIndex = midDataIndex;

        // 将数据分为左右两部分
        vector<unsigned int> leftDataIndexList;
        leftDataIndexList.reserve(dataIndexList.size() * 2 / 3); // 预先分配好内存, 防止在push_back过程中多次重复分配提高效率
        vector<unsigned int> rightDataIndexList;
        rightDataIndexList.reserve(dataIndexList.size() * 2 / 3); // 预先分配好内存, 防止在push_back过程中多次重复分配提高效率
        for (unsigned int i = 0; i < dataIndexList.size(); i++)
        {
            unsigned int m = dataIndexList[i];
            if (m == midDataIndex)
                continue;

            if (this->m_dataSet[m][bestColIndex] <= this->m_dataSet[midDataIndex][bestColIndex])
                leftDataIndexList.push_back(dataIndexList[i]);
            else
                rightDataIndexList.push_back(dataIndexList[i]);
        }

        // 构建左子树
        if (leftDataIndexList.size() == 0)
        {
            pNode->LeftChildren = 0;
        }
        if (leftDataIndexList.size() != 0)
        {
            pNode->LeftChildren = new LKDTreeNode;
            this->CreateTree(pNode, pNode->LeftChildren, leftDataIndexList);
        }

        // 构建右子树
        if (rightDataIndexList.size() == 0)
        {
            pNode->RightChildren = 0;
        }
        if (rightDataIndexList.size() != 0)
        {
            pNode->RightChildren = new LKDTreeNode;
            this->CreateTree(pNode, pNode->RightChildren, rightDataIndexList);
        }
    }

    /// @brief 在指定的数据集上, 找出最大方差列
    /// @param[in] dataIndexList 数据索引列表, 要求至少要有两行数据
    /// @param[out] col 存储列索引
    /// @return 成功返回true, 失败返回false
    bool FindMaxVarianceColumn(IN const vector<unsigned int>& dataIndexList, OUT unsigned int& col)
    {
        if (dataIndexList.size() < 2)
            return false;


        // 找出具有最大方差的维度
        float maxVariance = 0.0f; // 标记所有维度上的数据方差的最大值
        unsigned int bestCol = 0; // 标记最大方差的维度索引
        for (unsigned int n = 0; n < this->m_dataSet.ColumnLen; n++)
        {
            float sumValue = 0.0f; // 指定列的数据和
            for (unsigned int i = 0; i < dataIndexList.size(); i++)
            {
                unsigned int m = dataIndexList[i];
                sumValue += this->m_dataSet[m][n];
            }

            float averageValue = sumValue / (float)dataIndexList.size(); // 计算指定列的平均值

            float variance = 0.0f; // 指定列的方差值
            for (unsigned int i = 0; i < dataIndexList.size(); i++)
            {
                unsigned int m = dataIndexList[i];
                float dif = averageValue - this->m_dataSet[m][n];
                variance += dif * dif;
            }
            variance = variance / (float)dataIndexList.size();

            if (variance > maxVariance)
            {
                maxVariance = variance;
                bestCol = n;
            }
        }

        col = bestCol;

        return true;
    }

    /// @brief 在指定的数据集上, 找出指定列的中值数据索引(最靠近平均数)
    /// @param[in] dataIndexList 数据索引列表, 要求至少要有两行数据
    /// @param[in] col 列索引
    /// @param[out] dataIndex 存储数据索引
    /// @return 成功返回true, 失败返回false
    bool FindMidValueOnColumn(IN const vector<unsigned int>& dataIndexList, IN unsigned int col, OUT unsigned int& dataIndex)
    {
        if (dataIndexList.size() < 1)
            return false;


        float sum = 0.0f;
        for (unsigned int i = 0; i < dataIndexList.size(); i++)
        {
            unsigned int m = dataIndexList[i];

            sum += m_dataSet[m][col];
        }

        float avg = sum / dataIndexList.size();

        unsigned int midDataIndex = dataIndexList[0];
        float miniDif = abs(m_dataSet[midDataIndex][col] - avg);

        for (unsigned int i = 0; i < dataIndexList.size(); i++)
        {
            unsigned int m = dataIndexList[i];

            float dif = abs(m_dataSet[m][col] - avg);
            if (dif < miniDif)
            {
                miniDif = dif;
                midDataIndex = m;
            }

        }

        dataIndex = midDataIndex;

        return true;
    }

    /// @brief 遍历树
    /// @param[in] pNode 树节点
    /// @param[out] nodeList 遍历出来的节点列表
    void TraverseTree(IN LKDTreeNode* pNode, OUT LKDTreeNodeList& nodeList)
    {
        if (pNode == 0)
            return;

        nodeList.push_back(pNode);

        this->TraverseTree(pNode->LeftChildren, nodeList);
        this->TraverseTree(pNode->RightChildren, nodeList);
    }

    /// @brief 搜索树
    /// @param[in] data 源数据
    /// @param[out] searchPath 搜索出的路径
    void SearchTree(IN const LKDTreeMatrix& data, OUT LKDTreeNodeList& searchPath)
    {
        searchPath.clear();

        LKDTreeNode* currentNode = m_pRootNode;
        while (currentNode != 0)
        {
            searchPath.push_back(currentNode);

            // 到达叶子节点, 结束搜索
            if (currentNode->Split == LKDTreeNode::UNDEFINE_SPLIT)
            {
                currentNode = 0;
                continue;
            }

            float splitData = m_dataSet[currentNode->DataIndex][currentNode->Split];
            if (data[0][currentNode->Split] <= splitData)
            {
                currentNode = currentNode->LeftChildren;
                continue;
            }
            if (data[0][currentNode->Split] > splitData)
            {
                currentNode = currentNode->RightChildren;
                continue;
            }

        }
    }

    /// @brief 计算指定数据与数据集中一个数据的距离值
    /// @param[in] data 指定的数据
    /// @param[in] index 数据集中的数据索引
    /// @return 返回距离值(欧几里得距离), 使用前请保证参数正确
    float CalculateDistance(IN const LKDTreeMatrix& data, IN unsigned int index)
    {
        float sqrSum = 0.0f;
        for (unsigned int i = 0; i < data.ColumnLen; i++)
        {
            float dif = data[0][i] - m_dataSet[index][i];
            sqrSum += dif * dif;
        }
        sqrSum = sqrtf(sqrSum);

        return sqrSum;
    }

    /// @brief 清理树
    /// @param[in] pNode 需要清理的节点
    void ClearTree(IN LKDTreeNode*& pNode)
    {
        if (pNode == 0)
            return;

        this->ClearTree(pNode->LeftChildren);
        this->ClearTree(pNode->RightChildren);

        delete pNode;
        pNode = 0;
    }

private:
    LKDTreeNode* m_pRootNode; ///< 根节点
    LKDTreeMatrix m_dataSet; ///< 数据集
};


LKDTree::LKDTree()
    : m_pKDTree(0)
{
    m_pKDTree = new CKDTree();
}

LKDTree::~LKDTree()
{
    if (m_pKDTree != 0)
    {
        delete m_pKDTree;
        m_pKDTree = 0;
    }
}

void LKDTree::BuildTree(IN const LKDTreeMatrix& dataSet)
{
    m_pKDTree->BuildTree(dataSet);
}

int LKDTree::SearchNearestNeighbor(IN const LKDTreeMatrix& data)
{
    return m_pKDTree->SearchNearestNeighbor(data);
}

bool LKDTree::SearchKNearestNeighbors(IN const LKDTreeMatrix& data, IN unsigned int k, OUT LKDTreeList& indexList)
{
    return m_pKDTree->SearchKNearestNeighbors(data, k, indexList);
}