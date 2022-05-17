
#include "LDataCluster.h"

#include <cstdlib>
#include <ctime>

#include <list>
using std::list;

#ifndef SAFE_DELETE
#define SAFE_DELETE(p) \
do\
{\
    if ((p) != NULL)\
    { \
        delete p;\
        p = NULL; \
    }\
}while(0)
#endif


LBiClusterTree::LBiClusterTree()
{
    m_dataSimilarMethod = PEARSON_CORRETATIO;
    m_pDataSimilar = NULL;
    m_pRootNode = NULL;
}

LBiClusterTree::~LBiClusterTree()
{
    Clear(m_pRootNode);
    SAFE_DELETE(m_pDataSimilar);
}


void LBiClusterTree::SetDataSimilerMethod(IN LDATA_SIMILAR_METHOD similarMethod)
{
    m_dataSimilarMethod = similarMethod;
}

void LBiClusterTree::Init()
{
    SAFE_DELETE(m_pDataSimilar);
    Clear(m_pRootNode);

    switch (m_dataSimilarMethod)
    {
    case EUCLIDEAN_DISTANCE:
        m_pDataSimilar = new LEuclideanDistance();
        break;
    case PEARSON_CORRETATIO:
        m_pDataSimilar = new LPearsonCorretation();
        break;
    default:
        m_pDataSimilar = new LPearsonCorretation();
    }
}


void LBiClusterTree::Cluster(IN const LDCDataMatrix& dataMatrix)
{
    Clear(m_pRootNode);
   
    if (dataMatrix.Length <= 0)
        return;

    LDCDataMatrix disMatrix; // 记录数据之间的距离值
    disMatrix.Reset(dataMatrix.Length * 2);
    for (int i = 0; i < disMatrix.Length; i++)
    {
        disMatrix.Data[i].Reset(dataMatrix.Length * 2);
        for (int j = 0; j < disMatrix.Data[i].Length; j++)
            disMatrix.Data[i].Data[j] = 100.0f;
    }

    list<LBiClusterTNode*> nodeList; // 结点指针列表

    // 生成所有的叶子聚类结点
    for (int i = 0; i < dataMatrix.Length; i++)
    {
        LBiClusterTNode* pNode = new LBiClusterTNode();
        pNode->DataList = dataMatrix.Data[i];
        pNode->Id = i;
        nodeList.push_back(pNode);
    }

    int currentId = dataMatrix.Length;
    while (nodeList.size() > 1)
    {
        auto iterA = nodeList.begin();
        auto iterB = nodeList.begin();
        iterB++;

        float minDis = 1.0f - m_pDataSimilar->Calculate((*iterA)->DataList, (*iterB)->DataList);

        for (auto iterOne = nodeList.begin(); iterOne != nodeList.end(); iterOne++)
        {
            auto iterTwo = iterOne;
            iterTwo++;
            for ( ; iterTwo != nodeList.end(); iterTwo++)
            {
                int oneId = (*iterOne)->Id;
                int twoId = (*iterTwo)->Id;
                if (disMatrix.Data[oneId].Data[twoId] == 100.0f)
                {
                    float similar = m_pDataSimilar->Calculate((*iterOne)->DataList, (*iterTwo)->DataList);
                    disMatrix.Data[oneId].Data[twoId] = 1.0f - similar;
                }

                float dis = disMatrix.Data[oneId].Data[twoId];

                if (dis < minDis)
                {
                    minDis = dis;
                    iterA = iterOne;
                    iterB = iterTwo;
                }
            }

        }

        // 生成新的聚类结点
        LBiClusterTNode* pNewNode = new LBiClusterTNode();
        pNewNode->Id = currentId;
        pNewNode->Distance =minDis;
        pNewNode->PLChild = *iterA;
        pNewNode->PRChild = *iterB;
        pNewNode->DataList.Reset((*iterA)->DataList.Length);
        for (int i = 0; i < (*iterA)->DataList.Length; i++)
        {
            pNewNode->DataList.Data[i] = (pNewNode->PLChild->DataList.Data[i] + pNewNode->PRChild->DataList.Data[i])/2.0f;
        }

        currentId += 1;
        nodeList.erase(iterA);
        nodeList.erase(iterB);
        nodeList.push_back(pNewNode);
    }

    auto beginIter = nodeList.begin();
    m_pRootNode = *beginIter;
}

void LBiClusterTree::Receive(IN const LBiClustarTreeVisitor& visitor)
{
    visitor.Visit(m_pRootNode);
}

void LBiClusterTree::Clear(IN LBiClusterTNode*& pNode)
{
    if (pNode == NULL)
        return;

    Clear(pNode->PLChild);
    Clear(pNode->PRChild);

    SAFE_DELETE(pNode);
}

LKMeansCluster::LKMeansCluster()
{
    m_k = 2;
    m_dataSimilarMethod = PEARSON_CORRETATIO;
    m_pDataSimilar = 0;
}

LKMeansCluster::~LKMeansCluster()
{
    SAFE_DELETE(m_pDataSimilar);
}

void LKMeansCluster::SetDataSimilerMethod(IN LDATA_SIMILAR_METHOD similarMethod)
{
    m_dataSimilarMethod = similarMethod;
}

void LKMeansCluster::SetK(IN int k)
{
    if (k <= 2)
        return;

    m_k = k;
}

void LKMeansCluster::Init()
{
    srand((int)time(0));

    SAFE_DELETE(m_pDataSimilar);
    switch (m_dataSimilarMethod)
    {
    case EUCLIDEAN_DISTANCE:
        m_pDataSimilar = new LEuclideanDistance();
        break;
    case PEARSON_CORRETATIO:
        m_pDataSimilar = new LPearsonCorretation();
        break;
    default:
        m_pDataSimilar = new LPearsonCorretation();
    }

    m_resultMatrix.Reset(m_k);
}

void LKMeansCluster::Cluster(IN const LDCDataMatrix& dataMatrix, OUT LDCResultMatrix& resultMatrix)
{
    if (dataMatrix.Length <= 0)
        return;

    LDCDataMatrix rangeList; ///< 范围列表
    rangeList.Reset(dataMatrix.Data[0].Length);
    // 确定每个点的最大值和最小值
    for (int col = 0; col < dataMatrix.Data[0].Length; col++)
    {
        float min = dataMatrix.Data[0].Data[col];
        float max = dataMatrix.Data[0].Data[col];

        for (int row = 0; row < dataMatrix.Length; row++)
        {
            float value = dataMatrix.Data[row].Data[col];
            if (value < min)
                min = value;
            if (value > max)
                max = value;
        }

        rangeList.Data[col].Reset(2);
        rangeList.Data[col].Data[0] = min;
        rangeList.Data[col].Data[1] = max;
    }

    // 随机创造K个中心点
    LDCDataMatrix centerPointList;
    centerPointList.Reset(m_k);
    for (int i = 0; i < centerPointList.Length; i++)
    {
        centerPointList.Data[i].Reset(dataMatrix.Data[0].Length);
        for (int j = 0; j < dataMatrix.Data[0].Length; j++)
        {
            centerPointList.Data[i].Data[j] = RandFloat() * (rangeList.Data[j].Data[1] - rangeList.Data[j].Data[0]) 
                + rangeList.Data[j].Data[0];
        }
    }

    
    LArray<list<int>> clusterList; // 分类列表
    clusterList.Reset(m_k);
    float bestTotalDis = dataMatrix.Length * 2.0f;// 2为最远距离值即1 - similar, similar的范围为-1~1
    while (true)
    {
        for (int i = 0; i < clusterList.Length; i++)
        {
            clusterList.Data[i].clear();
        }

        float totalDis = 0.0f;
        for (int row = 0; row < dataMatrix.Length; row++)
        {
            LDCDataList& dataRow = dataMatrix.Data[row];

            float minDis = 1.0f - m_pDataSimilar->Calculate(dataRow, centerPointList.Data[0]);
            int bestMatchIndex = 0;
            for (int i = 1; i< centerPointList.Length; i++)
            {
                float dis = 1.0f - m_pDataSimilar->Calculate(dataRow, centerPointList.Data[i]);
                if (dis < minDis)
                {
                    minDis = dis;
                    bestMatchIndex = i;
                }
            }

            totalDis += minDis;
            clusterList.Data[bestMatchIndex].push_back(row);
        }

        if (totalDis >= bestTotalDis)
            break;

        bestTotalDis = totalDis;

        LDCDataList centerPointTotal;
        centerPointTotal.Reset(dataMatrix.Data[0].Length);
        //移动中心点
        for (int i = 0; i < clusterList.Length; i++)
        {
            for (int j = 0; j < centerPointTotal.Length; j++)
                centerPointTotal.Data[j] = 0.0f;

            for (auto iter = clusterList.Data[i].begin(); iter != clusterList.Data[i].end(); iter++)
            {
                LDCDataList& dataList = dataMatrix.Data[*iter];
                for (int j = 0; j < dataList.Length; j++)
                    centerPointTotal.Data[j] += dataList.Data[j];
            }

            for (int j = 0; j < centerPointTotal.Length; j++)
            {
                centerPointList.Data[i].Data[j] = centerPointTotal.Data[j]/(float)clusterList.Data[i].size();
            }

        }
        
    }

    resultMatrix.Reset(m_k);
    for (int i = 0; i < m_k; i++)
    {
        list<int>& resultList = clusterList.Data[i];
        resultMatrix.Data[i].Reset(resultList.size());

        int k = 0;
        for (auto iter = resultList.begin(); iter != resultList.end(); iter++)
        {
            resultMatrix.Data[i].Data[k++] = *iter;
        }
    }

}

float LKMeansCluster::RandFloat()
{
    return (rand())/(RAND_MAX + 1.0f);
}