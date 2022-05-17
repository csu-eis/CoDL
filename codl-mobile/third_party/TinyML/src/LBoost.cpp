

#include <cmath>
#include <cstdio>

#include <vector>
using std::vector;

#include "LBoost.h"

/// @brief 桩分类规则
enum LSTUMP_CLASSIFY_RULE
{
    LARGER_SUN = 1, ///< 大于则分类为BOOST_SUN, 小于等于则分类为BOOST_MOON
    LARGER_MOON ///< 大于则分类为BOOST_MOON, 小于等于则分类为BOOST_SUN
};

/// @brief 桩结构
struct LStump
{
    unsigned int FeatureIndex; ///< 特征索引, 使用该特征索引对样本进行分类
    float FeatureThreshold; ///< 特征阈值, 使用该特征阈值对样本进行分类
    LSTUMP_CLASSIFY_RULE ClassifyRule; ///< 分类规则, 使用该分类规则对样本进行分类
};

/// @brief 树桩分类器
/// 弱分类器
class CStumpClassifer
{
public:
    /// @brief 构造函数
    CStumpClassifer()
    {
        m_bTrained = false;
    }

    /// @brief 析构函数
    ~CStumpClassifer()
    {

    }

    /// @brief 训练
    /// @param[in] problem 原始问题
    /// @param[inout] weightVector 训练样本的权重向量, 成功训练后保存更新后的权重向量
    /// @param[out] pResultVector 存储该分类桩的分类结果, 不能为0
    /// @return 成功返回true, 失败返回false, 参数有误返回false
    bool Train(
        IN const LBoostProblem& problem, 
        INOUT vector<float>& weightVector, 
        OUT vector<float>* pResultVector)
    {
        // 检查参数
        if (problem.XMatrix.ColumnLen < 1)
            return false;
        if (problem.XMatrix.RowLen < 2)
            return false;
        if (problem.YVector.ColumnLen != 1)
            return false;
        if (problem.YVector.RowLen != problem.XMatrix.RowLen)
            return false;
        if (problem.YVector.RowLen != weightVector.size())
            return false;
        if (0 == pResultVector)
            return false;

        for (unsigned int i = 0; i < problem.YVector.RowLen; i++)
        {
            if (problem.YVector[i][0] != LBOOST_SUN &&
                problem.YVector[i][0] != LBOOST_MOON)
                return false;
        }

        const LBoostMatrix& X = problem.XMatrix; // 样本矩阵
        const LBoostMatrix& Y = problem.YVector; // 标签矩阵

        const unsigned int M = X.RowLen; // 样本数量
        const unsigned int N = X.ColumnLen; // 样本特征数量

        const int StepNum = 10;

        LStump stump; // 树桩临时变量
        vector<float> classisVector(M);

        LStump bestStump; // 最好的树桩
        float minWeightError = 1.0f; // 最小权重错误率
        vector<float> bestClassisVector(M); // 最好的分类结果向量
        

        // 对每一个特征
        for (unsigned int n = 0; n < N; n++)
        {
            stump.FeatureIndex = n;

            float rangeMin = X[0][n]; // 所有样本中列i(特征)中的最小值
            float rangeMax = X[0][n]; // 所有样本中列i(特征)中的最大值
            for (unsigned int m = 0; m < M; m++)
            {
                if (X[m][n] < rangeMin)
                    rangeMin = X[m][n];
                if (X[m][n] > rangeMax)
                    rangeMax = X[m][n];
            }

            float stepSize = (rangeMax - rangeMin)/(float)StepNum;

            for (int k = -1; k <= StepNum + 1; k++)
            {
                stump.FeatureThreshold = rangeMin + k * stepSize;
                stump.ClassifyRule = LARGER_SUN;
                this->Classify(X, stump, &classisVector);

                float weightError = 0.0f;
                for (unsigned int m = 0; m < M; m++)
                {
                    if (classisVector[m] != Y[m][0])
                        weightError += weightVector[m];
                }

                // 如果错误率大于0.5f则表示分类规则应该取反, 分类结果直接取反
                if (weightError > 0.5f)
                {
                    stump.ClassifyRule = LARGER_MOON;
                    weightError = 1.0f - weightError;
                    for (unsigned int m = 0; m < M; m++)
                        classisVector[m] *= LBOOST_MOON;
                }

                if (weightError < minWeightError)
                {
                    minWeightError = weightError;
                    bestStump = stump;
                    bestClassisVector = classisVector;
                }

            }
        }


        // 确保没有错误时, 不会发生除0溢出
        if (minWeightError < 1e-16)
            minWeightError = (float)1e-16;

        m_featureNum = N;

        m_alpha = 0.5f * log((1.0f-minWeightError)/minWeightError);

        m_stump = bestStump;
        (*pResultVector) = bestClassisVector;
        for (auto iter = pResultVector->begin(); iter != pResultVector->end(); iter++)
        {
            (*iter) *= m_alpha; 
        }

        // 使用alpha更新权重向量
        float sumWeight = 0.0f;
        for (unsigned int m = 0; m < M; m++)
        {
            weightVector[m] = weightVector[m] * exp(-1 * m_alpha * bestClassisVector[m] * problem.YVector[m][0]);
            sumWeight += weightVector[m];
        }
        for (unsigned int m = 0; m < M; m++)
        {
            weightVector[m] = weightVector[m]/sumWeight;
        }

        m_bTrained = true;
        return true;

    }

    /// @brief 使用分类器进行预测
    /// @param[in] sampleMatrix 样本矩阵
    /// @param[out] pResultVector 存储结果向量, 不能为0
    /// @return 成功返回true, 失败返回false, 参数有误或者分类器未训练会失败
    bool Predict(IN const LBoostMatrix& sampleMatrix, OUT vector<float>* pResultVector)
    {
        if (!m_bTrained)
            return false;

        if (sampleMatrix.ColumnLen != m_featureNum)
            return false;

        if (sampleMatrix.RowLen < 1)
            return false;

        if (0 == pResultVector)
            return false;

        vector<float> classisVector;
        this->Classify(sampleMatrix, m_stump, &classisVector);
        pResultVector->resize(classisVector.size());
        for (unsigned int i = 0; i < classisVector.size(); i++)
        {
            (*pResultVector)[i] = classisVector[i] * m_alpha;
        }

        return true;
    }

private:
    /// @brief 分类
    /// @param[in] ampleMatrix 样本矩阵
    /// @param[in] stump 分类桩结构
    /// @param[in] pClassisVector 存储分类结果向量
    void Classify(
        IN const LBoostMatrix& sampleMatrix,
        IN const LStump& stump,
        OUT vector<float>* pClassisVector)
    {
        pClassisVector->resize(sampleMatrix.RowLen);

        for (unsigned int i = 0; i < sampleMatrix.RowLen; i++)
        {
            if (stump.ClassifyRule == LARGER_SUN)
            {
                if (sampleMatrix[i][stump.FeatureIndex] > stump.FeatureThreshold)
                    (*pClassisVector)[i] = LBOOST_SUN;
                else
                    (*pClassisVector)[i] = LBOOST_MOON;
            }

            if (stump.ClassifyRule == LARGER_MOON)
            {
                if (sampleMatrix[i][stump.FeatureIndex] > stump.FeatureThreshold)
                    (*pClassisVector)[i] = LBOOST_MOON;
                else
                    (*pClassisVector)[i] = LBOOST_SUN;
            }
        }
    }

private:
    bool m_bTrained; ///< 标识该决策桩是否已经被训练
    unsigned int m_featureNum; ///< 决策桩要求的样本特征数
    

    LStump m_stump; ///< 桩结构
    float m_alpha; //< 决策桩的权重值
};

/// @brief 提升树
///
/// 以决策树为基函数的提升方法称为提升树
class CBoostTree
{
public:
    /// @brief 构造函数
    CBoostTree()
    {
        this->m_maxWeakClassifierNum = 40;
    }

    /// @brief 析构函数
    ~CBoostTree()
    {

    }

    /// @brief 设置最大弱分类器数量
    /// 详细解释见头文件LBoostTree中的声明
    void SetMaxClassifierNum(IN unsigned int num)
    {
        m_maxWeakClassifierNum = num;
    }

    /// @brief 训练模型
    /// 详细解释见头文件LBoostTree中的声明
    bool TrainModel(IN const LBoostProblem& problem)
    {
        if (problem.XMatrix.ColumnLen < 1)
            return false;
        if (problem.XMatrix.RowLen < 2)
            return false;
        if (problem.YVector.ColumnLen != 1)
            return false;
        if (problem.YVector.RowLen != problem.XMatrix.RowLen)
            return false;

        for (unsigned int i = 0; i < problem.YVector.RowLen; i++)
        {
            if (problem.YVector[i][0] != LBOOST_SUN &&
                problem.YVector[i][0] != LBOOST_MOON)
                return false;
        }

        m_featureNum = problem.XMatrix.ColumnLen;

        // 构造并且初始化权重向量(列向量)
        vector<float> weightVector(problem.XMatrix.RowLen);
        for (unsigned int i =0; i < weightVector.size(); i++)
        {
            weightVector[i] = 1.0f/(float)weightVector.size();
        }

        // 构造累加类别向量(列向量)并且初始化
        vector<float> sumClassisVector(problem.XMatrix.RowLen, 0.0f);

        // 弱分类器分类的结果向量
        vector<float> resultVector;

        m_weakClassifierList.clear();

        CStumpClassifer stumpClassifer;
        for (unsigned int i = 0; i < m_maxWeakClassifierNum; i++)
        {
            stumpClassifer.Train(problem, weightVector, &resultVector);
            m_weakClassifierList.push_back(stumpClassifer);

            // 计算累加类别向量
            for (unsigned int m = 0; m < sumClassisVector.size(); m++)
            {
                sumClassisVector[m] += resultVector[m];
            }

            // 计算累加错误率
            int errorCount = 0; // 错误分类计数
            for (unsigned int m = 0; m < sumClassisVector.size(); m++)
            {
                if (sumClassisVector[m] * problem.YVector[m][0] < 0)
                    errorCount++;
            }
            float sumError = (float)errorCount/(float)sumClassisVector.size();
            if (sumError == 0.0f)
                break;

        }

        return true;
    }

    /// @brief 使用训练好的模型进行预测(单样本预测)
    /// 详细解释见头文件LBoostTree中的声明
    float Predict(IN const LBoostMatrix& sample)
    {
        if (sample.RowLen != 1)
            return 0.0f;

        LBoostMatrix classisVector(1, 1);
        bool bRet = this->Predict(sample, &classisVector);
        if (!bRet)
            return 0.0f;

        return classisVector[0][0];
    }

    /// @brief 使用训练好的模型进行预测(多样本预测)
    /// 详细解释见头文件LBoostTree中的声明
    bool Predict(IN const LBoostMatrix& sampleMatrix, OUT LBoostMatrix* pClassisVector)
    {
        if (this->m_weakClassifierList.size() < 1)
            return false;

        if (sampleMatrix.RowLen < 1)
            return false;

        if (sampleMatrix.ColumnLen != this->m_featureNum)
            return false;

        if (0 == pClassisVector)
            return false;

        pClassisVector->Reset(sampleMatrix.RowLen, 1);

        vector<float> resultVector(sampleMatrix.RowLen);
        vector<float> sumResultVector(sampleMatrix.RowLen, 0.0f);
        for (unsigned int i = 0; i < this->m_weakClassifierList.size(); i++)
        {
            CStumpClassifer& stumpClassifer = m_weakClassifierList[i];
            stumpClassifer.Predict(sampleMatrix, &resultVector);
            for (unsigned int j = 0; j < resultVector.size(); j++)
            {
                sumResultVector[j] += resultVector[j];
            }
        }

        for (unsigned int m = 0; m < sumResultVector.size(); m++)
        {
            if (sumResultVector[m] >= 0.0f)
                (*pClassisVector)[m][0] = LBOOST_SUN;
            else
                (*pClassisVector)[m][0] = LBOOST_MOON;
        }

        return true;
    }

private:
    vector<CStumpClassifer> m_weakClassifierList; ///< 弱分类器列表
    unsigned int m_maxWeakClassifierNum; ///< 最大弱分类器数量
    unsigned int m_featureNum; ///< 分类器要求的样本特征数
};

LBoostTree::LBoostTree()
{
    m_pBoostTree = 0;
    m_pBoostTree = new CBoostTree();
}

LBoostTree::~LBoostTree()
{
    if (m_pBoostTree != 0)
    {
        delete m_pBoostTree;
        m_pBoostTree = 0;
    }
}

void LBoostTree::SetMaxClassifierNum(IN unsigned int num)
{
    m_pBoostTree->SetMaxClassifierNum(num);
}

bool LBoostTree::TrainModel(IN const LBoostProblem& problem)
{
    return m_pBoostTree->TrainModel(problem);
}

float LBoostTree::Predict(IN const LBoostMatrix& sample)
{
    return m_pBoostTree->Predict(sample);
}

bool LBoostTree::Predict(IN const LBoostMatrix& sampleMatrix, OUT LBoostMatrix* pClassisVector)
{
    return m_pBoostTree->Predict(sampleMatrix, pClassisVector);
}

