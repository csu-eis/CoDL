
#include "LBayesClassifier.h"

#include <cmath>

#include <map>
using std::map;
#include <vector>
using std::vector;

/// @brief 贝叶斯分类器虚基类
class CBayesClassifier
{
public:
    /// @brief 析构函数
    virtual ~CBayesClassifier() = 0{}

    /// @brief 训练模型
    /// @param[in] problem 贝叶斯问题
    /// @return 成功返回true, 失败返回false, 参数错误的情况下会返回false
    virtual bool TrainModel(IN const LBayesProblem& problem) = 0;

    /// @brief 使用训练好的模型进行预测
    ///  
    /// 请保证需要预测的样本的特征长度和训练样本的特征长度相同
    /// @param[in] sample 需要预测的样本
    /// @param[out] pClassValue 存储预测结果, 不能为0
    /// @return 成功预测返回true, 失败返回false, 参数错误或模型未训练的情况下会返回false
    virtual bool Predict(IN const LBayesMatrix& sample, OUT int* pClassValue) = 0;
};

/// @brief 特征类别计数类
class CFeatureClassCount
{
public:
    /// @brief 将指定特征的指定的类别计数加1
    /// @param[in] featureValue 特征值
    /// @param[in] classValue 类别值
    void CountInc(IN int featureValue, IN int classValue)
    {
        m_featureClassMap[featureValue][classValue]++;
    }

    /// @brief 获取指定特征的指定类别的计数
    /// @param[in] featureValue 特征值
    /// @param[in] classValue 类别值
    /// @return 类别的计数
    unsigned int GetCount(IN int featureValue, IN int classValue)
    {
        return m_featureClassMap[featureValue][classValue];
    }

    /// @brief 获取指定特征的总计数
    /// @param[in] featureValue 特征值
    /// @return 特征值得总计数
    unsigned int GetTotalCount(IN int featureValue)
    {
        auto classMap = m_featureClassMap[featureValue];
        unsigned int totalCount = 0;
        for (auto iter = classMap.begin(); iter != classMap.end(); iter++)
        {
            totalCount += iter->second;
        }

        return totalCount;
    }

    /// @brief 清除数据
    void Clear()
    {
        m_featureClassMap.clear();
    }

private:
    map<int, map<int, unsigned int>> m_featureClassMap; ///< 特征映射, <特征值, <类别值, 类别计数>>
};

/// @brief 贝叶斯分类器(离散)实现类
class CBayesClassifierDiscrete : public CBayesClassifier
{    
public:
    /// @brief 构造函数
    CBayesClassifierDiscrete()
    {
        m_featureCount = 0;
        m_sampleCount = 0;
    }

    /// @brief 析构函数
    ~CBayesClassifierDiscrete()
    {

    }

    /// @brief 训练模型
    /// @param[in] problem 贝叶斯问题
    /// @return 成功返回true, 失败返回false, 参数错误的情况下会返回false
    virtual bool TrainModel(IN const LBayesProblem& problem)
    {
        // 进行参数检查
        if (problem.XMatrix.ColumnLen < 1)
            return false;
        if (problem.XMatrix.RowLen < 1)
            return false;
        if (problem.YVector.ColumnLen != 1)
            return false;
        if (problem.XMatrix.RowLen != problem.YVector.RowLen)
            return false;

        m_sampleClassCount.clear();
        m_featureClassCountList.clear();
        m_sampleCount = problem.XMatrix.RowLen;
        m_featureCount = problem.XMatrix.ColumnLen;
        for (unsigned int i = 0; i < m_featureCount; i++)
        {
            m_featureClassCountList.push_back(CFeatureClassCount());
        }

        for (unsigned int row = 0; row < problem.XMatrix.RowLen; row++)
        {
            int classValue = problem.YVector[row][0];
            m_sampleClassCount[classValue]++;

            for (unsigned int col = 0; col < problem.XMatrix.ColumnLen; col++)
            {
                int featureValue = problem.XMatrix[row][col];
                m_featureClassCountList[col].CountInc(featureValue, classValue);
            }

        }

        return true;
    }

    /// @brief 使用训练好的模型进行预测
    /// 请保证需要预测的样本的特征长度和训练样本的特征长度相同
    /// @param[in] sample 需要预测的样本
    /// @param[out] pClassValue 存储预测结果, 不能为0
    /// @return 成功预测返回true, 失败返回false, 参数错误或模型未训练的情况下会返回false
    virtual bool Predict(IN const LBayesMatrix& sample, OUT int* pClassValue)
    {
        // 检查参数
        if (1 != sample.RowLen)
            return false;
        if (m_featureCount != sample.ColumnLen)
            return false;
        if (0 == pClassValue)
            return false;

        if (m_sampleCount == 0)
            return false;


        float maxProb = 0;
        int bestClassValue;

        for (auto iter = m_sampleClassCount.begin(); iter != m_sampleClassCount.end(); iter++)
        {
            int classValue = iter->first;
            float prob = this->GetProbSampleInClass(sample, classValue);
            if (prob > maxProb)
            {
                maxProb = prob;
                bestClassValue = classValue;
            }
        }

        (*pClassValue) = bestClassValue;

        return true;
    }

private:
    /// @brief 获取指定样本属于指定类别的概率值, Pr(class | sample)
    /// @param[in] sample 样本
    /// @param[in] classValue 类别值
    /// @return 概率值
    float GetProbSampleInClass(IN const LBayesMatrix& sample, IN int classValue)
    {
        // 贝叶斯公式:
        // P(y|x) = P(x|y) * P(y) / P(x)
        // 对于每个样本来说P(x)值相同, 所以我们只需考察分母的值, 也就是P(x|y) * P(y)
        // 因为各个特征独立所以
        // P(x|y) * P(y) = P(a1|y) * P(a2|y) * ... * P(an|y) * P(y)

        unsigned int classCount = m_sampleClassCount[classValue];

        float prob = 1.0f;
        for (unsigned int col = 0; col < sample.ColumnLen; col++)
        {
            int featureValue = sample[0][col];
            unsigned int featureClassCount = m_featureClassCountList[col].GetCount(featureValue, classValue);
            float basicProb = (float)featureClassCount/(float)classCount;
            unsigned int featureTotalCount = m_featureClassCountList[col].GetTotalCount(featureValue);
            // w = 0.5 + totalCount/(1 + totalCount) * (basicProb - 0.5)
            // 使用权重概率可以解决以下问题:
            // 特征值在指定分类出现次数为0导致概率为0的情况
            float weightedProb = 0.5f + (float)featureTotalCount/(1.0f + (float)featureTotalCount) * (basicProb - 0.5f);

            prob *= weightedProb;
        }

        prob *= (float)classCount/(float)m_sampleCount;

        return prob;
    }

private:
    vector<CFeatureClassCount> m_featureClassCountList; ///< 特征类别计数组
    map<int, unsigned int> m_sampleClassCount; ///< 训练样本类别计数
    unsigned int m_featureCount; ///< 样本特征数量
    unsigned int m_sampleCount; ///< 训练样本总数
};


/// @brief 特征类别数据结构
struct CFeatureClassData
{
    map<int, vector<int>> DataMap; ///< 类别数据映射, <类别值, 数据列表>
};

/// @brief 高斯分布结构
struct CGauss
{
    float Mean; ///< 均值
    float Div; ///< 标准差
};

/// @brief 特征类别高斯分布结构
struct CFeatureClassGauss
{
    map<int, CGauss> GaussMap; ///< 类别高斯分布映射, <类别值, 高斯分布>
};

/// @brief 贝叶斯分类器连续(非离散)实现类
class CBayesClassifierContinues : public CBayesClassifier
{   
public:
    /// @brief 构造函数
    CBayesClassifierContinues()
    {
        m_featureCount = 0;
        m_sampleCount = 0;
    }

    /// @brief 析构函数
    ~CBayesClassifierContinues()
    {
        m_featureCount = 0;
        m_sampleCount = 0;
    }

    /// @brief 训练模型
    /// @param[in] problem 贝叶斯问题
    /// @return 成功返回true, 失败返回false, 参数错误的情况下会返回false
    virtual bool TrainModel(IN const LBayesProblem& problem)
    {
        // 进行参数检查
        if (problem.XMatrix.ColumnLen < 1)
            return false;
        if (problem.XMatrix.RowLen < 1)
            return false;
        if (problem.YVector.ColumnLen != 1)
            return false;
        if (problem.XMatrix.RowLen != problem.YVector.RowLen)
            return false;

        vector<CFeatureClassData> featureClassDataList; ///< 特征类别数据列表

        m_sampleClassCount.clear();
        m_featureClassGaussList.clear();
        m_sampleCount = problem.XMatrix.RowLen;
        m_featureCount = problem.XMatrix.ColumnLen;
        for (unsigned int i = 0; i < m_featureCount; i++)
        {
            featureClassDataList.push_back(CFeatureClassData());
            m_featureClassGaussList.push_back(CFeatureClassGauss());
        }

        // 将每列特征值按类别归类
        for (unsigned int row = 0; row < problem.XMatrix.RowLen; row++)
        {
            int classValue = problem.YVector[row][0];
            m_sampleClassCount[classValue]++;

            for (unsigned int col = 0; col < problem.XMatrix.ColumnLen; col++)
            {
                int featureValue = problem.XMatrix[row][col];
                CFeatureClassData& featureClassData = featureClassDataList[col];
                featureClassData.DataMap[classValue].push_back(featureValue);
            }

        }


        // 计算数据的高斯分布
        for (unsigned int i = 0; i < featureClassDataList.size(); i++)
        {
            for (auto iter = m_sampleClassCount.begin(); iter != m_sampleClassCount.end(); iter++)
            {
                int classValue = iter->first;

                CFeatureClassData& featureClassData = featureClassDataList[i];
                CGauss gauss = this->CalculateGauss(featureClassData.DataMap[classValue]);
                if (gauss.Div == 0.0f)// 方差为0, 表示数据有问题, 无法使用高斯分布
                {
                    m_featureCount = 0;
                    m_sampleCount = 0;
                    m_sampleClassCount.clear();
                    m_featureClassGaussList.clear();
                    return false;
                }

                m_featureClassGaussList[i].GaussMap[classValue] = gauss;

            }
        }

        return true;
    }

    /// @brief 使用训练好的模型进行预测
    /// 请保证需要预测的样本的特征长度和训练样本的特征长度相同
    /// @param[in] sample 需要预测的样本
    /// @param[out] pClassValue 存储预测结果, 不能为0
    /// @return 成功预测返回true, 失败返回false, 参数错误或模型未训练的情况下会返回false
    virtual bool Predict(IN const LBayesMatrix& sample, OUT int* pClassValue)
    {
        // 检查参数
        if (1 != sample.RowLen)
            return false;
        if (m_featureCount != sample.ColumnLen)
            return false;
        if (0 == pClassValue)
            return false;

        if (m_sampleCount == 0)
            return false;


        float maxProb = 0;
        int bestClassValue;

        for (auto iter = m_sampleClassCount.begin(); iter != m_sampleClassCount.end(); iter++)
        {
            int classValue = iter->first;
            float prob = this->GetProbSampleInClass(sample, classValue);
            if (prob > maxProb)
            {
                maxProb = prob;
                bestClassValue = classValue;
            }
        }

        (*pClassValue) = bestClassValue;

        return true;
    }

private:
    /// @brief 获取指定样本属于指定类别的概率值, Pr(class | sample)
    /// @param[in] sample 样本
    /// @param[in] classValue 类别值
    /// @return 概率值
    float GetProbSampleInClass(IN const LBayesMatrix& sample, IN int classValue)
    {
        // 贝叶斯公式:
        // P(y|x) = P(x|y) * P(y) / P(x)
        // 对于每个样本来说P(x)值相同, 所以我们只需考察分母的值, 也就是P(x|y) * P(y)
        // 因为各个特征独立所以
        // P(x|y) * P(y) = P(a1|y) * P(a2|y) * ... * P(an|y) * P(y)

        unsigned int classCount = m_sampleClassCount[classValue];

        float prob = 1.0f;
        for (unsigned int col = 0; col < sample.ColumnLen; col++)
        {
            int featureValue = sample[0][col];
            const CGauss& gauss = m_featureClassGaussList[col].GaussMap[classValue];
            float temp1 = 1.0f/ (sqrt(2.0f * 3.14159f) * gauss.Div);
            float temp2 = (float)featureValue-gauss.Mean;
            float temp3 = exp(-1.0f * temp2 * temp2 / (2.0f * gauss.Div * gauss.Div));
            float gaussProb = temp1 * temp3;

            prob *= gaussProb;
        }

        prob *= (float)classCount/(float)m_sampleCount;

        return prob;
    }

    /// @brief 计算数据的高斯分布
    /// @param[in] dataList 数据列表
    /// @return 高斯分布结构
    CGauss CalculateGauss(IN const vector<int>& dataList)
    {
        CGauss gauss;
        gauss.Mean = 0.0f;
        gauss.Div = 0.0f;

        if (dataList.size() < 1)
            return gauss;

        float total = 0.0f;
        for (unsigned int i = 0; i < dataList.size(); i++)
        {
            total += (float)dataList[i];
        }

        gauss.Mean = total/dataList.size();

        float div = 0.0f;
        for (unsigned int i = 0; i < dataList.size(); i++)
        {
            float temp = (float)dataList[i]-gauss.Mean;
            div += (temp * temp);
        }
        div = div/dataList.size();

        gauss.Div = sqrt(div);

        return gauss;
    }

private:
    vector<CFeatureClassGauss> m_featureClassGaussList; ///< 特征类别高斯分布列表
    map<int, unsigned int> m_sampleClassCount; ///< 训练样本类别计数
    unsigned int m_featureCount; ///< 样本特征数量
    unsigned int m_sampleCount; ///< 训练样本总数
};

LBayesClassifier::LBayesClassifier()
{
    m_pBayesClassifier = 0;
}

LBayesClassifier::~LBayesClassifier()
{
    if (0 != m_pBayesClassifier)
    {
        delete m_pBayesClassifier;
        m_pBayesClassifier = 0;
    }
}

bool LBayesClassifier::TrainModel(IN const LBayesProblem& problem)
{
    if (0 != m_pBayesClassifier)
    {
        delete m_pBayesClassifier;
        m_pBayesClassifier = 0;
    }

    if (problem.FeatureDistribution == BAYES_FEATURE_DISCRETE)
        m_pBayesClassifier = new CBayesClassifierDiscrete();
    else if (problem.FeatureDistribution == BAYES_FEATURE_CONTINUS)
        m_pBayesClassifier = new CBayesClassifierContinues();
    else
        return false;


    return m_pBayesClassifier->TrainModel(problem);
}

bool LBayesClassifier::Predict(IN const LBayesMatrix& sample, OUT int* pClassValue)
{
    return m_pBayesClassifier->Predict(sample, pClassValue);
}
