

#include "LPerceptron.h"

#include <vector>
using std::vector;

/// @brief 感知机 实现类
class CPerceptron
{
public:
    /// @brief 构造函数
    CPerceptron()
    {
        m_learningRate = 1.0f;
    }

    /// @brief析构函数
    ~CPerceptron()
    {

    }

    /// @brief 设置学习速率(默认值为1.0f)
    /// 详细解释见头文件LPerceptron中的声明  
    bool SetLearningRate(IN float rate)
    {
        if (rate <= 0)
            return false;

        this->m_learningRate = rate;

        return true;
    }

    /// @brief 训练模型
    /// 详细解释见头文件LPerceptron中的声明 
    bool TrainModel(IN const LPerceptronProblem& problem)
    {
        const LPerceptronMatrix& X = problem.XMatrix;
        const LPerceptronMatrix& Y = problem.YVector;
        vector<float>& W = this->m_weightVector;
        float& B = this->m_b;
        const float Alpha = this->m_learningRate;

        // 检查参数 符不符合要求
        if (X.ColumnLen < 1)
            return false;
        if (X.RowLen < 2)
            return false;
        if (Y.ColumnLen != 1)
            return false;
        if (X.RowLen != Y.RowLen)
            return false;

        for (unsigned int n = 0; n < Y.RowLen; n++)
        {
            if (Y[n][0] != LPERCEPTRON_SUN &&
                Y[n][0] != LPERCEPTRON_MOON)
                return false;
        }


        // 初始化权重向量和截距
        W.resize(X.ColumnLen, 0.0f);
        B = 0.0f;


        bool bErrorClass = false; // 标记是否存在错误分类
        while (true)
        {
            // 检验每一个训练样本查看是否被错误分类
            for (unsigned int i = 0; i < X.RowLen; i++)
            {
                float WXi = 0.0f;
                for (unsigned int n = 0; n < W.size(); n++)
                {
                    WXi += W[n] * X[i][n];
                }

                // 误分类点
                if (Y[i][0] * (WXi + B) <= 0)
                {
                    bErrorClass = true;

                    // 更新W和B
                    for (unsigned int n = 0; n < W.size(); n++)
                    {
                        W[n] = W[n] + Alpha * Y[i][0] * X[i][n];
                    }
                    B = B + Alpha * Y[i][0];
                }

            }

            // 如果没有错误分类则退出循环
            if (!bErrorClass)
            {
                break;
            }

            // 如果有错误分类则继续
            if (bErrorClass)
            {
                bErrorClass = false;
                continue;
            }
        }

        return true;
    }

    /// @brief 使用训练好的模型进行预测(单样本预测)
    /// 详细解释见头文件LPerceptron中的声明
    float Predict(IN const LPerceptronMatrix& sample)
    {
        if (sample.RowLen != 1)
            return 0.0f;

        if (this->m_weightVector.size() < 1)
            return 0.0f;

        if (this->m_weightVector.size() != sample.ColumnLen)
            return 0.0f;

        float y = 0.0f;
        for (unsigned int i = 0; i < sample.ColumnLen; i++)
        {
            y += sample[0][i] * m_weightVector[i];
        }

        y += m_b;

        if (y >= 0)
            return LPERCEPTRON_SUN;
        else
            return LPERCEPTRON_MOON;
    }

private:

    float m_learningRate; ///< 学习速率
    float m_b; ///< 分割超平面的截距
    vector<float> m_weightVector; ///< 权重向量(列向量), 列数为1, 行数为样本的特征数
};

LPerceptron::LPerceptron()
{
    m_pPerceptron = 0;
    m_pPerceptron = new CPerceptron();
}

LPerceptron::~LPerceptron()
{
    if (0 != m_pPerceptron)
    {
        delete m_pPerceptron;
        m_pPerceptron = 0;
    }
}

bool LPerceptron::SetLearningRate(IN float rate)
{
    return m_pPerceptron->SetLearningRate(rate);
}

bool LPerceptron::TrainModel(IN const LPerceptronProblem& problem)
{
    return m_pPerceptron->TrainModel(problem);
}

float LPerceptron::Predict(IN const LPerceptronMatrix& sample)
{
    return m_pPerceptron->Predict(sample);
}