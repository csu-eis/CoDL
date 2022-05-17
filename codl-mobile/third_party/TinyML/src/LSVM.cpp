
#include "LSVM.h"

#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <cmath>

#include <Windows.h>

#ifndef LMAX
#define LMAX(a,b)    (((a) > (b)) ? (a) : (b))
#endif

#ifndef LMIN
#define LMIN(a,b)    (((a) < (b)) ? (a) : (b))
#endif

/// @brief SVM解结构
struct LSVMSolution
{
    /// @brief 构造函数
    ///  
    /// @param[in] m 样本数量
    LSVMSolution(unsigned int m)
    {
        this->AVector.Reset(m, 1);
        for (unsigned int i = 0; i < m; i++)
        {
            this->AVector[i][0] = 0.0f;
        }

        this->B = 0.0f;
    }

    float B; ///< 分割超平面的截距
    LSVMMatrix AVector; ///< alpha向量(列向量)
};

LSVMKRBF::LSVMKRBF(IN float gamma)
{
    m_gamma = gamma;
}

LSVMKRBF::~LSVMKRBF()
{

}

float LSVMKRBF::Translate(IN const LSVMMatrix& vectorA, IN const LSVMMatrix& vectorB) 
{
    LSVMMatrix::SUB(vectorA, vectorB, m_deltaRow);
    LSVMMatrix::T(m_deltaRow, m_deltaRowT);
    LSVMMatrix::MUL(m_deltaRow, m_deltaRowT, m_k);

    return exp(m_k[0][0]/(-2 * m_gamma * m_gamma));
}

/// @brief 原始函数(不使用核函数, 直接计算内积)
class LSVMKOriginal : public ISVMKernelFunc
{
public:
    /// @brief 构造函数
    LSVMKOriginal()
    {

    }

    /// @brief 析构函数
    ~LSVMKOriginal()
    {

    }

    /// @brief 转换函数
    virtual float Translate(IN const LSVMMatrix& vectorA, IN const LSVMMatrix& vectorB)
    {
        LSVMMatrix::T(vectorB, m_bT);
        LSVMMatrix::MUL(vectorA, m_bT, m_abT);
        return m_abT[0][0];
    }

private:

    /*
    以下变量被设为成员变量为优化程序效率目的
    */
    LSVMMatrix m_bT;
    LSVMMatrix m_ab;
    LSVMMatrix m_abT;
};

/// @brief 支持向量机实现类
class CSVM
{
public:
    /// @brief 构造函数
    explicit CSVM(IN const LSVMParam& param)
    {
        srand((unsigned int)time(0));

        m_pProblem = 0;
        m_pSolution = 0;
        m_pKMatrix = 0;
        m_pKernelFunc = NULL;

        m_pParam = new LSVMParam;
        (*m_pParam) = param;

        // 获取核函数接口
        if (m_pParam->PKernelFunc != 0)
        {
            m_pKernelFunc = m_pParam->PKernelFunc;
        }
        else
        {
            m_pKernelFunc = &m_kOriginal;
        }
    }

    /// @brief 析构函数
    ~CSVM()
    {
        if (m_pSolution != 0)
        {
            delete m_pSolution;
            m_pSolution = 0;
        }

        if (m_pKMatrix != 0)
        {
            delete m_pKMatrix;
            m_pKMatrix = 0;
        }

        if (m_pParam != 0)
        {
            delete m_pParam;
            m_pParam = 0;
        }
    }

    /// @brief 训练模型
    bool TrainModel(IN const LSVMProblem& problem, OUT LSVMResult& result)
    {
        // 进行参数检查
        if (problem.XMatrix.ColumnLen < 1)
            return false;
        if (problem.XMatrix.RowLen < 2)
            return false;
        if (problem.YVector.ColumnLen != 1)
            return false;
        if (problem.XMatrix.RowLen != problem.YVector.RowLen)
            return false;

        m_pProblem = &problem;

        // 初始化解结构
        if (m_pSolution != 0)
        {
            delete m_pSolution;
            m_pSolution = 0;
        }
        m_pSolution = new LSVMSolution(problem.XMatrix.RowLen);

        // 通过核函数计算所有样本之间在高纬空间的的内积, 并存储在K矩阵中
        if (m_pKMatrix != 0)
        {
            delete m_pKMatrix;
            m_pKMatrix = 0;
        }
        m_pKMatrix = new LSVMMatrix(problem.XMatrix.RowLen, problem.XMatrix.RowLen);
        LSVMMatrix sampleA;
        LSVMMatrix sampleB;
        for (unsigned int i = 0; i < problem.XMatrix.RowLen; i++)
        {
            problem.XMatrix.GetRow(i, sampleA);
            for (unsigned int j = 0; j < problem.XMatrix.RowLen; j++)
            {
                problem.XMatrix.GetRow(j, sampleB);
                (*m_pKMatrix)[i][j] = m_pKernelFunc->Translate(sampleA, sampleB);
            }
        }

        // 训练模型
        result.IterCount = this->SMOTrainModel(*m_pProblem, *m_pSolution);

        // 获取支持向量的索引
        unsigned int supportVectorNum = 0;
        for (unsigned int i = 0; i < m_pSolution->AVector.RowLen; i++)
        {
            if (m_pSolution->AVector[i][0] != 0.0f)
                supportVectorNum++;
        }
        m_supportVectorIndex.Reset(supportVectorNum, 1);
        unsigned int j = 0;
        for (unsigned int i = 0; i < m_pSolution->AVector.RowLen; i++)
        {
            if (m_pSolution->AVector[i][0] != 0.0f)
            {
                m_supportVectorIndex[j][0] = i;
                j++;
            }
        }

        result.SupportVectorNum = supportVectorNum;


        return true;
    }

    /// @brief 使用训练好的模型进行预测
    bool Predict(IN const LSVMMatrix& sampleSet, OUT LSVMMatrix& yVector)
    {
        // 检查参数
        if (this->m_pProblem == 0)
            return false;
        if (sampleSet.RowLen < 1)
            return false;
        if (sampleSet.ColumnLen != this->m_pProblem->XMatrix.ColumnLen)
            return false;

        yVector.Reset(sampleSet.RowLen, 1);
       

        LSVMMatrix AY;
        LSVMMatrix::DOTMUL(this->m_pSolution->AVector, this->m_pProblem->YVector, AY); // 列向量
        LSVMMatrix AYT = AY.T(); // 行向量

        LSVMMatrix sampleA;
        LSVMMatrix sampleB;
        LSVMMatrix KColumn(this->m_pProblem->XMatrix.RowLen, 1); // 列向量

        for (unsigned int row = 0; row < sampleSet.RowLen; row++)
        {
            sampleSet.GetRow(row, sampleB);

            // 只对支持向量做内积, 节省时间
            for (unsigned int i = 0; i < KColumn.RowLen; i++)
            {
                KColumn[i][0] = 0.0f;
            }
            for (unsigned int i = 0; i < m_supportVectorIndex.RowLen; i++)
            {
                unsigned int j = m_supportVectorIndex[i][0];
                sampleA = this->m_pProblem->XMatrix.GetRow(j);
                KColumn[j][0] = m_pKernelFunc->Translate(sampleA, sampleB);
            }
            LSVMMatrix AYTK = AYT * KColumn;
            if (AYTK[0][0] + this->m_pSolution->B >= 0.0f)
                yVector[row][0] = 1.0f;
            else
                yVector[row][0] = -1.0f;
        }
        
        return true;

    }

private:
    /// @brief 产生一个不等于i的随机数, 范围[0, max]
    /// @param[in] i 随机数中被过滤的值
    /// @param[in] max 随机数中的最大值, 要求大于0
    /// @return 随机数
    int SelectRand(IN unsigned int i, IN unsigned int max)
    {
        int j = i; 
        while (j == i)
        {
            j = rand()%(max + 1);
        }
        return j;
    }

    /// @brief 使用启发式方法选择第二个alpha
    /// @param[in] i 第一个alpha索引
    /// @param[in] E 所有样本误差向量
    /// @return 第二个alpha索引
    unsigned int SelectSecondAlpha(IN unsigned int i, IN const LSVMMatrix& E)
    {
        unsigned int j = 0;
        float maxDeltaE = 0.0f;

        for (unsigned int k = 0; k < E.RowLen; k++)
        {

            float deltaE = abs(E[i][0] - E[k][0]);
            if (deltaE > maxDeltaE)
            {
                maxDeltaE = deltaE;
                j = k;
            }
        }

        if (maxDeltaE == 0.0f)
        {
            j = this->SelectRand(i, E.RowLen-1);
        }

        return j;

    }

    /// @brief 修正a值
    ///  a应处于min~max, 当a > max时a = max, 当a < min时a = min
    /// @param[in] a 需要修正的值
    /// @param[in] min 最小值
    /// @param[in] max 最大值
    /// @return 修正后的值
    float ClipAlpha(IN float a, IN float min, IN float max)
    {
        if (a > max)
            a = max;
        if (a < min)
            a = min;

        return a;
    }

    /// @brief 根据解计算出所有样本的误差
    /// @param[in] solution 解
    /// @param[in] k 样本索引
    /// @return 误差值
    void CalculateError(IN const LSVMSolution& solution, OUT LSVMMatrix& errorVector)
    {
        LSVMMatrix AY;
        LSVMMatrix::DOTMUL(solution.AVector, m_pProblem->YVector, AY); // 列向量
        LSVMMatrix AYT = AY.T(); // 行向量

        LSVMMatrix KColumn;
        LSVMMatrix AYTK;
        for (unsigned int i = 0; i < m_pProblem->XMatrix.RowLen; i++)
        {
            m_pKMatrix->GetColumn(i, KColumn); // 列向量
            LSVMMatrix::MUL(AYT, KColumn, AYTK);
            float E = AYTK[0][0] + solution.B - m_pProblem->YVector[i][0]; // 样本i标签的误差
            errorVector[i][0] = E;
        }
    }

    /// @brief SMO训练算法
    /// @param[in] problem 原始问题
    /// @param[out] solution 问题的解
    /// @return 遍历次数
    unsigned int SMOTrainModel(IN const LSVMProblem& problem, OUT LSVMSolution& solution)
    {
        const unsigned int M = problem.XMatrix.RowLen; // 样本数量

        LSVMMatrix E(M, 1); ///< 误差缓存向量(列向量)
        for (unsigned int i = 0; i < M; i++)
        {
            E[i][0] = 0.0f;
        }

        this->CalculateError(solution, E);

        bool entireSet = true;
        bool alphaChanged = false;

        unsigned int iter = 0;
        while (iter < m_pParam->MaxIterCount)
        {
            alphaChanged = false;

            // 遍历整个alpha集合
            if (entireSet)
            {
                for (unsigned int i = 0; i < M; i++)
                {
                    if (!this->SMOOptimizeAlpha(i, E, solution))
                        continue;

                    alphaChanged = true;
                    this->CalculateError(solution, E);
                }

            }

            // 遍历不在边界上的alpha
            if (!entireSet)
            {
                for (unsigned int i = 0; i < M; i++)
                {
                    if (solution.AVector[i][0] == 0 || solution.AVector[i][0] == m_pParam->C)
                        continue;

                    if (!this->SMOOptimizeAlpha(i, E, solution))
                        continue;

                    alphaChanged = true;
                    this->CalculateError(solution, E);
                }
            }

            iter++;

            // 遍历整个集合并且alpha值都没改变, 那么结束遍历
            if (entireSet && !alphaChanged)
                break;

            // 遍历整个集合并且改变了alpha值, 那么下次切换到遍历非边界集合
            if (entireSet && alphaChanged)
                entireSet = false;

            // 遍历非边界集合并且alpha值得到了改变, 那么下次继续遍历非边界集合
            if (!entireSet && alphaChanged)
                continue;

            // 遍历非边界集合并且alpha值没有改变, 那么下次切换到遍历整个集合
            if (!entireSet && !alphaChanged)
                entireSet = true;
        }

        return iter;

    }

    /// @brief SMO优化Alpha 
    /// @param[in] fristAlpha 需要优化的alpha的索引
    /// @param[in] error 误差缓存向量
    /// @param[out] solution 问题的解
    /// @return 成功优化返回true, 优化失败返回false
    bool SMOOptimizeAlpha(
        IN unsigned int fristAlpha,   
        IN const LSVMMatrix& error, 
        OUT LSVMSolution& solution)
    {
        const unsigned int i = fristAlpha;
        const LSVMMatrix& X = m_pProblem->XMatrix; // 样本矩阵
        const LSVMMatrix& Y = m_pProblem->YVector; // 标签向量(列向量)
        const unsigned int M = X.RowLen; // 样本数量
        const LSVMMatrix& K = *m_pKMatrix; // K矩阵

        const float C = m_pParam->C; // 常数C

        float& B = solution.B; // 分割超平面的截距
        LSVMMatrix& A = solution.AVector; // alpha向量(列向量)

        const LSVMMatrix& E = error; ///< 误差缓存向量(列向量)

        float marginXi = E[i][0] * Y[i][0] + 1;

        // KKT条件::
        // alpha如果小于C, 那么margin应该大于等于1, 如不符合那么违反KKT条件, 则需要优化
        // alpha如果大于0, 那么margin应该小于等于1, 如不符合那么违反KKT条件, 则需要优化
        if (((A[i][0] < C) && (marginXi < 1)) ||
            ((A[i][0] > 0) && (marginXi > 1)))
        {
            unsigned int j = this->SelectSecondAlpha(i, E);

            float LAj = 0.0f; // alpha j 的下届
            float HAj = 0.0f; // alpha j 的上届
            // 计算alpha j 的上下界
            if (Y[i][0] != Y[j][0])
            {
                LAj = LMAX(0, A[j][0] - A[i][0]);
                HAj = LMIN(C, C + A[j][0] - A[i][0]);
            }
            if (Y[i][0] == Y[j][0])
            {
                LAj = LMAX(0, A[j][0] - A[i][0] - C);
                HAj = LMIN(C, A[j][0] - A[i][0]);
            }

            if (LAj == HAj)
            {
                return false;
            }

            float eta = 2 * K[i][j] - K[i][i] - K[j][j];

            if (eta == 0.0f)
            {
                return false;
            }

            // 计算新的alpha j 
            float AjOld = A[j][0];
            A[j][0] -= Y[j][0] * (E[i][0] - E[j][0]) / eta;
            // 裁剪alpha j
            A[j][0] = this->ClipAlpha(A[j][0], LAj, HAj);

            // 如果alpha j 改变量太小, 则退出
            if (abs(A[j][0] - AjOld) < 0.0001)
            {
                return false;
            }

            // 计算新的alpha i
            float AiOld = A[i][0];
            A[i][0] += Y[i][0] * Y[j][0] * (AjOld - A[j][0]);

            // 计算截距B的值
            float B1 = B - E[i][0] - Y[i][0] * (A[i][0] - AiOld) * K[i][i] - 
                Y[j][0] * (A[j][0] - AjOld) * K[i][j];
            float B2 = B - E[j][0] - Y[i][0] * (A[i][0] - AiOld) * K[i][j] -
                Y[j][0] * (A[j][0] - AjOld) * K[j][j];

            if (A[i][0] > 0 && A[i][0] < C)
            {
                B = B1;
            }
            else if (A[j][0] > 0 && A[j][0] < C)
            {
                B = B1;
            }
            else
            {
                B = (B1 + B2)/2;
            }

            return true;

        }

        return false;
    }

private:
    const LSVMProblem* m_pProblem; ///< 原始问题
    LSVMParam* m_pParam; ///< 参数
    LSVMSolution* m_pSolution; ///< SVM的解
    LSVMMatrix* m_pKMatrix; ///< K矩阵
    ISVMKernelFunc* m_pKernelFunc; ///< 核函数接口
    LSVMKOriginal m_kOriginal; ///< 原始核函数
    LMatrix<unsigned int> m_supportVectorIndex; ///< 记录支持向量的样本的索引(列向量)
};

LSVM::LSVM(IN const LSVMParam& param)
{
    m_pSVM = new CSVM(param);
}

LSVM::~LSVM()
{
    if (m_pSVM != 0)
    {
        delete m_pSVM;
    }
}

bool LSVM::TrainModel(IN const LSVMProblem& problem, OUT LSVMResult& result)
{
    return m_pSVM->TrainModel(problem, result);
}

bool LSVM::Predict(IN const LSVMMatrix& sampleSet, OUT LSVMMatrix& yVector)
{
    return m_pSVM->Predict(sampleSet, yVector);
}




