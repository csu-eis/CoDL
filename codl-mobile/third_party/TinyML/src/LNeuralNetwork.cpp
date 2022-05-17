
#include "LNeuralNetwork.h"

#include <cmath>
#include <cstdlib>

#include <vector>
using std::vector;

/// @brief 产生随机小数, 范围0~1
/// @return 随机小数
static float RandFloat()           
{
    return (rand())/(RAND_MAX + 1.0f);
}

/// @brief 产生随机小数, 范围-1~1
/// @return 随机小数
static float RandClamped()     
{
    return RandFloat() - RandFloat();
}

/// @brief BP网络中的神经元
/// 该类中的方法不会检查参数的有效性
class CBPNeuron
{
public:
    /// @brief 构造函数
    /// @param[in] inputNum 输入个数, 必须大于等于1
    explicit CBPNeuron(IN unsigned int inputNum)
    {
        m_weightList.resize(inputNum + 1);
        for (unsigned int i = 0; i < m_weightList.size(); i++)
        {
            m_weightList[i] = RandClamped();
        }
    }

    /// @brief 析构函数
    ~CBPNeuron()
    {

    }

    /// @brief 激活神经元
    /// @param[in] inputVector 输入向量(行向量), 向量长度必须等于神经元的输入个数
    /// @return 激活值, 激活值范围0~1
    double Active(IN const LNNMatrix& inputVector)
    {
        double sum = 0.0;
        for (unsigned int i = 0; i < inputVector.ColumnLen; i++)
        {
            sum += inputVector[0][i] * m_weightList[i];
        }

        sum += m_weightList[inputVector.ColumnLen] * 1.0f;

        return this->Sigmoid(sum);
    }

    /// @brief 反向训练
    /// @param[in] inputList 该神经元的输入列表
    /// @param[in] error 该神经元的输出误差
    /// @param[in] learnRate 学习速率
    /// @param[out] pFrontErrorList 存储前层输出误差列表 , 不能为0
    /// @return 成功返回true, 失败返回false, 参数有误会失败
    bool BackTrain(
        IN const vector<double>& inputList, 
        IN double error, 
        IN double learnRate,
        OUT vector<double>* pFrontErrorList)
    {
        if (inputList.size() != (m_weightList.size()-1))
            return false;

        if (0 == pFrontErrorList)
            return false;

        for (unsigned int i = 0; i < inputList.size(); i++)
        {
            m_weightList[i] += learnRate * inputList[i] * error;
            (*pFrontErrorList)[i] = m_weightList[i] * error;
        }

        m_weightList[m_weightList.size()-1] += learnRate * 1.0f * error;

        return true;

    }

private:
    /// @brief S型激活函数
    /// @param[in] input 激励值
    /// @return 激活值
    double Sigmoid(IN double input)
    {
        return ( 1.0 / ( 1.0 + exp(-input)));
    }

private:
    vector<double> m_weightList; ///< 权重列表, 权重值最后一项为偏移值
};

/// @brief BP网络中的神经元层
class CBPNeuronLayer
{
public:
    /// @brief 构造函数
    /// @param[in] neuronInputNum 神经元输入个数, 必须为大于等于1的数
    /// @param[in] neuronNum 神经元个数, 必须为大于等于1的数
    CBPNeuronLayer(IN unsigned int neuronInputNum, IN unsigned int neuronNum)
    {
        m_neuronInputNum = neuronInputNum;

        for (unsigned int i = 0; i < neuronNum; i++)
        {
            CBPNeuron* pNeuron = new CBPNeuron(neuronInputNum);
            m_neuronList.push_back(pNeuron);
        }

        m_inputList.resize(neuronInputNum);
        m_frontErrorList.resize(neuronInputNum);
    }

    ~CBPNeuronLayer()
    {
        for (unsigned int i = 0; i < m_neuronList.size(); i++)
        {
            if (m_neuronList[i] != 0)
            {
                delete m_neuronList[i];
                m_neuronList[i] = 0;
            }
            
        }
    }

    /// @brief 激活神经元层
    /// @param[in] inputVector 输入向量(行向量), 向量长度必须等于神经元的输入个数
    /// @param[out] pOutputVector 输出向量(行向量), 存储神经元层的输出, 输出向量的长度等于神经元的个数, 该值不能为0
    /// @return 成功返回true, 失败返回false, 参数有误会失败
    bool Active(IN const LNNMatrix& inputVector, OUT LNNMatrix* pOutputVector)
    {
        if (m_neuronInputNum < 1 || m_neuronList.size() < 1)
            return false;

        if (inputVector.ColumnLen != m_neuronInputNum)
            return false;

        if (0 == pOutputVector)
            return false;

        for (unsigned int i = 0; i < inputVector.ColumnLen; i++)
        {
            m_inputList[i] = inputVector[0][i];
        }

        for (unsigned int i = 0; i < m_neuronList.size(); i++)
        {
            (*pOutputVector)[0][i] = m_neuronList[i]->Active(inputVector);
        }

        return true;
    }

    /// @brief 反向训练
    /// @param[in] opErrorList 本层的输出误差列表
    /// @param[in] learnRate 学习速率
    /// @param[out] pFrontOpErrorList 存储前一层的输出误差列表, 不能为0
    /// @return 成功返回true, 失败返回false, 参数有误会失败
    bool BackTrain(IN const vector<double>& opErrorList, IN double learnRate, OUT vector<double>* pFrontOpErrorList)
    {
        if (opErrorList.size() != m_neuronList.size())
            return false;

        if (0 == pFrontOpErrorList)
            return false;

        // 初始化前一层的输出误差列表
        for (unsigned int i = 0; i < pFrontOpErrorList->size(); i++)
        {
            (*pFrontOpErrorList)[i] = 0.0f; 
        }

        // 对每个神经元进行反向训练, 并获取每个神经元对前层输出的误差列表
        for (unsigned int i = 0; i < m_neuronList.size(); i++)
        {
            m_neuronList[i]->BackTrain(m_inputList, opErrorList[i], learnRate, &m_frontErrorList);

            // 累加各个神经元的误差
            for (unsigned int j = 0; j < pFrontOpErrorList->size(); j++)
            {
                (*pFrontOpErrorList)[j] += m_frontErrorList[j];
            }
        }

        for (unsigned int i = 0; i < pFrontOpErrorList->size(); i++)
        {
            (*pFrontOpErrorList)[i] *= m_inputList[i] * (1.0f-m_inputList[i]);
        }

        return true;

    }

private:
    unsigned int m_neuronInputNum; ///< 神经元输入个数
    vector<CBPNeuron*> m_neuronList; ///< 神经元列表
    vector<double> m_inputList; ///< 神经元的输入值列表, 每次调用Active函数被更新
    vector<double> m_frontErrorList; ///< 神经元前层输出误差
};

/// @brief BP网络实现类
class CBPNetwork
{
public:
    /// @brief 构造函数
    CBPNetwork(IN const LBPNetworkPogology& pogology)
    {
        m_networkPogology.InputNumber = 0;
        m_networkPogology.OutputNumber = 0;
        m_networkPogology.HiddenLayerNumber = 0;
        m_networkPogology.NeuronsOfHiddenLayer = 0;

        m_bInitDone = false;

        this->Init(pogology);
    }

    /// @brief 析构函数
    ~CBPNetwork()
    {
        this->CleanUp();
    }

    /// @brief 训练BP网络
    /// 详细解释见头文件LBPNetwork中的声明
    bool Train(IN const LNNMatrix& inputMatrix, IN const LNNMatrix& outputMatrix, IN float rate)
    {
        if (!m_bInitDone)
            return false;

        // 检查参数
        if (inputMatrix.RowLen < 1)
            return false;

        if (inputMatrix.ColumnLen != m_networkPogology.InputNumber)
            return false;

        if (outputMatrix.ColumnLen != m_networkPogology.OutputNumber)
            return false;

        if (outputMatrix.RowLen != inputMatrix.RowLen)
            return false;


        // 针对每个训练样本, 分别训练
        for (unsigned int row = 0; row < inputMatrix.RowLen; row++)
        {
            inputMatrix.GetRow(row, m_inputVectorForTrain);
            this->Active(m_inputVectorForTrain, &m_outputVectorForTrain);

            // 计算输出层误差
            vector<double>& errorList = m_layerErrorList[m_layerErrorList.size()-1];
            for (unsigned int i = 0; i < m_outputVectorForTrain.ColumnLen; i++)
            {
                errorList[i] = outputMatrix[row][i]-m_outputVectorForTrain[0][i];
                errorList[i] *= m_outputVectorForTrain[0][i] * (1.0f-m_outputVectorForTrain[0][i]);
            }

            // 从后向前进行反向训练
            for (int i = int(m_layerList.size()-1); i >= 0; i--)
            {
                m_layerList[i]->BackTrain(m_layerErrorList[i + 1], rate, &m_layerErrorList[i]);
            }

        }

        
        return true;
    }

    /// @brief 激活BP网络
    /// 详细解释见头文件LBPNetwork中的声明
    bool Active(IN const LNNMatrix& inputMatrix, OUT LNNMatrix* pOutputMatrix)
    {
        if (!m_bInitDone)
            return false;

        if (inputMatrix.RowLen < 1)
            return false;

        if (inputMatrix.ColumnLen != m_networkPogology.InputNumber)
            return false;

        if (0 == pOutputMatrix)
            return false;

        pOutputMatrix->Reset(inputMatrix.RowLen, m_networkPogology.OutputNumber);


        for (unsigned int row = 0; row < inputMatrix.RowLen; row++)
        {
            inputMatrix.GetRow(row, m_inputVectorForActive);

            for (unsigned int i = 0; i < m_layerList.size(); i++)
            {
                if (0 == i)
                    m_layerList[i]->Active(m_inputVectorForActive, &m_layerOutList[i]);
                else
                    m_layerList[i]->Active(m_layerOutList[i-1], &m_layerOutList[i]);
            }

            for (unsigned int col = 0; col < pOutputMatrix->ColumnLen; col++)
            {
                (*pOutputMatrix)[row][col] = m_layerOutList[m_layerOutList.size()-1][0][col];
            }
        }

        return true;
    }

private:
    /// @brief 初始化BP网络
    bool Init(IN const LBPNetworkPogology& pogology)
    {
        if (pogology.InputNumber < 1 || pogology.OutputNumber < 1 ||
            pogology.HiddenLayerNumber < 1 || pogology.NeuronsOfHiddenLayer < 1)
            return false;

        m_networkPogology = pogology;

        this->CleanUp();

        m_layerOutList.resize(pogology.HiddenLayerNumber + 1);
        m_layerErrorList.resize(pogology.HiddenLayerNumber + 2);
        m_inputVectorForTrain.Reset(1, pogology.InputNumber);
        m_outputVectorForTrain.Reset(1, pogology.OutputNumber);
        m_inputVectorForActive.Reset(1, pogology.InputNumber);

        m_layerErrorList[0].resize(pogology.InputNumber);


        // 创建第一个隐藏层
        CBPNeuronLayer* pFirstHiddenLayer = new CBPNeuronLayer(pogology.InputNumber, pogology.NeuronsOfHiddenLayer);
        m_layerList.push_back(pFirstHiddenLayer);
        m_layerOutList[0].Reset(1, pogology.NeuronsOfHiddenLayer);
        m_layerErrorList[1].resize(pogology.NeuronsOfHiddenLayer);

        // 创建剩余的隐藏层
        for (unsigned int i = 1; i < pogology.HiddenLayerNumber; i++)
        {
            CBPNeuronLayer* pHiddenLayer = new CBPNeuronLayer(pogology.NeuronsOfHiddenLayer, pogology.NeuronsOfHiddenLayer);
            m_layerList.push_back(pHiddenLayer);
            m_layerOutList[i].Reset(1, pogology.NeuronsOfHiddenLayer);
            m_layerErrorList[i + 1].resize(pogology.NeuronsOfHiddenLayer);
        }

        // 创建输出层
        CBPNeuronLayer* pOutputLayer = new CBPNeuronLayer(pogology.NeuronsOfHiddenLayer, pogology.OutputNumber);
        m_layerList.push_back(pOutputLayer);
        m_layerOutList[pogology.HiddenLayerNumber].Reset(1, pogology.OutputNumber);
        m_layerErrorList[pogology.HiddenLayerNumber + 1].resize(pogology.OutputNumber);


        m_bInitDone = true;
        return true;
    }

    /// @brief 清理资源
    void CleanUp()
    {
        m_bInitDone = false;
        for (unsigned int i = 0; i < m_layerList.size(); i++)
        {
            if (0 != m_layerList[i])
            {
                delete m_layerList[i];
                m_layerList[i] = 0;
            }
        }

        m_layerList.clear();
        m_layerOutList.clear();
        m_layerErrorList.clear();
    }

private:
    bool m_bInitDone; ///< 标识是否初始化网络完成
    LBPNetworkPogology m_networkPogology; ///< 网络拓扑结构
    vector<CBPNeuronLayer*> m_layerList; ///< 神经元层列表

    /*
    以下成员变量为Train或Active所用, 为了在多次调用Train或Active函数时提高程序效率
    */
    vector<LNNMatrix> m_layerOutList; ///< 神经元层输出列表
    vector<vector<double>> m_layerErrorList; ///< 神经元层输出误差列表
    LNNMatrix m_inputVectorForTrain; ///< 输入向量Train函数使用
    LNNMatrix m_outputVectorForTrain; ///< 输出向量Train函数使用
    LNNMatrix m_inputVectorForActive; ///< 输入向量Active函数使用
};

LBPNetwork::LBPNetwork(IN const LBPNetworkPogology& pogology)
{
    m_pBPNetwork = 0;
    m_pBPNetwork = new CBPNetwork(pogology);
}

LBPNetwork::~LBPNetwork()
{
    if (0 != m_pBPNetwork)
    {
        delete m_pBPNetwork;
        m_pBPNetwork = 0;
    }
}

bool LBPNetwork::Train(IN const LNNMatrix& inputMatrix, IN const LNNMatrix& outputMatrix, IN float rate)
{
    return m_pBPNetwork->Train(inputMatrix, outputMatrix, rate);
}

bool LBPNetwork::Active(IN const LNNMatrix& inputMatrix, OUT LNNMatrix* pOutputMatrix)
{
    return m_pBPNetwork->Active(inputMatrix, pOutputMatrix);
}