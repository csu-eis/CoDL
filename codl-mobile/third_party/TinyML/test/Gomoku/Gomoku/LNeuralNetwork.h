/// @file LNeuralNetwork.h
/// @brief 该头文件中声明了一些神经网络
/// 
/// Detail:
/// LBPNetwork(反向传播网络): 有监督学习, BP网络的输入数据最好归一化(即输入数据全部调整为0~1之间的值), 
/// BP网络的输出数据为0~1, 训练BP网络所用的目标输出数据必须归一化
/// 
/// @author Jie Liu Email:coderjie@outlook.com
/// @version   
/// @date 2018/06/06


#ifndef _LNEURALNETWORK_H_
#define _LNEURALNETWORK_H_

#include "LMatrix.h"

typedef LMatrix<double> LNNMatrix; // 神经网络矩阵


/// @brief BP网络的拓扑结构
/// BP网络至少有一个输入, 一个输出, 一个隐藏层
/// 隐藏层中至少有一个神经元
struct LBPNetworkPogology
{
    unsigned int InputNumber;           // 输入个数, 要求大于等于1的数
    unsigned int OutputNumber;          // 输出个数, 要求大于等于1的数
    unsigned int HiddenLayerNumber;     // 隐藏层层数, 要求大于等于1的数
    unsigned int NeuronsOfHiddenLayer;  // 单个隐藏层中的神经元个数, 要求大于等于1的数
};

class CBPNetwork;

/// @brief 反向传播网络(BackPropagation)
class LBPNetwork
{
public:
    /// @brief 构造函数
    /// @param[in] pogology BP网络拓扑结构
    explicit LBPNetwork(IN const LBPNetworkPogology& pogology);

    /// @brief 构造函数
    /// 从文件中加载神经网络
    /// @param[in] pFilePath
    explicit LBPNetwork(IN char* pFilePath);

    /// @brief 析构函数
    ~LBPNetwork();

    /// @brief 训练BP网络
    /// 输入数据最好归一化(即输入数据全部调整为0~1之间的值), 目标输出数据必须归一化
    /// @param[in] inputMatrix 输入数据矩阵, 每一行为一个输入, 矩阵的列数必须等于BP网络的输入个数
    /// @param[in] outputMatrix 目标输出数据矩阵, 每一行为一个目标输出, 输出矩阵的行数必须等于数据矩阵的行数, 输出矩阵的列数必须等于BP网络的输出个数
    /// @param[in] rate 学习速率为大于0的数
    /// @return 成功训练返回true, , 失败返回false, 参数有误或者网络未初始化会失败
    bool Train(IN const LNNMatrix& inputMatrix, IN const LNNMatrix& outputMatrix, IN float rate);

    /// @brief 激活神经网络
    /// 
    /// 输入数据最好归一化(即输入数据全部调整为0~1之间的值), 输出数据为0~1之间的值
    /// @param[in] inputMatrix 输入数据矩阵, 每一行为一个输入, 矩阵的列数必须等于BP网络的输入个数
    /// @param[out] pOutputMatrix 存储输出数据矩阵, 每一行为一个输出, 该值不能为0
    /// @return 成功返回true, 失败返回false, 参数有误或者网络未初始化会失败
    bool Active(IN const LNNMatrix& inputMatrix, OUT LNNMatrix* pOutputMatrix);

    /// @brief 将神经网络保存到文件中
    /// @param[in] pFilePath 文件路径
    void Save2File(IN char* pFilePath);

private:
    CBPNetwork* m_pBPNetwork; ///< BP网络的实现对象

private:
    // 禁止拷贝构造函数和赋值操作符
    LBPNetwork(const LBPNetwork&);
    LBPNetwork& operator = (const LBPNetwork&);

};


#endif