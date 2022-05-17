
/// @file LReinforcementLearning.h
/// @brief 本文件中声明了一些增强学习算法
/// 策略迭代, 值迭代
/// Detail:
/// 策略迭代:
/// 
/// 值迭代:
/// 
/// @author Jie Liu Email:coderjie@outlook.com
/// @version   
/// @date 2018/08/17

#ifndef _LREINFORCEMENT_LEARNING_H_
#define _LREINFORCEMENT_LEARNING_H_


#include <vector>
#include <set>
#include <map>
using std::vector;
using std::set;
using std::map;


#ifndef IN
#define IN
#endif

#ifndef INOUT
#define INOUT
#endif

#ifndef OUT
#define OUT
#endif


typedef set<unsigned int>               LStateSet;          // 状态集合
typedef set<unsigned int>               LActionSet;         // 动作集合
typedef map<unsigned int, double>       LStateProbTable;    // 状态概率表, 键为状态, 值为概率
typedef map<unsigned int, double>       LStateValueTable;   // 状态回报表, 键为状态, 值为状态的回报 
typedef map<unsigned int, unsigned int> LPolicyTable;       // 策略表, 键为状态, 值为动作


/// @brief 立即回报函数
/// @param[in] s 状态值
/// @param[in] a 动作值
/// @param[out] pReward 存储s状态下动作a的立即回报值
/// @return 成功返回true, 失败返回false
typedef bool (*RewardFunc)(IN int s, IN int a, OUT double* pReward);

/// @brief 状态转移概率函数
/// @param[in] s 状态值
/// @param[in] a 动作值
/// @param[out] pStateProbVec 存储s状态下进行动作a后转移到各个状态的概率
/// @return 成功返回true, 失败返回false
typedef bool(*StateProbFunc)(IN int s, IN int a, OUT LStateProbTable* pStateProbTable);

class CMDPDynamicProgram;

/// @brief 策略迭代
class LPolicyIteration
{
public:
    /// @brief 构造函数
    /// @param[in] stateSet 状态集合, 不能为空集合
    /// @param[in] actionSet 动作集合, 不能为空集合
    LPolicyIteration(IN const LStateSet& stateSet, IN const LActionSet& actionSet);

    /// @brief 析构函数
    ~LPolicyIteration();

    /// @brief 训练模型
    /// @param[in] r 立即回报函数
    /// @param[in] p 状态转移概率函数
    /// @param[in] theta 阈值(终止条件, 状态值变化的最小量, 正值)
    /// @param[in] gamma 折合因子(未来回报的重要程度), [0-1]
    /// @return 成功返回true, 失败返回false(参数错误的情况下会返回失败)
    bool TrainModel(IN RewardFunc r, IN StateProbFunc p, IN double theta, IN double gamma);

    /// @brief 获取状态值表
    const LStateValueTable& GetStateValueTable();

    /// @brief 获取策略值表
    const LPolicyTable& GetPolicyTable();

private:
    CMDPDynamicProgram* m_pMDPDynamicProgram;       // 动态规划算法对象
};

/// @brief 值迭代
class LValueIteration
{
public:
    /// @brief 构造函数
    /// @param[in] stateSet 状态集合, 不能为空集合
    /// @param[in] actionSet 动作集合, 不能为空集合
    LValueIteration(IN const LStateSet& stateSet, IN const LActionSet& actionSet);

    /// @brief 析构函数
    ~LValueIteration();

    /// @brief 训练模型
    /// @param[in] r 立即回报函数
    /// @param[in] p 状态转移概率函数
    /// @param[in] theta 阈值(终止条件, 状态值变化的最小量, 正值)
    /// @param[in] gamma 折合因子(未来回报的重要程度), [0-1]
    /// @return 成功返回true, 失败返回false(参数错误的情况下会返回失败)
    bool TrainModel(IN RewardFunc r, IN StateProbFunc p, IN double theta, IN double gamma);

    /// @brief 获取状态值表
    const LStateValueTable& GetStateValueTable();

    /// @brief 获取策略值表
    const LPolicyTable& GetPolicyTable();

private:
    CMDPDynamicProgram* m_pMDPDynamicProgram;       // 动态规划算法对象
};

#endif