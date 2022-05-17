

#include "../../../Src/LReinforcementLearning.h"

/// @brief 立即回报函数
/// @param[in] s 状态值
/// @param[in] a 动作值
/// @param[out] pReward 存储s状态下动作a的立即回报值
/// @return 成功返回true, 失败返回false
bool Reward(IN int s, IN int a, OUT double* pReward)
{
    if (pReward == nullptr)
        return false;

    (*pReward) = 0.0;

    if (s == 0 && a == 3)
        (*pReward) = -10.0;
    if (s == 1 && a == 0)
        (*pReward) = -10.0;
    if (s == 2 && a == 2)
        (*pReward) = -10.0;
    if (s == 4 && a == 0)
        (*pReward) = -10.0;
    if (s == 5 && a == 1)
        (*pReward) = 10.0;
    if (s == 7 && a == 3)
        (*pReward) = 10.0;
    if (s == 8 && a == 1)
        (*pReward) = 10.0;
    if (s == 8 && a == 3)
        (*pReward) = 10.0;

    return true;
}

/// @brief 状态转移概率函数
/// @param[in] s 状态值
/// @param[in] a 动作值
/// @param[out] pStateProbVec 存储s状态下进行动作a后转移到各个状态的概率
/// @return 成功返回true, 失败返回false
bool StateProb(IN int s, IN int a, OUT LStateProbTable* pStateProbTable)
{
    static unsigned int SStateActionTable[9][4] = 
    {
        {0, 3, 0, 1},
        {1, 4, 0, 2},
        {2, 5, 1, 2},
        {0, 6, 3, 4},
        {1, 7, 3, 5},
        {2, 8, 4, 5},
        {3, 6, 6, 7},
        {4, 7, 6, 8},
        {5, 8, 7, 8}
    };

    if (pStateProbTable == nullptr)
        return false;

    unsigned int state = SStateActionTable[s][a];

    (*pStateProbTable)[state] = 1.0f;

    return true;
}


int main()
{
    LStateSet stateSet;
    for (unsigned int i = 0; i < 9; i++)
    {
        stateSet.insert(i);
    }

    LActionSet actionSet;
    for (unsigned int i = 0; i < 4; i++)
    {
        actionSet.insert(i);
    }

    LValueIteration iteration(stateSet, actionSet);

    iteration.TrainModel(Reward, StateProb, 0.0001, 0.9);

    LPolicyTable policyTable = iteration.GetPolicyTable();
    LStateValueTable stateValueTable = iteration.GetStateValueTable();


    system("pause");
}