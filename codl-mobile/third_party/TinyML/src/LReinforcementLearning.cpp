
#include "LReinforcementLearning.h"

#ifdef _DEBUG
#define DebugPrint(format, ...) printf(format, __VA_ARGS__)
#else
#define DebugPrint(format, ...)
#endif


/// @brief 产生随机整数
/// @param[in] min 随机整数的最小值(包含该值)
/// @param[in] max 随机整数的最大值(包含该值)
/// @return 随机整数
static int RandInt(int min, int max)
{
    if (min > max)
    {
        int t = max; max = min; min = t;
    }

    return rand() % (max - min + 1) + min;
}

/// @brief 大值比较函数
template<typename T>
inline T Max(T a, T b)
{
    if (a > b)
        return a;
    else
        return b;
}


/// @brief 动态规划算法
class CMDPDynamicProgram
{
public:
    /// @brief 构造函数
    /// @param[in] stateSet 状态集合, 不能为空集合
    /// @param[in] actionSet 动作集合, 不能为空集合
    CMDPDynamicProgram(IN const LStateSet& stateSet, IN const LActionSet& actionSet)
        : m_stateSet(stateSet), m_actionSet(actionSet)
    {

    }

    /// @brief 析构函数
    ~CMDPDynamicProgram()
    {

    }

    /// @brief 策略迭代算法
    /// @param[in] r 立即回报函数
    /// @param[in] p 状态转移概率函数
    /// @param[in] theta 阈值(终止条件, 状态值变化的最小量)
    /// @return 成功返回true, 失败返回false(参数错误的情况下会返回失败)
    bool PolicyIteration(IN RewardFunc r, IN StateProbFunc p, IN double theta, IN double gamma)
    {
        // 检查参数
        if (r == nullptr || p == nullptr)
            return false;
        if (theta <= 0)
            return false;
        if (gamma < 0 || gamma > 1)
            return false;
        if (m_stateSet.size() < 1)
            return false;
        if (m_actionSet.size() < 1)
            return false;

        DebugPrint("Policy Iteration: \n");

        m_stateValueTable.clear();
        m_policyTable.clear();

        for (auto iter = m_stateSet.begin(); iter != m_stateSet.end(); iter++)
        {
            // 将状态值表赋值为0
            m_stateValueTable[*iter] = 0;

            // 将策略表随机赋值
            m_policyTable[*iter] = RandomAction(m_actionSet);
        }

        while (true)
        {
            DebugPrint("Policy Evaluation: \n");
            // 策略计算
            while (true)
            {
                double delta = 0.0;
                // 针对每一个状态修改状态值
                for (auto iterState = m_stateSet.begin(); iterState != m_stateSet.end(); iterState++)
                {
                    unsigned int state = *iterState;
                    unsigned int action = m_policyTable[state];

                    // 计算立即回报
                    double reward = 0.0;
                    r(state, action, &reward);

                    // 计算未来回报
                    double futureReward = FutureReward(state, action, p);

                    // 更新状态值
                    double oldValue = m_stateValueTable[state];
                    m_stateValueTable[state] = reward + gamma * futureReward;

                    double dif = m_stateValueTable[state] - oldValue;
                    if (dif < 0.0)
                        dif *= -1.0;

                    delta = Max(delta, dif);
                }

                DebugPrint("Delta: %f\n", delta);

                // 状态值收敛到符合要求
                if (delta < theta)
                    break;

            }

            DebugPrint("Policy Improvement: \n");
            // 策略提高
            bool policyUpdate = false;
            for (auto iterState = m_stateSet.begin(); iterState != m_stateSet.end(); iterState++)
            {
                unsigned int state = *iterState;
                unsigned int oldAction = m_policyTable[state];

                // 找出最好动作
                unsigned int bestAction;
                double maxActionValue;

                bool bFirst = true;
                for (auto iterAction = m_actionSet.begin(); iterAction != m_actionSet.end(); iterAction++)
                {
                    unsigned int action = *iterAction;

                    // 计算立即回报
                    double reward = 0.0;
                    r(state, action, &reward);

                    // 计算未来回报
                    double futureReward = FutureReward(state, action, p);

                    double actionValue = reward + gamma * futureReward;

                    if (bFirst)
                    {
                        bestAction = action;
                        maxActionValue = actionValue;
                        bFirst = false;
                    }
                    else
                    {
                        if (actionValue > maxActionValue)
                        {
                            maxActionValue = actionValue;
                            bestAction = action;
                        }
                    }

                }

                // 更新策略表
                m_policyTable[state] = bestAction;
                if (bestAction != oldAction)
                    policyUpdate = true;
            }

            // 找到最佳策略
            if (!policyUpdate)
                break;

        }

        return true;

    }

    /// @brief 值迭代算法
    /// @param[in] r 立即回报函数
    /// @param[in] p 状态转移概率函数
    /// @param[in] theta 阈值(终止条件, 状态值变化的最小量)
    /// @return 成功返回true, 失败返回false(参数错误的情况下会返回失败)
    bool ValueIteration(IN RewardFunc r, IN StateProbFunc p, IN double theta, IN double gamma)
    {
        // 检查参数
        if (r == nullptr || p == nullptr)
            return false;
        if (theta <= 0)
            return false;
        if (gamma < 0 || gamma > 1)
            return false;
        if (m_stateSet.size() < 1)
            return false;
        if (m_actionSet.size() < 1)
            return false;

        DebugPrint("Value Iteration: \n");

        m_stateValueTable.clear();
        m_policyTable.clear();

        for (auto iter = m_stateSet.begin(); iter != m_stateSet.end(); iter++)
        {
            // 将状态值表赋值为0
            m_stateValueTable[*iter] = 0;
        }

        while (true)
        {
            double delta = 0.0;
            // 针对每一个状态修改状态值
            for (auto iterState = m_stateSet.begin(); iterState != m_stateSet.end(); iterState++)
            {
                unsigned int state = *iterState;

                double maxStateValue;
                unsigned int bestAction;

                bool bFirst = true;
                for (auto iterAction = m_actionSet.begin(); iterAction != m_actionSet.end(); iterAction++)
                {
                    unsigned int action = *iterAction;

                    // 计算立即回报
                    double reward = 0.0;
                    r(state, action, &reward);

                    // 计算未来回报
                    double futureReward = FutureReward(state, action, p);

                    double actionValue = reward + gamma * futureReward;

                    if (bFirst)
                    {
                        maxStateValue = actionValue;
                        bestAction = action;

                        bFirst = false;
                    }
                    else
                    {
                        if (actionValue > maxStateValue)
                        {
                            maxStateValue = actionValue;
                            bestAction = action;
                        }
                    }

                }

                // 更新状态值
                double oldValue = m_stateValueTable[state];
                m_stateValueTable[state] = maxStateValue;
                m_policyTable[state] = bestAction;


                double dif = m_stateValueTable[state] - oldValue;
                if (dif < 0.0)
                    dif *= -1.0;

                delta = Max(delta, dif);
            }

            DebugPrint("Delta: %f\n", delta);

            // 状态值收敛到符合要求
            if (delta < theta)
                break;

        }

        return true;

    }

    /// @brief 获取状态值表
    const LStateValueTable& GetStateValueTable()
    {
        return m_stateValueTable;
    }

    /// @brief 获取策略值表
    const LPolicyTable& GetPolicyTable()
    {
        return m_policyTable;
    }

private:
    /// @brief 获得一个随机的动作
    /// @param[in] actionSet 动作集合
    /// @return 成功一个随机的动作
    unsigned int RandomAction(IN const LActionSet& actionSet)
    {
        int size = (int)actionSet.size();
        int count = RandInt(1, size);

        auto iter = actionSet.begin();
        for (int i = 1; i < count; i++)
        {
            iter++;
        }

        return *iter;
    }

    /// @brief 获取未来回报值
    /// @param[in] s 状态值
    /// @param[in] a 动作值
    /// @param[in] p 状态转移概率函数
    /// @return 未来回报值
    double FutureReward(IN unsigned int s, IN unsigned int a, IN StateProbFunc p)
    {
        double sum = 0.0;
        LStateProbTable probTable;
        p(s, a, &probTable);
        for (auto iterProb = probTable.begin(); iterProb != probTable.end(); iterProb++)
        {
            unsigned int newSate = iterProb->first;
            double newStateProb = iterProb->second;
            sum += (newStateProb * m_stateValueTable[newSate]);
        }

        return sum;
    }

private:
    const LStateSet& m_stateSet;            // 状态集合
    const LActionSet& m_actionSet;          // 动作集合
    LPolicyTable m_policyTable;             // 策略表
    LStateValueTable m_stateValueTable;     // 状态值表
};

LPolicyIteration::LPolicyIteration(IN const LStateSet& stateSet, IN const LActionSet& actionSet)
{
    m_pMDPDynamicProgram = new CMDPDynamicProgram(stateSet, actionSet);
}

LPolicyIteration::~LPolicyIteration()
{
    if (m_pMDPDynamicProgram != nullptr)
    {
        delete m_pMDPDynamicProgram;
        m_pMDPDynamicProgram = nullptr;
    }
}

bool LPolicyIteration::TrainModel(IN RewardFunc r, IN StateProbFunc p, IN double theta, IN double gamma)
{
    return m_pMDPDynamicProgram->PolicyIteration(r, p, theta, gamma);
}

const LStateValueTable& LPolicyIteration::GetStateValueTable()
{
    return m_pMDPDynamicProgram->GetStateValueTable();
}

const LPolicyTable& LPolicyIteration::GetPolicyTable()
{
    return m_pMDPDynamicProgram->GetPolicyTable();
}

LValueIteration::LValueIteration(IN const LStateSet& stateSet, IN const LActionSet& actionSet)
{
    m_pMDPDynamicProgram = new CMDPDynamicProgram(stateSet, actionSet);
}

LValueIteration::~LValueIteration()
{
    if (m_pMDPDynamicProgram != nullptr)
    {
        delete m_pMDPDynamicProgram;
        m_pMDPDynamicProgram = nullptr;
    }
}

bool LValueIteration::TrainModel(IN RewardFunc r, IN StateProbFunc p, IN double theta, IN double gamma)
{
    return m_pMDPDynamicProgram->ValueIteration(r, p, theta, gamma);
}

const LStateValueTable& LValueIteration::GetStateValueTable()
{
    return m_pMDPDynamicProgram->GetStateValueTable();
}

const LPolicyTable& LValueIteration::GetPolicyTable()
{
    return m_pMDPDynamicProgram->GetPolicyTable();
}


