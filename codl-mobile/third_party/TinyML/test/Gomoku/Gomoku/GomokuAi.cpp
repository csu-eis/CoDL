#include "GomokuAi.h"
#include "LNeuralNetwork.h"

#include <cmath>
#include <cstdlib>
#include <vector>
using std::vector;

#ifdef _DEBUG
#define DebugPrint(format, ...) printf(format, __VA_ARGS__)
#else
#define DebugPrint(format, ...) printf(format, __VA_ARGS__)
#endif

#define INPUT_DATA_PLANE    4     // 输入数据平面数 

/// @brief 打印矩阵
static void MatrixDebugPrint(IN const LNNMatrix& dataMatrix)
{
    DebugPrint("Matrix Row: %u  Col: %u\n", dataMatrix.RowLen, dataMatrix.ColumnLen);
    for (unsigned int i = 0; i < dataMatrix.RowLen; i++)
    {
        for (unsigned int j = 0; j < dataMatrix.ColumnLen; j++)
        {
            DebugPrint("%.5f  ", dataMatrix[i][j]);
        }
        DebugPrint("\n");
    }
    DebugPrint("\n");
}

/// @brief 产生随机整数
/// @param[in] min 随机整数的最小值
/// @param[in] max 随机整数的最大值
/// @return 随机整数
inline int RandInt(int min, int max)
{
    return rand() % (max - min + 1) + min;
}

/// @brief 五子棋Ai, 执白子
class CGomokuAi
{
public:
    /// @brief 构造函数
    /// @param[in] rule 游戏规则
    /// @param[in] brainParam 大脑参数
    CGomokuAi(const LGameRule& rule, const LAiBrainParam& brainParam)
    {
        m_gameRule = rule;
        m_inputDataSize = rule.BoardSize * rule.BoardSize * INPUT_DATA_PLANE;
        m_outputDataSize = rule.BoardSize * rule.BoardSize;

        LBPNetworkPogology pogology;
        pogology.InputNumber = m_inputDataSize;
        pogology.OutputNumber = m_outputDataSize;
        pogology.HiddenLayerNumber = brainParam.BrainLayersNum;
        pogology.NeuronsOfHiddenLayer = brainParam.LayerNeuronsNum;
        m_pBrain = new LBPNetwork(pogology);

        m_inputCache.Reset(1, m_inputDataSize);
        m_trainInputCache1.Reset(1, m_inputDataSize);
        m_trainInputCache2.Reset(1, m_inputDataSize);

        m_actionVecCache.reserve(m_outputDataSize);
    }

    /// @brief 构造函数
    /// 从文件中加载五子棋Ai
    /// @param[in] rule 游戏规则
    /// @param[in] pFilePath 文件路径
    CGomokuAi(const LGameRule& rule, IN char* pFilePath)
    {
        m_gameRule = rule;
        m_inputDataSize = rule.BoardSize * rule.BoardSize * INPUT_DATA_PLANE;
        m_outputDataSize = rule.BoardSize * rule.BoardSize;

        m_pBrain = new LBPNetwork(pFilePath);

        m_inputCache.Reset(1, m_inputDataSize);
        m_trainInputCache1.Reset(1, m_inputDataSize);
        m_trainInputCache2.Reset(1, m_inputDataSize);

        m_actionVecCache.reserve(m_outputDataSize);
    }

    /// @brief 析构函数
    ~CGomokuAi()
    {
        if (m_pBrain != nullptr)
        {
            delete m_pBrain;
            m_pBrain = nullptr;
        }
    }

    /// @brief 设置训练参数
    /// @param[in] trainParam 训练参数
    void SetTrainParam(const LAiTrainParam& trainParam)
    {
        m_trainParam = trainParam;
    }

    /// @brief 落子
    /// @param[in] chessBoard 当前棋局
    /// @param[in] preAction 对手最近落子位置
    /// @param[out] pPos 存储落子位置
    void Action(IN const LChessBoard& chessBoard, IN LChessPos& preAction, OUT LChessPos* pPos)
    {
        if (chessBoard.RowLen != m_gameRule.BoardSize ||
            chessBoard.ColumnLen != m_gameRule.BoardSize)
            return;
        if (pPos == nullptr)
            return;

        // 思考后执行
        this->ChessBoard2Input(chessBoard, preAction, &m_inputCache);

        m_pBrain->Active(m_inputCache, &m_outputCache);

        // 找出最大动作值
        double maxAction = GAME_LOSE_SCORE;

        for (unsigned int i = 0; i < m_outputCache.ColumnLen; i++)
        {
            if (m_outputCache[0][i] > maxAction)
            {
                m_actionVecCache.clear();
                m_actionVecCache.push_back(i);
                maxAction = m_outputCache[0][i];
            }
            else if (m_outputCache[0][i] == maxAction)
            {
                m_actionVecCache.push_back(i);
            }
        }

        unsigned int actionIdx = 0;
        if (m_actionVecCache.size() > 1)
        {
            int i = RandInt(0, int(m_actionVecCache.size() - 1));
            actionIdx = i;
        }

        unsigned int action = m_actionVecCache[actionIdx];

        pPos->Row = action / chessBoard.ColumnLen;
        pPos->Col = action % chessBoard.ColumnLen;
    }

    /// @brief 随机落子
    /// @param[in] chessBoard 当前棋局
    /// @param[out] pPos 存储落子位置
    void ActionRandom(IN const LChessBoard& chessBoard, OUT LChessPos* pPos)
    {
        if (chessBoard.RowLen != m_gameRule.BoardSize ||
            chessBoard.ColumnLen != m_gameRule.BoardSize)
            return;
        if (pPos == nullptr)
            return;

        m_actionVecCache.clear();
        for (unsigned int row = 0; row < chessBoard.RowLen; row++)
        {
            for (unsigned int col = 0; col < chessBoard.ColumnLen; col++)
            {
                if (chessBoard[row][col] == SPOT_NONE)
                    m_actionVecCache.push_back(row * chessBoard.ColumnLen + col);
            }
        }

        unsigned int actionIdx = 0;
        if (m_actionVecCache.size() > 1)
        {
            int i = RandInt(0, int(m_actionVecCache.size() - 1));
            actionIdx = i;
        }

        unsigned int action = m_actionVecCache[actionIdx];

        pPos->Row = action / chessBoard.ColumnLen;
        pPos->Col = action % chessBoard.ColumnLen;
    }

    /// @brief 训练
    /// 训练前必须至少设置一次训练参数
    /// @param[in] data 训练数据
    void Train(IN const LTrainData& data)
    {

        this->ChessBoard2Input(data.State, data.PreAction, &m_trainInputCache1);

        m_pBrain->Active(m_trainInputCache1, &m_trainOutputCache1);

        for (unsigned int row = 0; row < data.State.RowLen; row++)
        {
            for (unsigned int col = 0; col < data.State.ColumnLen; col++)
            {
                if (data.State[row][col] != SPOT_NONE)
                {
                    m_trainOutputCache1[0][row* data.State.ColumnLen + col] = GAME_LOSE_SCORE;
                }
                else
                {
                    m_trainOutputCache1[0][row* data.State.ColumnLen + col] = 0.05;
                }

            }
        }

        //MatrixDebugPrint(m_trainOutputCache1);
        /*
        double newActionValue = 0.0;
        unsigned int action = data.Action.Row * data.State.ColumnLen + data.Action.Col;

        if (!data.GameEnd)
        {

            this->ChessBoard2Input(data.NextState, data.Action, &m_trainInputCache2);

            m_pBrain->Active(m_trainInputCache2, &m_trainOutputCache2);

            double currentActionValue = m_trainOutputCache1[0][action];

            double nextActionValueMax = GAME_LOSE_SCORE;
            for (unsigned int col = 0; col < m_trainOutputCache2.ColumnLen; col++)
            {
                if (m_trainOutputCache2[0][col] > nextActionValueMax)
                    nextActionValueMax = m_trainOutputCache2[0][col];
            }

            double difValue = data.Reward + m_trainParam.QLearningGamma * nextActionValueMax - currentActionValue;
            newActionValue = currentActionValue + m_trainParam.QLearningRate * difValue;
            if (newActionValue < GAME_LOSE_SCORE)
                newActionValue = GAME_LOSE_SCORE;
            if (newActionValue > GAME_WIN_SCORE)
                newActionValue = GAME_WIN_SCORE;
        }
        else
        {

            newActionValue = data.Reward;
            
        }

        m_trainOutputCache1[0][action] = newActionValue;
        */
        

        //MatrixDebugPrint(m_trainOutputCache1);

        unsigned int trainCount = m_trainParam.BrainTrainCountMax;
        for (unsigned int i = 0; i < trainCount; i++)
        {
            double rate = m_trainParam.BrainLearningRate;
            m_pBrain->Train(m_trainInputCache1, m_trainOutputCache1, (float)rate);

            m_pBrain->Active(m_trainInputCache1, &m_trainOutputCache2);

            double dif = 0.0;
            for (unsigned int col = 0; col < m_trainOutputCache1.ColumnLen; col++)
            {
                dif += pow(m_trainOutputCache1[0][col] - m_trainOutputCache2[0][col], 2);
            }
            dif = dif / m_trainOutputCache2.ColumnLen;
            if (dif < 0.0001)
            {
                DebugPrint("Train Good Use %u \n", i);
                //MatrixDebugPrint(m_trainOutputCache2);
                //system("pause");
                break;
            }

            if (i == (trainCount - 1))
            {
                DebugPrint("Train Bad\n");
            }
                
        }
    }

    /// @brief 训练
    /// @param[in] datas 训练数据
    void Train(IN const vector<LTrainData>& datas)
    {
        /*
        
        m_trainInputCache1.Reset((unsigned int)datas.size(), CHESSMAN_NUM);

        for (unsigned int i = 0; i < datas.size(); i++)
        {
            const LTrainData& data = datas[i];

            for (unsigned int row = 0; row < data.State.RowLen; row++)
            {
                for (unsigned int col = 0; col < data.State.ColumnLen; col++)
                {
                    unsigned int idx = row * CHESS_BOARD_COLUMN + col;
                    m_trainInputCache1[i][idx] = data.State[row][col];
                }
            }
        }

        m_pBrain->Active(m_trainInputCache1, &m_trainOutputCache1);

        for (unsigned int i = 0; i < datas.size(); i++)
        {
            const LTrainData& data = datas[i];
            double newActionValue = 0.0;
            unsigned int action = data.Action.Row * CHESS_BOARD_COLUMN + data.Action.Col;

            if (!data.GameEnd)
            {
                for (unsigned int row = 0; row < data.NextState.RowLen; row++)
                {
                    for (unsigned int col = 0; col < data.NextState.ColumnLen; col++)
                    {
                        unsigned int idx = row * CHESS_BOARD_COLUMN + col;
                        m_trainInputCache2[0][idx] = data.NextState[row][col];
                    }
                }

                m_pBrain->Active(m_trainInputCache2, &m_trainOutputCache2);

                double currentActionValue = m_trainOutputCache1[i][action];

                double nextActionValueMax = GAME_LOSE_SCORE;
                for (unsigned int col = 0; col < m_trainOutputCache2.ColumnLen; col++)
                {
                    if (m_trainOutputCache2[0][col] > nextActionValueMax)
                        nextActionValueMax = m_trainOutputCache2[0][col];
                }

                double difValue = data.Reward + m_trainParam.QLearningGamma * nextActionValueMax - currentActionValue;
                newActionValue = currentActionValue + m_trainParam.QLearningRate * difValue;
                if (newActionValue < GAME_LOSE_SCORE)
                    newActionValue = GAME_LOSE_SCORE;
                if (newActionValue > GAME_WIN_SCORE)
                    newActionValue = GAME_WIN_SCORE;
            }
            else
            {
                newActionValue = data.Reward;
            }

            m_trainOutputCache1[i][action] = newActionValue;
        }


        unsigned int trainCount = m_trainParam.BrainTrainCount;
        for (unsigned int i = 0; i < trainCount; i++)
        {
            double rate = (m_trainParam.BrainLearningRate * (trainCount - i)) / trainCount;
            m_pBrain->Train(m_trainInputCache1, m_trainOutputCache1, (float)rate);
        }
        */
    }

    /// @brief 将五子棋Ai保存到文件中
    /// @param[in] pFilePath 文件路径
    void Save2File(IN char* pFilePath)
    {
        m_pBrain->Save2File(pFilePath);
    }

private:
    /// @brief 棋盘转换为输入数据
    void ChessBoard2Input(const LChessBoard& chessBoard, const LChessPos& preAction, LNNMatrix* pInput)
    {
        if (pInput == nullptr)
            return;

        // 输入数据包含3个平面
        pInput->Reset(1, m_inputDataSize, 0.0);

        int totalCount = 0;
        for (unsigned int row = 0; row < chessBoard.RowLen; row++)
        {
            for (unsigned int col = 0; col < chessBoard.ColumnLen; col++)
            {
                if (chessBoard[row][col] == SPOT_WHITE ||
                    chessBoard[row][col] == SPOT_BLACK)
                    totalCount++;
            }
        }

        // 当前棋盘上棋子数为偶数, 则下个动作由黑子进行
        bool bBlackAction = ((totalCount % 2) == 0);

        for (unsigned int row = 0; row < chessBoard.RowLen; row++)
        {
            for (unsigned int col = 0; col < chessBoard.ColumnLen; col++)
            {
                double spotType = chessBoard[row][col];

                // 第一个平面表示自己棋子位置
                if (bBlackAction && spotType == SPOT_BLACK)
                    (*pInput)[0][row*m_gameRule.BoardSize + col] = 1.0;
                if ((!bBlackAction) && spotType == SPOT_WHITE)
                    (*pInput)[0][row*m_gameRule.BoardSize + col] = 1.0;

                // 第二个平面表示对手棋子位置
                if (bBlackAction && spotType == SPOT_WHITE)
                    (*pInput)[0][m_outputDataSize + row*m_gameRule.BoardSize + col] = 1.0;
                if (!bBlackAction && spotType == SPOT_BLACK)
                    (*pInput)[0][m_outputDataSize + row*m_gameRule.BoardSize + col] = 1.0;

                // 第三个平面表示对手最近下子位置
                if (totalCount != 0)
                {
                    (*pInput)[0][m_outputDataSize * 2 + preAction.Row*m_gameRule.BoardSize + preAction.Col] = 1.0;
                }
                // 第4个平面表示自己是否是先手
                if (bBlackAction)
                {
                    for (unsigned int i = m_outputDataSize * 3; i < pInput->ColumnLen; i++)
                        (*pInput)[0][i] = 1.0;
                }
            }
        }
    }

    /// @brief 判断当前局面是否由黑子进行动作
    bool IsBlackAction(const LChessBoard& chessBoard)
    {
        int count = 0;
        for (unsigned int row = 0; row < chessBoard.RowLen; row++)
        {
            for (unsigned int col = 0; col < chessBoard.ColumnLen; col++)
            {
                if (chessBoard[row][col] == SPOT_WHITE ||
                    chessBoard[row][col] == SPOT_BLACK)
                    count++;
            }
        }

        // 当前棋盘上棋子数为偶数, 则下个动作由黑子进行
        if (count % 2 == 0)
            return true;
        
        return false;
    }

private:
    LBPNetwork*         m_pBrain;           // Ai的大脑
    LAiTrainParam       m_trainParam;       // 学习参数
    LGameRule           m_gameRule;         // 游戏规则

    unsigned int        m_inputDataSize;    // 输入数据大小
    unsigned int        m_outputDataSize;   // 输出数据大小

    LNNMatrix m_inputCache;                 // 输入缓存, 提高程序执行效率
    LNNMatrix m_outputCache;                // 输出缓存, 提高程序执行效率
    LNNMatrix m_trainInputCache1;           // 输入缓存, 提高程序执行效率
    LNNMatrix m_trainOutputCache1;          // 输出缓存, 提高程序执行效率
    LNNMatrix m_trainInputCache2;           // 输入缓存, 提高程序执行效率
    LNNMatrix m_trainOutputCache2;          // 输出缓存, 提高程序执行效率

    vector<unsigned int> m_actionVecCache;  // 动作缓存, 提高程序执行效率

};

LGomokuAi::LGomokuAi(const LGameRule& rule, const LAiBrainParam& brainParam)
{
    m_pGomokuAi = new CGomokuAi(rule, brainParam);
}

LGomokuAi::LGomokuAi(const LGameRule& rule, IN char* pFilePath)
{
    m_pGomokuAi = new CGomokuAi(rule, pFilePath);
}

LGomokuAi::~LGomokuAi()
{
    if (m_pGomokuAi != nullptr)
    {
        delete m_pGomokuAi;
        m_pGomokuAi = nullptr;
    }
}

void LGomokuAi::SetTrainParam(const LAiTrainParam& trainParam)
{
    m_pGomokuAi->SetTrainParam(trainParam);
}

void LGomokuAi::Action(IN const LChessBoard& chessBoard, IN LChessPos& preAction, OUT LChessPos* pPos)
{
    m_pGomokuAi->Action(chessBoard, preAction, pPos);
}

void LGomokuAi::ActionRandom(IN const LChessBoard& chessBoard, OUT LChessPos* pPos)
{
    m_pGomokuAi->ActionRandom(chessBoard, pPos);
}

void LGomokuAi::Train(IN const LTrainData& data)
{
    m_pGomokuAi->Train(data);
}

void LGomokuAi::Train(IN const vector<LTrainData>& datas)
{
    m_pGomokuAi->Train(datas);
}

void LGomokuAi::Save2File(IN char* pFilePath)
{
    m_pGomokuAi->Save2File(pFilePath);
}


/// @brief 训练数据池
class CTrainDataPool
{
public:
    /// @brief 构造函数
    /// @param[in] maxSize 训练池最大数据个数
    CTrainDataPool(unsigned int maxSize)
    {
        m_dataMaxSize = maxSize;

        m_dataVec.resize(maxSize);
        m_dataUsedVec.resize(maxSize);
        for (unsigned int i = 0; i < maxSize; i++)
        {
            m_dataUsedVec[i] = false;
        }
        m_dataUsedSize = 0;
    }

    /// @brief 析构函数
    ~CTrainDataPool()
    {

    }

    /// @brief 获取数据池数据数量
    unsigned int Size()
    {
        return m_dataUsedSize;
    }

    /// @brief 在数据池中创建新数据
    /// @return 成功创建返回数据地址, 失败返回nullptr, 数据池已满会失败
    LTrainData* NewData()
    {
        if (m_dataUsedSize >= m_dataMaxSize)
            return nullptr;

        // 找到一个可用空间
        for (unsigned int i = 0; i < m_dataMaxSize; i++)
        {
            if (m_dataUsedVec[i] == false)
            {
                m_dataUsedSize += 1;

                m_dataUsedVec[i] = true;
                return &(m_dataVec[i]);
            }
        }
        return nullptr;
    }

    /// @brief 从数据池中随机弹出一个数据
    /// @param[out] pData 存储弹出的数据
    /// @return 成功放入返回true, 失败返回false, 数据池为空会失败
    bool Pop(OUT LTrainData* pData)
    {
        if (m_dataUsedSize < 1)
            return false;
        if (pData == nullptr)
            return false;

        int randCount = RandInt(1, (int)m_dataUsedSize);

        int count = 0;
        for (unsigned int i = 0; i < m_dataMaxSize; i++)
        {
            if (m_dataUsedVec[i] == true)
            {
                count += 1;

                if (randCount == count)
                {
                    (*pData) = m_dataVec[i];
                    m_dataUsedSize -= 1;
                    m_dataUsedVec[i] = false;
                    break;
                }
            }
        }

        return true;
    }

private:
    unsigned int m_dataMaxSize;                 // 数据池最大数据个数
    unsigned int m_dataUsedSize;                // 记录已使用的个数
    vector<LTrainData> m_dataVec;               // 数据池
    vector<bool> m_dataUsedVec;                 // 标记数据池中的对应数据是否被使用
    
};

LTrainDataPool::LTrainDataPool(unsigned int maxSize)
{
    m_pDataPool = new CTrainDataPool(maxSize);
}

LTrainDataPool::~LTrainDataPool()
{
    if (m_pDataPool != nullptr)
    {
        delete m_pDataPool;
        m_pDataPool = nullptr;
    }
}

unsigned int LTrainDataPool::Size()
{
    return m_pDataPool->Size();
}

LTrainData* LTrainDataPool::NewData()
{
    return m_pDataPool->NewData();
}

bool LTrainDataPool::Pop(OUT LTrainData* pData)
{
    return m_pDataPool->Pop(pData);
}