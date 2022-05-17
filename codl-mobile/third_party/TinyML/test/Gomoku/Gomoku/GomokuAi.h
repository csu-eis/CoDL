#pragma once

#include "LMatrix.h"

#include <vector>
using std::vector;

#define SPOT_WHITE           1.0        // 白子
#define SPOT_NONE            0.0        // 无子
#define SPOT_BLACK          -1.0        // 黑子

#define GAME_WIN_SCORE       1.0        // 赢棋得分
#define GAME_DRAWN_SCORE     0.05       // 和棋得分
#define GAME_LOSE_SCORE      0.0        // 输棋得分

typedef LMatrix<double> LChessBoard;    // 棋盘

/// @brief 棋子位置
struct LChessPos 
{
    unsigned int Row;       // 行数 
    unsigned int Col;       // 列数
};

/// @brief 训练数据
struct LTrainData
{
    LChessPos   PreAction;  // 对手最近动作
    LChessBoard State;      // 当前状态
    LChessPos   Action;     // 执行动作(落子位置)
    bool        GameEnd;    // 标记游戏是否结束
    double      Reward;     // 立即回报值(得分值), 1.0(赢), 0.05(和棋), 0.0(输棋)
    LChessBoard NextState;  // 下个状态
};

/// @brief 游戏规则
struct LGameRule
{
    unsigned int ContinuNum;                // 连子数
    unsigned int BoardSize;                 // 棋盘大小, 例: 8, 则表示是8*8的棋盘
};

/// @brief Ai大脑参数
struct LAiBrainParam
{
    unsigned int    BrainLayersNum;         // 大脑层数, 范围: [1, +)
    unsigned int    LayerNeuronsNum;        // 每层神经元数, 范围: 大于等于1
};

/// @brief Ai训练参数
struct LAiTrainParam
{
    double          QLearningRate;          // QLearning学习速度, 范围: (0, 1)
    double          QLearningGamma;         // QLearning折合因子, 范围: [0, 1]
    double          BrainLearningRate;      // 大脑学习速度, 范围: (0.0, +)
    unsigned int    BrainTrainCountMax;     // 大脑最大训练次数, 范围: [1, +)
};

class CGomokuAi;

/// @brief 五子棋Ai
/// 黑子先手
class LGomokuAi
{
public:
    /// @brief 构造函数
    /// @param[in] rule 游戏规则
    /// @param[in] brainParam 大脑参数
    LGomokuAi(const LGameRule& rule, const LAiBrainParam& brainParam);

    /// @brief 构造函数
    /// 从文件中加载五子棋Ai
    /// @param[in] rule 游戏规则
    /// @param[in] pFilePath 文件路径
    LGomokuAi(const LGameRule& rule, IN char* pFilePath);

    /// @brief 析构函数
    ~LGomokuAi();

    /// @brief 设置训练参数
    /// @param[in] trainParam 训练参数
    void SetTrainParam(const LAiTrainParam& trainParam);

    /// @brief 落子
    /// @param[in] chessBoard 当前棋局
    /// @param[in] preAction 对手最近落子位置, 如果目前为空棋局, 则该值没用
    /// @param[out] pPos 存储落子位置
    void Action(IN const LChessBoard& chessBoard, IN LChessPos& preAction, OUT LChessPos* pPos);

    /// @brief 随机落子
    /// @param[in] chessBoard 当前棋局
    /// @param[out] pPos 存储落子位置
    void ActionRandom(IN const LChessBoard& chessBoard, OUT LChessPos* pPos);

    /// @brief 训练
    /// 训练前必须至少设置一次训练参数
    /// @param[in] data 训练数据
    void Train(IN const LTrainData& data);

    /// @brief 训练
    /// 训练前必须至少设置一次训练参数
    /// @param[in] datas 训练数据
    void Train(IN const vector<LTrainData>& datas);

    /// @brief 将五子棋Ai保存到文件中
    /// @param[in] pFilePath 文件路径
    void Save2File(IN char* pFilePath);

private:
    CGomokuAi* m_pGomokuAi;         // 五子棋Ai实现对象
};

class CTrainDataPool;

/// @brief 训练数据池
class LTrainDataPool
{
public:
    /// @brief 构造函数
    /// @param[in] maxSize 训练池最大数据个数
    LTrainDataPool(unsigned int maxSize);

    /// @brief 析构函数
    ~LTrainDataPool();

    /// @brief 获取数据池数据数量
    unsigned int Size();

    /// @brief 在数据池中创建新数据
    /// @return 成功创建返回数据地址, 失败返回nullptr, 数据池已满会失败
    LTrainData* NewData();

    /// @brief 从数据池中随机弹出一个数据
    /// @param[out] pData 存储弹出的数据
    /// @return 成功放入返回true, 失败返回false, 数据池为空会失败
    bool Pop(OUT LTrainData* pData);

private:
    CTrainDataPool* m_pDataPool;            // 数据池实现对象
};

