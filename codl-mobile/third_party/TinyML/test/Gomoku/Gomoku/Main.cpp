
#include "GomokuAi.h"

#define TRAIN_POOL_SIZE     500        // 训练池大小
#define TRAIN_DATAT_NUM     50         // 每次训练数 
#define SELF_GAME_NUM       10000      // 自我对弈数


#ifdef _DEBUG
#define DebugPrint(format, ...) printf(format, __VA_ARGS__)
#else
#define DebugPrint(format, ...) printf(format, __VA_ARGS__)
#endif

/// @brief 产生随机小数, 范围0~1
/// @return 随机小数
static float RandFloat()
{
    return (rand()) / (RAND_MAX + 1.0f);
}

/// @brief 打印棋盘
static void ChessBoardDebugPrint(IN const LChessBoard& dataMatrix)
{
    for (unsigned int i = 0; i < dataMatrix.RowLen; i++)
    {
        for (unsigned int j = 0; j < dataMatrix.ColumnLen; j++)
        {
            if (dataMatrix[i][j] == SPOT_WHITE)
                DebugPrint("0 ");
            if (dataMatrix[i][j] == SPOT_BLACK)
                DebugPrint("* ");
            if (dataMatrix[i][j] == SPOT_NONE)
                DebugPrint("  ");
        }
        DebugPrint("\n");
    }
    DebugPrint("\n");
}


/// @brief 棋盘状态
enum CHESS_BOARD_STATE
{
    STATE_BLACK_WIN =       0,          // 黑子胜
    STATE_WHITE_WIN =       1,          // 白子胜
    STATE_NONE_WIN =        2,          // 和棋
    STATE_BLACK_WHITE =     3,          // 未结束
    STATE_ERROR_LOCATION =  4           // 错误位置
};


/// @brief 检查棋盘
/// @param[in] chessBoard 当前棋盘
/// @param[in] chessPos 下子位置
/// @param[in] chessType 棋子类型
/// @return 棋盘执行下子后的状态
CHESS_BOARD_STATE CheckChessBoard(const LChessBoard& chessBoard, LChessPos& chessPos, double chessType)
{
    // 检查是否下在已有棋子的位置
    unsigned int targetRow = chessPos.Row;
    unsigned int targetCol = chessPos.Col;
    if (chessBoard[targetRow][targetCol] != SPOT_NONE)
    {
        return STATE_ERROR_LOCATION;
    }

    // 检查横向是否连成5子
    unsigned int spotCount = 1;
    for (int col = int(targetCol-1); col >= 0; col--)
    {
        if (chessBoard[targetRow][col] == chessType)
            spotCount++;
        else
            break;
    }
    for (unsigned int col = targetCol + 1; col < chessBoard.ColumnLen; col++)
    {
        if (chessBoard[targetRow][col] == chessType)
            spotCount++;
        else
            break;
    }
    if (spotCount >= 4)
    {
        if (chessType == SPOT_WHITE)
            return STATE_WHITE_WIN;
        if (chessType == SPOT_BLACK)
            return STATE_BLACK_WIN;
    }

    // 检查纵向是否连成5子
    spotCount = 1;
    for (int row = int(targetRow - 1); row >= 0; row--)
    {
        if (chessBoard[row][targetCol] == chessType)
            spotCount++;
        else
            break;
    }
    for (unsigned int row = targetRow + 1; row < chessBoard.RowLen; row++)
    {
        if (chessBoard[row][targetCol] == chessType)
            spotCount++;
        else
            break;
    }
    if (spotCount >= 4)
    {
        if (chessType == SPOT_WHITE)
            return STATE_WHITE_WIN;
        if (chessType == SPOT_BLACK)
            return STATE_BLACK_WIN;
    }

    // 检查斜线是否连成5子
    spotCount = 1;
    for (int row = int(targetRow - 1), col = int(targetCol - 1); 
        row >= 0 && col >= 0; 
        row--, col--)
    {
        if (chessBoard[row][col] == chessType)
            spotCount++;
        else
            break;
    }
    for (unsigned int row = targetRow + 1, col = targetCol + 1; 
        row < chessBoard.RowLen && col < chessBoard.ColumnLen; 
        row++, col++)
    {

        if (chessBoard[row][col] == chessType)
            spotCount++;
        else
            break;
    }
    if (spotCount >= 4)
    {
        if (chessType == SPOT_WHITE)
            return STATE_WHITE_WIN;
        if (chessType == SPOT_BLACK)
            return STATE_BLACK_WIN;
    }

    // 检查反斜线是否连成5子
    spotCount = 1;
    for (int row = int(targetRow - 1),  col = int(targetCol + 1);
        row >= 0 && col < int(chessBoard.ColumnLen);
        row--, col++)
    {
        if (chessBoard[row][col] == chessType)
            spotCount++;
        else
            break;
    }
    for (int row = int(targetRow + 1), col = int(targetCol - 1);
        row < int(chessBoard.RowLen) && col >= 0;
        row++, col--)
    {
        if (chessBoard[row][col] == chessType)
            spotCount++;
        else
            break;
    }
    if (spotCount >= 4)
    {
        if (chessType == SPOT_WHITE)
            return STATE_WHITE_WIN;
        if (chessType == SPOT_BLACK)
            return STATE_BLACK_WIN;
    }


    // 检查是否和棋
    bool bDrawn = true;
    for (unsigned int row = 0; row < chessBoard.RowLen; row++)
    {
        for (unsigned int col = 0; col < chessBoard.ColumnLen; col++)
        {
            if (row == targetRow &&
                col == targetCol)
                continue;

            if (chessBoard[row][col] == SPOT_NONE)
            {
                bDrawn = false;
                break;
            }
        }
        if (!bDrawn)
            break;
    }
    if (bDrawn)
        return STATE_NONE_WIN;


    return STATE_BLACK_WHITE;

}

/// @brief 进行训练
void Train()
{
    LGameRule gameRule;
    gameRule.BoardSize = 8;
    gameRule.ContinuNum = 4;

    LAiBrainParam aiParam;
    aiParam.BrainLayersNum = 4;
    aiParam.LayerNeuronsNum = 512;

    LAiTrainParam trainParam;
    trainParam.QLearningRate = 0.5;
    trainParam.QLearningGamma = 0.9;
    trainParam.BrainTrainCountMax = 1000;
    trainParam.BrainLearningRate = 0.1;

    LGomokuAi ai(gameRule, aiParam);
    ai.SetTrainParam(trainParam);

    LChessBoard chessBoard;                         // 棋盘
    LTrainDataPool dataPool(TRAIN_POOL_SIZE + 8);   // 训练池

    // 自我对弈, 进行训练
    int gameCount = SELF_GAME_NUM;
    for (int i = 0; i < gameCount; i++)
    {
        DebugPrint("SelfGame%i\n", i);

        // 每进行1000盘自我对弈就保存一次Ai到文件中
        if ((i + 1) % 1000 == 0)
        {
            char fileBuffer[64] = { 0 };
            sprintf_s(fileBuffer, ".\\%d-Train.ai", (i + 1));
            ai.Save2File(fileBuffer);
        }

        // 重置棋盘
        chessBoard.Reset(gameRule.BoardSize, gameRule.BoardSize, SPOT_NONE);

        LChessPos prePos;                   // 前一动作
        LChessPos currentPos;               // 当前动作
        LTrainData dataTmp1;                // 临时数据
        LTrainData dataTmp2;                // 临时数据

        // 计算随机率
        double e = (gameCount - i) / (double)(gameCount * 4 /3);
        e = 0.0;

        while (true)
        {
            // Ai以黑子身份下棋
            if (RandFloat() < e)
                ai.ActionRandom(chessBoard, &currentPos);
            else
                ai.Action(chessBoard, prePos, &currentPos);

            CHESS_BOARD_STATE state = CheckChessBoard(chessBoard, currentPos, SPOT_BLACK);

            // 游戏结束
            if (state != STATE_BLACK_WHITE)
            {
                DebugPrint("End Action: %2u %2u\n", currentPos.Row, currentPos.Col);
                LTrainData* pDataEnd = dataPool.NewData();
                pDataEnd->PreAction = prePos;
                pDataEnd->State = chessBoard;
                pDataEnd->Action = currentPos;
                pDataEnd->GameEnd = true;
                if (state == STATE_ERROR_LOCATION)
                {
                    pDataEnd->Reward = GAME_LOSE_SCORE;
                    DebugPrint("EndGame: Error Location\n");
                }
                if (state == STATE_BLACK_WIN)
                {
                    pDataEnd->Reward = GAME_WIN_SCORE;
                    DebugPrint("EndGame: Black Win\n");
                }
                    
                if (state == STATE_NONE_WIN)
                {
                    pDataEnd->Reward = GAME_DRAWN_SCORE;
                    DebugPrint("EndGame: None Win\n");
                }
                    
                ChessBoardDebugPrint(chessBoard);
                break;
            }

            dataTmp1.PreAction = prePos;
            dataTmp1.State = chessBoard;
            dataTmp1.Action = currentPos;

            chessBoard[currentPos.Row][currentPos.Col] = SPOT_BLACK;
            prePos = currentPos;

            // Ai以白子身份下棋
            if (RandFloat() < e)
                ai.ActionRandom(chessBoard, &currentPos);
            else
                ai.Action(chessBoard, prePos, &currentPos);
            state = CheckChessBoard(chessBoard, currentPos, SPOT_WHITE);

            // 游戏结束
            if (state != STATE_BLACK_WHITE)
            {
                DebugPrint("End Action: %2u %2u\n", currentPos.Row, currentPos.Col);
                dataTmp1.GameEnd = true;

                LTrainData* pDataEnd = dataPool.NewData();
                pDataEnd->PreAction = prePos;
                pDataEnd->State = chessBoard;
                pDataEnd->Action = currentPos;
                pDataEnd->GameEnd = true;
                if (state == STATE_ERROR_LOCATION)
                {
                    pDataEnd->Reward = GAME_LOSE_SCORE;
                    DebugPrint("EndGame: Error Location\n");
                }
                if (state == STATE_WHITE_WIN)
                {
                    pDataEnd->Reward = GAME_WIN_SCORE;
                    dataTmp1.Reward = GAME_LOSE_SCORE;
                    LTrainData* pDataContinue = dataPool.NewData();
                    (*pDataContinue) = dataTmp1;
                    DebugPrint("EndGame: White Win\n");
                }
                if (state == STATE_NONE_WIN)
                { 
                    pDataEnd->Reward = GAME_DRAWN_SCORE;
                    dataTmp1.Reward = GAME_DRAWN_SCORE;
                    LTrainData* pDataContinue = dataPool.NewData();
                    (*pDataContinue) = dataTmp1;
                    DebugPrint("EndGame: None Win\n");
                }

                ChessBoardDebugPrint(chessBoard);
                break;
            }

            chessBoard[currentPos.Row][currentPos.Col] = SPOT_WHITE;
            prePos = currentPos;

            dataTmp1.GameEnd = false;
            dataTmp1.Reward = GAME_DRAWN_SCORE;
            dataTmp1.NextState = chessBoard;

            LTrainData* pDataContinue = dataPool.NewData();
            (*pDataContinue) = dataTmp1;

            if (dataPool.Size() >= TRAIN_POOL_SIZE)
            {
                DebugPrint("Training...\n");
                LTrainData data;
                for (unsigned int i = 0; i < TRAIN_DATAT_NUM; i++)
                {
                    dataPool.Pop(&data);
                    ai.Train(data);
                }
                
            }

        }

        if (dataPool.Size() >= TRAIN_POOL_SIZE)
        {
            DebugPrint("Training...\n");
            LTrainData data;
            for (unsigned int i = 0; i < TRAIN_DATAT_NUM; i++)
            {
                dataPool.Pop(&data);
                ai.Train(data);
            }
        }


    }

    DebugPrint("Training completed\n");
}

/// @brief 测试
void Test(char* pFilePath)
{

    LGameRule gameRule;
    gameRule.BoardSize = 8;
    gameRule.ContinuNum = 4;

    LAiBrainParam aiParam;
    aiParam.BrainLayersNum = 4;
    aiParam.LayerNeuronsNum = 512;

//     LAiTrainParam trainParam;
//     trainParam.QLearningRate = 0.5;
//     trainParam.QLearningGamma = 0.9;
//     trainParam.BrainTrainCountMax = 10000;
//     trainParam.BrainLearningRate = 0.1;

    LGomokuAi ai(gameRule, pFilePath);

    LChessBoard chessBoard;            // 棋盘

    while (true)
    {
        // 重置棋盘
        chessBoard.Reset(gameRule.BoardSize, gameRule.BoardSize, SPOT_NONE);

        LChessPos pos;
        LChessPos prePos;

        while (true)
        {

            CHESS_BOARD_STATE state;

            // Ai以黑子身份下棋即在反转棋盘上以白子下棋
            ai.Action(chessBoard, prePos, &pos);
            DebugPrint("Black: %2u %2u \n", pos.Row, pos.Col);

            state = CheckChessBoard(chessBoard, pos, SPOT_BLACK);
            // 游戏结束
            if (state != STATE_BLACK_WHITE)
            {
                if (state == STATE_ERROR_LOCATION)
                    DebugPrint("End Error Location\n");
                if (state == STATE_BLACK_WIN)
                    DebugPrint("End Blacke Win\n");
                if (state == STATE_NONE_WIN)
                    DebugPrint("End Drawn\n");

                ChessBoardDebugPrint(chessBoard);
                system("pause");
                break;
            }

            chessBoard[pos.Row][pos.Col] = SPOT_BLACK;
            prePos = pos;

            // Ai以白子身份下棋
            ai.Action(chessBoard, prePos, &pos);
            DebugPrint("White: %2u %2u \n", pos.Row, pos.Col);

            state = CheckChessBoard(chessBoard, pos, SPOT_WHITE);
            // 游戏结束
            if (state != STATE_BLACK_WHITE)
            {
                if (state == STATE_ERROR_LOCATION)
                    DebugPrint("End Error Location\n");

                if (state == STATE_WHITE_WIN)
                    DebugPrint("End White Win\n");

                if (state == STATE_NONE_WIN)
                    DebugPrint("End Drawn\n");

                ChessBoardDebugPrint(chessBoard);
                system("pause");
                break;
            }

            chessBoard[pos.Row][pos.Col] = SPOT_WHITE;
        }
    }


}
#include "LNeuralNetwork.h"

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

int main()
{
    /*
    https://wanjun0511.github.io/2017/11/05/DQN/
    https://zhuanlan.zhihu.com/p/32089487
	https://charlesliuyx.github.io/2017/10/18/%E6%B7%B1%E5%85%A5%E6%B5%85%E5%87%BA%E7%9C%8B%E6%87%82AlphaGo%E5%85%83/
    */
    Train();
    //Test(".\\10000-Train.ai");

//     LGameRule gameRule;
//     gameRule.BoardSize = 8;
//     gameRule.ContinuNum = 4;
// 
//     LAiBrainParam aiParam;
//     aiParam.BrainLayersNum = 2;
//     aiParam.LayerNeuronsNum = 128;
// 
//     LAiTrainParam trainParam;
//     trainParam.QLearningRate = 0.5;
//     trainParam.QLearningGamma = 0.9;
//     trainParam.BrainTrainCountMax = 100000;
//     trainParam.BrainLearningRate = 0.1;
// 
//     LGomokuAi ai(gameRule, aiParam);
//     ai.SetTrainParam(trainParam);
// 
//     LChessBoard chessBoard(gameRule.BoardSize, gameRule.BoardSize, SPOT_NONE);                         // 棋盘
// 
//     LChessPos prePos;
//     LChessPos currentPos;
// 
//     ai.Action(chessBoard, prePos, &currentPos);
// 
//     chessBoard[currentPos.Row][currentPos.Col] = SPOT_BLACK;
//     prePos = currentPos;
//     ai.Action(chessBoard, prePos, &currentPos);
// 
//     LTrainData data;
//     data.PreAction = prePos;
//     data.State = chessBoard;
//     data.Action = currentPos;
//     data.GameEnd = true;
//     data.Reward = GAME_LOSE_SCORE;
//     ai.Train(data);
// 
//     ai.Action(chessBoard, prePos, &currentPos);

    system("pause");
    return 0;
}