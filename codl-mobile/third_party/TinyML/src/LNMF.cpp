#include "LNMF.h"

#include <stdlib.h>

/// @brief 产生随机小数, 范围0.0f~1.0f
/// @return 随机小数
static float RandFloat()		   
{
	return (rand())/(RAND_MAX + 1.0f);
}

/// @brief 将矩阵中的值填入随机值(0.0f~1.0f)
static void RandNMFMatrix(INOUT LNMFMatrix& m)
{
    for (unsigned int i = 0; i < m.RowLen; i++)
    {
        for (unsigned int j = 0; j < m.ColumnLen; j++)
        {
            m[i][j] = RandFloat();
        }
    }
}


LNMF::LNMF()
{

}

LNMF::~LNMF()
{

}

bool LNMF::Factoring(IN const LNMFProblem& problem, OUT LNMFMatrix* pW, OUT LNMFMatrix* pH)
{
    // 检查参数
    if (0 == pW || 0 == pH)
        return false;

    if (0 == problem.R || 0 == problem.IterCount)
        return false;

    if (0 == problem.V.RowLen || 0 == problem.V.ColumnLen)
    {
        return false;
    }

    for (unsigned int row = 0; row < problem.V.RowLen; row++)
    {
        for (unsigned int col = 0; col < problem.V.ColumnLen; col++)
        {
            if (problem.V[row][col] < 0.0f)
                return false;
        }
    }

    const LNMFMatrix& V = problem.V; // 原始矩阵
    LNMFMatrix W; // 基矩阵
    LNMFMatrix H; // 系数矩阵

    // 生成基矩阵
    W.Reset(V.RowLen, problem.R);
    RandNMFMatrix(W);

    // 生成特征矩阵
    H.Reset(problem.R, V.ColumnLen);
    RandNMFMatrix(H);

    LNMFMatrix TF;
    LNMFMatrix TW;
    LNMFMatrix TWW;

    LNMFMatrix HN;
    LNMFMatrix HD;
    LNMFMatrix HHN;

    LNMFMatrix WN;
    LNMFMatrix WD;
    LNMFMatrix WH;
    LNMFMatrix WWN;

    // 迭代求解
    for (unsigned int i = 0; i < problem.IterCount; i++)
    {
        LNMFMatrix::T(W, TW);
        LNMFMatrix::MUL(TW, V, HN);
        LNMFMatrix::MUL(TW, W, TWW);
        LNMFMatrix::MUL(TWW, H, HD);

        LNMFMatrix::DOTMUL(H, HN, HHN);
        LNMFMatrix::DOTDIV(HHN, HD, H);


        LNMFMatrix::T(H, TF);
        LNMFMatrix::MUL(V, TF, WN);
        LNMFMatrix::MUL(W, H, WH);
        LNMFMatrix::MUL(WH, TF, WD);

        LNMFMatrix::DOTMUL(W, WN, WWN);
        LNMFMatrix::DOTDIV(WWN, WD, W);
    }

    (*pW) = W;
    (*pH) = H;


    return true;

}