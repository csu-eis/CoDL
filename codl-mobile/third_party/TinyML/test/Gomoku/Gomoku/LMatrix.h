/// @file LMatrix.h
/// @brief 矩阵模板头文件
/// 
/// Detail:
/// @author Jie Liu Email:coderjie@outlook.com
/// @version   
/// @date 2018/05/23

#ifndef _DATASTRUCT_LMATRIX_H_
#define _DATASTRUCT_LMATRIX_H_

#ifndef LTEMPLATE
#define LTEMPLATE template<typename Type>
#endif

#ifndef IN
#define IN
#endif

#ifndef INOUT
#define INOUT
#endif

#ifndef OUT
#define OUT
#endif

/// @brief 矩阵
LTEMPLATE
class LMatrix
{
public:
    /// @brief 矩阵加法
    /// 要求矩阵A的大小等于矩阵B的大小
    /// @param[in] A 被加数
    /// @param[in] B 加数
    /// @param[out] C 结果矩阵
    /// @return 参数错误返回false
    static bool ADD(IN const LMatrix<Type>& A, IN const LMatrix<Type>& B, OUT LMatrix<Type>& C);

    /// @brief 矩阵减法
    /// 要求矩阵A的大小等于矩阵B的大小
    /// @param[in] A 被减数
    /// @param[in] B 减数
    /// @param[out] C 结果矩阵
    /// @return 参数错误返回false
    static bool SUB(IN const LMatrix<Type>& A, IN const LMatrix<Type>& B, OUT LMatrix<Type>& C);

    /// @brief 矩阵乘法
    /// 要求矩阵A的列数等于矩阵B的行数
    /// 注意: 该操作会改变矩阵的结构, 所以C不能和A或B是相同矩阵的引用
    /// @param[in] A 被乘数
    /// @param[in] B 乘数
    /// @param[out] C 结果矩阵
    /// @return 参数错误返回false
    static bool MUL(IN const LMatrix<Type>& A, IN const LMatrix<Type>& B, OUT LMatrix<Type>& C);

    /// @brief 矩阵数乘
    /// @param[in] A 被乘数
    /// @param[in] B 乘数
    /// @param[out] C 结果矩阵
    /// @return 返回true
    static bool SCALARMUL(IN const LMatrix<Type>& A, IN const Type& B, OUT LMatrix<Type>& C);

    /// @brief 矩阵点乘
    /// 要求矩阵A的大小等于矩阵B的大小
    /// @param[in] A 被乘数
    /// @param[in] B 乘数
    /// @param[out] C 结果矩阵
    /// @return 参数错误返回false
    static bool DOTMUL(IN const LMatrix<Type>& A, IN const LMatrix<Type>& B, OUT LMatrix<Type>& C);

    /// @brief 矩阵数除
    /// @param[in] A 被除数
    /// @param[in] B 除数
    /// @param[out] C 结果矩阵
    /// @return 返回true
    static bool SCALARDIV(IN const LMatrix<Type>& A, IN const Type& B, OUT LMatrix<Type>& C);

    /// @brief 矩阵点除
    /// 要求矩阵A的大小等于矩阵B的大小
    /// @param[in] A 被除数
    /// @param[in] B 除数
    /// @param[out] C 结果矩阵
    /// @return 参数错误返回false
    static bool DOTDIV(IN const LMatrix<Type>& A, IN const LMatrix<Type>& B, OUT LMatrix<Type>& C);

    /// @brief 矩阵转置
    /// 注意: 该操作会改变矩阵的结构, 所以B不能和A是相同矩阵的引用
    /// @param[in] A 需要转置的矩阵
    /// @param[out] B 转置后的结果矩阵
    /// @return 返回true
    static bool T(IN const LMatrix<Type>& A, OUT LMatrix<Type>& B);

    /// @brief 子矩阵
    /// 该操作只能取得连续的子矩阵
    /// 注意: 该操作会改变矩阵的结构, 所以D不能和S是相同矩阵的引用
    /// @param[in] S 源矩阵
    /// @param[in] rowStart 子矩阵开始行
    /// @param[in] rowLen 子矩阵行长度
    /// @param[in] colStart 子矩阵开始列
    /// @param[in] colLen 子矩阵列长度
    /// @param[out] D 存储子矩阵
    /// @return 参数错误返回false
    static bool SUBMATRIX(
        IN const LMatrix<Type>& S, 
        IN unsigned int rowStart, 
        IN unsigned int rowLen, 
        IN unsigned int colStart, 
        IN unsigned int colLen, 
        OUT LMatrix<Type>& D);

public:
    /// @brief 默认构造函数
    /// 默认矩阵长度为0, 即没有数据可被访问
    LMatrix();

    /// @brief 析构函数
    ~LMatrix();

    /// @brief 构造函数
    /// 如果row或col中任一项为0, 则矩阵行数和列数都为0
    /// @param[in] row 矩阵行大小
    /// @param[in] col 矩阵列大小
    LMatrix(IN unsigned int row, IN unsigned int col);

    /// @brief 构造函数, 构造矩阵, 并使用指定数据初始化矩阵
    /// 如果row或col中任一项为0, 则矩阵行数和列数都为0
    /// @param[in] row 矩阵行大小
    /// @param[in] col 矩阵列大小
    /// @param[in] initValue 初始化数据
    LMatrix(IN unsigned int row, IN unsigned int col, IN const Type& initValue);

    /// @brief 构造函数, 构造矩阵, 并使用数组数据初始化矩阵
    /// 如果row或col中任一项为0, 则矩阵行数和列数都为0
    /// @param[in] row 矩阵行大小
    /// @param[in] col 矩阵列大小
    /// @param[in] pDataList 矩阵数据
    LMatrix(IN unsigned int row, IN unsigned int col, IN const Type* pDataList);

    /// @brief 拷贝构造函数
    LMatrix(IN const LMatrix<Type>& rhs);

    /// @brief 赋值操作符
    LMatrix<Type>& operator = (IN const LMatrix<Type>& rhs);

    /// @brief 矩阵加法
    /// 要求自身矩阵大小等于矩阵B的大小
    /// @param[in] B 加数
    /// @return 结果矩阵
    LMatrix<Type> operator + (IN const LMatrix<Type>& B) const;

    /// @brief 矩阵自加
    /// 要求自身矩阵大小等于矩阵B的大小
    /// @param[in] B 加数
    LMatrix<Type>& operator += (IN const LMatrix<Type>& B);

    /// @brief 矩阵减法
    /// 要求自身矩阵大小等于矩阵B的大小
    /// @param[in] B 减数
    /// @return 结果矩阵
    LMatrix<Type> operator - (IN const LMatrix<Type>& B) const;

    /// @brief 矩阵自减
    /// 要求自身矩阵大小等于矩阵B的大小
    /// @param[in] B 减数
    LMatrix<Type>& operator -= (IN const LMatrix<Type>& B);

    /// @brief 矩阵乘法
    /// 要求自身矩阵的列数等于矩阵B的行数
    /// @param[in] B 乘数
    /// @return 结果矩阵
    LMatrix<Type> operator * (IN const LMatrix<Type>& B) const;

    /// @brief 矩阵自乘
    /// 要求自身矩阵的列数等于矩阵B的行数
    /// @param[in] B 乘数
    LMatrix<Type>& operator *= (IN const LMatrix<Type>& B);

    /// @brief 矩阵数乘
    /// @param[in] B 乘数
    /// @return 结果矩阵
    LMatrix<Type> ScalarMul(IN const Type& B) const;

    /// @brief 矩阵数除
    /// @param[in] B 除数
    /// @return 结果矩阵
    LMatrix<Type> ScalarDiv(IN const Type& B) const;

    /// @brief []操作符
    /// @param[in] row 矩阵行
    Type*& operator[](IN unsigned int row);

    /// @brief []操作符
    /// @param[in] row 矩阵行
    const Type* operator[](IN unsigned int row) const;

    /// @brief 判断矩阵是否为空
    /// 行数或列数为0的矩阵为空
    /// @return true, false
    bool Empty() const;

    /// @brief 判断是否是方阵
    /// 非空以及行数列数相等的矩阵为方阵
    /// @return true, false
    bool Square() const;

    /// @brief 矩阵转置
    /// @return 转置后的结果矩阵
    LMatrix<Type> T() const;

    /// @brief 矩阵行拆分(拆分为两个矩阵)
    /// @param[in] rowIdx 拆分的行索引(索引行被包含在下矩阵中)
    /// @param[out] up 存储上矩阵
    /// @param[out] down 存储下矩阵
    void SplitRow(IN unsigned int rowIdx, OUT LMatrix<Type>& up, OUT LMatrix<Type>& down) const;

    /// @brief 矩阵列拆分(拆分为两个矩阵)
    /// @param[in] colIdx 拆分的列索引(索引列被包含在右矩阵中)
    /// @param[out] left 存储左矩阵
    /// @param[out] right 存储右矩阵
    void SplitCloumn(IN unsigned int colIdx, OUT LMatrix<Type>& left, OUT LMatrix<Type>& right) const;

    /// @brief 获取子矩阵
    /// @param[in] rowStart 子矩阵开始行
    /// @param[in] rowLen 子矩阵行长度
    /// @param[in] colStart 子矩阵开始列
    /// @param[in] colLen 子矩阵列长度
    /// @return 子矩阵
    LMatrix<Type> SubMatrix(IN unsigned int rowStart, IN unsigned int rowLen, IN unsigned int colStart, IN unsigned int colLen) const;

    /// @brief 获取子矩阵
    /// @param[in] rowStart 子矩阵开始行
    /// @param[in] rowLen 子矩阵行长度
    /// @param[in] colStart 子矩阵开始列
    /// @param[in] colLen 子矩阵列长度
    /// @param[out] D 存储子矩阵
    void SubMatrix(IN unsigned int rowStart, IN unsigned int rowLen, IN unsigned int colStart, IN unsigned int colLen, OUT LMatrix<Type>& D) const;

    /// @brief 获取矩阵中的一行数据
    /// @param[in] row 行索引
    /// @return 行向量
    LMatrix<Type> GetRow(IN unsigned int row) const;

    /// @brief 获取矩阵中的一行数据
    /// @param[in] row 行索引
    /// @param[out] rowVector 存储行数据
    void GetRow(IN unsigned int row, OUT LMatrix<Type>& rowVector) const;

    /// @brief 获取矩阵中的一列数据
    /// @param[in] col 列索引
    /// @return 列向量
    LMatrix<Type> GetColumn(IN unsigned int col) const;

    /// @brief 获取矩阵中的一列数据
    /// @param[in] col 列索引
    /// @param[out] colVector 存储列数据
    void GetColumn(IN unsigned int col, OUT LMatrix<Type>& colVector) const;

    /// @brief 重置矩阵
    /// 如果row或col中任一项为0, 则矩阵行数和列数都为0
    /// @param[in] row 矩阵行大小
    /// @param[in] col 矩阵列大小
    void Reset(IN unsigned int row, IN unsigned int col);

    /// @brief 重置矩阵, 并使用initValue初始化矩阵中的所有值
    /// 如果row或col中任一项为0, 则矩阵行数和列数都为0
    /// @param[in] row 矩阵行大小
    /// @param[in] col 矩阵列大小
    /// @param[in] initValue 初始化值
    void Reset(IN unsigned int row, IN unsigned int col, IN const Type& initValue);

public:
    const unsigned int& RowLen;     ///< 行长度属性
    const unsigned int& ColumnLen;  ///< 列长度属性

private:
    Type** m_dataTable;             ///< 二维数据表
    Type*  m_dataList;              ///< 实际存储的数据列表
    unsigned int m_rowLen;          ///< 矩阵行长度
    unsigned int m_columnLen;       ///< 矩阵列长度
};

LTEMPLATE
LMatrix<Type>::LMatrix()
: m_rowLen(0), m_columnLen(0), RowLen(m_rowLen), ColumnLen(m_columnLen), m_dataTable(0), m_dataList(0)
{

}

LTEMPLATE
LMatrix<Type>::~LMatrix()
{
    if (this->m_dataTable)
    {
        delete[] this->m_dataTable;
        this->m_dataTable = 0;
    }

    if (this->m_dataList)
    {
        delete[] this->m_dataList;
        this->m_dataList = 0;
    }

    this->m_rowLen = 0;
    this->m_columnLen = 0;
}

LTEMPLATE
LMatrix<Type>::LMatrix(IN unsigned int row, IN unsigned int col)
: m_rowLen(0), m_columnLen(0), RowLen(m_rowLen), ColumnLen(m_columnLen), m_dataTable(0), m_dataList(0)
{
    this->Reset(row, col);
}

LTEMPLATE
LMatrix<Type>::LMatrix(IN unsigned int row, IN unsigned int col, IN const Type& initValue)
: m_rowLen(0), m_columnLen(0), RowLen(m_rowLen), ColumnLen(m_columnLen), m_dataTable(0), m_dataList(0)
{
    this->Reset(row, col);

    unsigned int size = row * col;
    for (unsigned int i = 0; i < size; i++)
    {
        this->m_dataList[i] = initValue;
    }
}

LTEMPLATE
LMatrix<Type>::LMatrix(IN unsigned int row, IN unsigned int col, IN const Type* pDataList)
: m_rowLen(0), m_columnLen(0), RowLen(m_rowLen), ColumnLen(m_columnLen), m_dataTable(0), m_dataList(0)
{
    this->Reset(row, col);

    unsigned int size = row * col;
    for (unsigned int i = 0; i < size; i++)
    {
        this->m_dataList[i] = pDataList[i];
    }

}

LTEMPLATE
LMatrix<Type>::LMatrix(IN const LMatrix<Type>& rhs)
: m_rowLen(0), m_columnLen(0), RowLen(m_rowLen), ColumnLen(m_columnLen), m_dataTable(0), m_dataList(0)
{
    this->Reset(rhs.RowLen, rhs.ColumnLen);

    unsigned int size = rhs.RowLen * rhs.ColumnLen;
    for (unsigned int i = 0; i < size; i++)
    {
        this->m_dataList[i] = rhs.m_dataList[i];
    }

}

LTEMPLATE
LMatrix<Type>& LMatrix<Type>::operator = (IN const LMatrix<Type>& rhs)
{
    this->Reset(rhs.RowLen, rhs.ColumnLen);

    unsigned int size = rhs.RowLen * rhs.ColumnLen;
    for (unsigned int i = 0; i < size; i++)
    {
        this->m_dataList[i] = rhs.m_dataList[i];
    }

    return *this;
}

LTEMPLATE
LMatrix<Type> LMatrix<Type>::operator + (IN const LMatrix<Type>& B) const
{
    LMatrix<Type> C;
    ADD(*this, B, C);
    return C;
}

LTEMPLATE
LMatrix<Type>& LMatrix<Type>::operator += (IN const LMatrix<Type>& B)
{
    ADD(*this, B, *this);
    return *this;
}

LTEMPLATE
LMatrix<Type> LMatrix<Type>::operator - (IN const LMatrix<Type>& B) const
{
    LMatrix<Type> C;
    SUB(*this, B, C);
    return C;
}

LTEMPLATE
LMatrix<Type>& LMatrix<Type>::operator -= (IN const LMatrix<Type>& B)
{
    SUB(*this, B, *this);
    return *this;
}

LTEMPLATE
LMatrix<Type> LMatrix<Type>::operator * (IN const LMatrix<Type>& B) const
{
    LMatrix<Type> C;
    MUL(*this, B, C);
    return C;
}

LTEMPLATE
LMatrix<Type>& LMatrix<Type>::operator *= (IN const LMatrix<Type>& B)
{
    LMatrix<Type> C;
    MUL(*this, B, C);
    (*this) = C;
    return *this;
}

LTEMPLATE
LMatrix<Type> LMatrix<Type>::ScalarMul(IN const Type& B) const
{
    LMatrix<Type> C;
    SCALARMUL(*this, B, C);
    return C;
}

LTEMPLATE
LMatrix<Type> LMatrix<Type>::ScalarDiv(IN const Type& B) const
{
    LMatrix<Type> C;
    SCALARDIV(*this, B, C);
    return C;
}

LTEMPLATE
Type*& LMatrix<Type>::operator[](IN unsigned int row)
{
    return this->m_dataTable[row];
}

LTEMPLATE
const Type* LMatrix<Type>::operator[](IN unsigned int row) const
{
    return this->m_dataTable[row];
}

LTEMPLATE
bool LMatrix<Type>::Empty() const
{
    if (this->m_columnLen == 0 || this->m_rowLen == 0)
        return true;

    return false;
}

LTEMPLATE
bool LMatrix<Type>::Square() const
{
    if (this->m_columnLen == this->m_rowLen)
    {
        if (this->m_columnLen != 0)
            return true;
    }

    return false;
}

LTEMPLATE
LMatrix<Type> LMatrix<Type>::T() const
{
    LMatrix<Type> B;
    T(*this, B);
    return B;
}

LTEMPLATE
void LMatrix<Type>::SplitRow(IN unsigned int rowIdx, OUT LMatrix<Type>& up, OUT LMatrix<Type>& down) const
{
    up.Reset(0, 0);
    down.Reset(0, 0);

    SUBMATRIX(*this, 0, rowIdx, 0, this->ColumnLen, up);
    SUBMATRIX(*this, rowIdx, this->RowLen-rowIdx, 0, this->ColumnLen, down);
}

LTEMPLATE
void LMatrix<Type>::SplitCloumn(IN unsigned int colIdx, OUT LMatrix<Type>& left, OUT LMatrix<Type>& right) const
{
    left.Reset(0, 0);
    right.Reset(0, 0);

    SUBMATRIX(*this, 0, this->RowLen, 0, colIdx, left);
    SUBMATRIX(*this, 0, this->RowLen, colIdx, this->ColumnLen - colIdx, right);
}

LTEMPLATE
LMatrix<Type> LMatrix<Type>::SubMatrix(IN unsigned int rowStart, IN unsigned int rowLen, IN unsigned int colStart, IN unsigned int colLen) const
{
    LMatrix<Type> D;
    SUBMATRIX(*this, rowStart, rowLen, colStart, colLen, D);
    return D;
}

LTEMPLATE
void LMatrix<Type>::SubMatrix(IN unsigned int rowStart, IN unsigned int rowLen, IN unsigned int colStart, IN unsigned int colLen, OUT LMatrix<Type>& D) const
{
    SUBMATRIX(*this, rowStart, rowLen, colStart, colLen, D);
}

LTEMPLATE
LMatrix<Type> LMatrix<Type>::GetRow(IN unsigned int row) const
{
    LMatrix<Type> rowVector(1, this->m_columnLen);
    this->GetRow(row, rowVector);

    return rowVector;
}

LTEMPLATE
void LMatrix<Type>::GetRow(IN unsigned int row, OUT LMatrix<Type>& rowVector) const
{
    rowVector.Reset(1, this->m_columnLen);
    for (unsigned int i = 0; i < this->m_columnLen; i++)
    {
        rowVector.m_dataTable[0][i] = this->m_dataTable[row][i];
    }
}

LTEMPLATE
LMatrix<Type> LMatrix<Type>::GetColumn(IN unsigned int col) const
{
    LMatrix<Type> columnVector(this->m_rowLen, 1);
    this->GetColumn(col, columnVector);

    return columnVector;
}

LTEMPLATE
void LMatrix<Type>::GetColumn(IN unsigned int col, OUT LMatrix<Type>& colVector) const
{
    colVector.Reset(this->m_rowLen, 1);
    for (unsigned int i = 0; i < this->m_rowLen; i++)
    {
        colVector.m_dataTable[i][0] = this->m_dataTable[i][col];
    }
}

LTEMPLATE
void LMatrix<Type>::Reset(IN unsigned int row, IN unsigned int col)
{
    if ((this->m_rowLen != row) || this->m_columnLen != col)
    {
        if (this->m_dataTable)
        {
            delete[] this->m_dataTable;
            this->m_dataTable = 0;
        }

        if (this->m_dataList)
        {
            delete[] this->m_dataList;
            this->m_dataList = 0;
        }

        
        if (row * col > 0)
        {
            this->m_rowLen = row;
            this->m_columnLen = col;

            this->m_dataTable = new Type*[this->m_rowLen];
            this->m_dataList = new Type[this->m_rowLen * this->m_columnLen];
            for (unsigned int i = 0; i < this->m_rowLen; i++)
            {
                this->m_dataTable[i] = &this->m_dataList[this->m_columnLen * i];
            }
        }
        else
        {
            this->m_rowLen = 0;
            this->m_columnLen = 0;
        }
    }
}

LTEMPLATE
void LMatrix<Type>::Reset(IN unsigned int row, IN unsigned int col, IN const Type& initValue)
{
    this->Reset(row, col);

    unsigned int size = row * col;
    for (unsigned int i = 0; i < size; i++)
    {
        this->m_dataList[i] = initValue;
    }
}

LTEMPLATE
bool LMatrix<Type>::ADD(IN const LMatrix<Type>& A, IN const LMatrix<Type>& B, OUT LMatrix<Type>& C)
{
    if ((A.RowLen != B.RowLen) || (A.ColumnLen != B.ColumnLen))
        return false;

    C.Reset(A.RowLen, A.ColumnLen);

    for (unsigned int i = 0; i < C.RowLen; i++)
    {
        for (unsigned int j = 0; j < C.ColumnLen; j++)
        {
            C.m_dataTable[i][j] = A.m_dataTable[i][j] + B.m_dataTable[i][j];
        }
    }

    return true;
}

LTEMPLATE
bool LMatrix<Type>::SUB(IN const LMatrix<Type>& A, IN const LMatrix<Type>& B, OUT LMatrix<Type>& C)
{
    if ((A.RowLen != B.RowLen) || (A.ColumnLen != B.ColumnLen))
        return false;

    C.Reset(A.RowLen, A.ColumnLen);

    for (unsigned int i = 0; i < C.RowLen; i++)
    {
        for (unsigned int j = 0; j < C.ColumnLen; j++)
        {
            C.m_dataTable[i][j] = A.m_dataTable[i][j] - B.m_dataTable[i][j];
        }
    }

    return true;
}



LTEMPLATE
bool LMatrix<Type>::MUL(IN const LMatrix<Type>& A, IN const LMatrix<Type>& B, OUT LMatrix<Type>& C)
{
    if (A.ColumnLen != B.RowLen)
        return false;

    C.Reset(A.RowLen, B.ColumnLen);

    for (unsigned int i = 0; i < C.RowLen; i++)
    {
        for (unsigned int j = 0; j < C.ColumnLen; j++)
        {
            C.m_dataTable[i][j] = A.m_dataTable[i][0] * B.m_dataTable[0][j];
            for (unsigned int k = 1; k < A.ColumnLen; k++)
            {
                C.m_dataTable[i][j] += A.m_dataTable[i][k] * B.m_dataTable[k][j];
            }
        }
    }

    return true;
}

LTEMPLATE
bool LMatrix<Type>::SCALARMUL(IN const LMatrix<Type>& A, IN const Type& B, OUT LMatrix<Type>& C)
{
    C.Reset(A.RowLen, A.ColumnLen);
    for (unsigned int row = 0; row < A.RowLen; row++)
    {
        for (unsigned int col = 0; col < A.ColumnLen; col++)
        {
            C.m_dataTable[row][col] = A.m_dataTable[row][col] * B;
        }
    }

    return true;
}

LTEMPLATE
bool LMatrix<Type>::DOTMUL(IN const LMatrix<Type>& A, IN const LMatrix<Type>& B, OUT LMatrix<Type>& C)
{
    if ((A.RowLen != B.RowLen) || (A.ColumnLen != B.ColumnLen))
        return false;

    C.Reset(A.RowLen, A.ColumnLen);

    for (unsigned int i = 0; i < C.RowLen; i++)
    {
        for (unsigned int j = 0; j < C.ColumnLen; j++)
        {
            C.m_dataTable[i][j] = A.m_dataTable[i][j] * B.m_dataTable[i][j];
        }
    }

    return true;
}

LTEMPLATE
bool LMatrix<Type>::SCALARDIV(IN const LMatrix<Type>& A, IN const Type& B, OUT LMatrix<Type>& C)
{
    C.Reset(A.RowLen, A.ColumnLen);
    for (unsigned int row = 0; row < A.RowLen; row++)
    {
        for (unsigned int col = 0; col < A.ColumnLen; col++)
        {
            C.m_dataTable[row][col] = A.m_dataTable[row][col] / B;
        }
    }

    return true;
}

LTEMPLATE
bool LMatrix<Type>::DOTDIV(IN const LMatrix<Type>& A, IN const LMatrix<Type>& B, OUT LMatrix<Type>& C)
{
    if ((A.RowLen != B.RowLen) || (A.ColumnLen != B.ColumnLen))
        return false;

    C.Reset(A.RowLen, A.ColumnLen);

    for (unsigned int i = 0; i < C.RowLen; i++)
    {
        for (unsigned int j = 0; j < C.ColumnLen; j++)
        {
            C.m_dataTable[i][j] = A.m_dataTable[i][j] / B.m_dataTable[i][j];
        }
    }

    return true;
}

LTEMPLATE
bool LMatrix<Type>::T(IN const LMatrix<Type>& A, OUT LMatrix<Type>& B)
{
    B.Reset(A.ColumnLen, A.RowLen);
    for (unsigned int i = 0; i < A.RowLen; i++)
    {
        for (unsigned int j = 0; j < A.ColumnLen; j++)
            B.m_dataTable[j][i] = A.m_dataTable[i][j];
    }

    return true;
}

LTEMPLATE
bool LMatrix<Type>::SUBMATRIX(
IN const LMatrix<Type>& S, 
IN unsigned int rowStart, 
IN unsigned int rowLen, 
IN unsigned int colStart, 
IN unsigned int colLen, 
OUT LMatrix<Type>& D)
{
    if ((rowStart + rowLen) > S.RowLen)
        return false;

    if ((colStart + colLen) > S.ColumnLen)
        return false;

    if (rowLen < 1 || colLen < 1)
        return false;

    D.Reset(rowLen, colLen);

    for (unsigned int row = 0; row < D.RowLen; row++)
    {
        for (unsigned int col = 0; col < D.ColumnLen; col++)
        {
            D[row][col] = S[rowStart + row][colStart + col];
        }
    }

    return true;
}




#endif
