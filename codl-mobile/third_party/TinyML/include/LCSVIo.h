/// @file LCSVIo.h
/// @brief 本文声明了CSV文件操作类
/// LCSVParser(CSV文件解析器)
/// Detail:
/// @author Jie Liu Email:coderjie@outlook.com
/// @version   
/// @date 2018/01/30

#ifndef _LCSVIO_H_
#define _LCSVIO_H_

#include "LMatrix.h"


typedef LMatrix<double> LDataMatrix;     ///< 数据矩阵

class CCSVParser;

/// @brief CSV文件解析器
/// 本解析器只支持解析数值数据, 并且不能有缺失数据
class LCSVParser
{
public:
    /// @brief 构造函数
    explicit LCSVParser(IN const char* fileName);

    /// @brief 析构函数
    ~LCSVParser();

    /// @brief 设置是否跳过首行
    /// 默认不跳过首行
    /// @param[in] skip true(跳过), false(不跳过)
    void SetSkipHeader(IN bool skip);

    /// @brief 设置分隔符
    /// 默认分隔符为 ','
    /// @param[in] ch 设置的分隔符
    void SetDelimiter(IN char ch);

    /// @brief 加载所有数据
    /// @param[in] dataMatrix 存储数据
    bool LoadAllData(OUT LDataMatrix& dataMatrix);

private:
    CCSVParser* m_pParser; ///< CSV文件解析器实现对象
};

#endif
