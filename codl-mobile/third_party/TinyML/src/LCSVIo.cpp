#include "../include/LCSVIo.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using std::ios;
using std::string;
using std::wstring;
using std::wifstream;
using std::wstringstream;
using std::vector;

/// @brief 字符串分割
/// @param[in] ch 分割字符
/// @param[in] srcStr 源字符串
/// @param[in] strList 分割后的字符串列表
static void WStringSplit(IN const wchar_t ch, IN const wstring& srcStr, OUT vector<wstring>& strList)
{
    strList.clear();

    size_t startPos = 0;
    size_t length = 0;

    size_t srcLen = srcStr.length();
    for (size_t i = 0; i < srcLen; i++)
    {
        if (srcStr[i] == ch)
        {
            strList.push_back(srcStr.substr(startPos, length));
            startPos = i + 1;
            length = 0;
        }
        else
        {
            length++;
        }
    }

    strList.push_back(srcStr.substr(startPos, length));
}

/// <SUMMARY>
/// 字符串裁剪
/// 去除字符串开始和结束的空白字符, 如: 空格, 换行, 回车, 制表符
/// </SUMMARY>
static void WStringTrimmed(INOUT wstring& str)
{
    if (str.empty())
    {
        return;
    }

    size_t i = 0;
    while (iswspace(str[i]) != 0)
    {
        i++;
    }
    str.erase(0, i);

    if (str.empty())
    {
        return;
    }

    size_t j = str.length() - 1;
    while (isspace(str[j]) != 0)
    {
        if (j <= 0)
            break;

        j--;
    }

    str.erase(j + 1);

}

/// <SUMMARY>
/// 字符串转换为浮点数
/// </SUMMARY>
static double StringToDouble(IN const wstring& str)
{
    double value;
    wstringstream strStream(str);
    strStream >> value;
    return value;
}



/// @brief CSV文件解析器
class CCSVParser
{
public:
    /// @brief 构造函数
    CCSVParser(IN const char* fileName)
    {
        m_fileName = fileName;
        m_bSkipHeader = false;
        m_delimiter = L',';
    }

    /// @brief 析构函数
    ~CCSVParser()
    {

    }

    /// @brief 设置是否跳过首行
    void SetSkipHeader(IN bool skip)
    {
        m_bSkipHeader = skip;
    }

    /// @brief 设置分隔符
    void SetDelimiter(IN char ch)
    {
        m_delimiter = ch;
    }

    /// @brief 加载所有数据
    bool LoadAllData(OUT LDataMatrix& dataMatrix)
    {
        wstring str;
        wifstream fin(m_fileName, ios::in);

        // 文件不存在
        if (!fin) 
        {
            return false;
        }

        // 检查是否需要跳过首行
        if (m_bSkipHeader)
            getline(fin, str);

        vector<vector<wstring>> strMatrix;
        while (getline(fin, str))
        {
            // 去除开头和结尾的空白字符
            WStringTrimmed(str);
            if (str.empty())
                continue;

            // 分割字符串
            vector<wstring> strList;
            WStringSplit(m_delimiter, str, strList);

            // 检查每个字符串是否为空
            for (auto iter = strList.begin(); iter != strList.end(); iter++)
            {
                WStringTrimmed(*iter);
                if (iter->empty())
                    return false;
            }

            strMatrix.push_back(strList);
        }

        // 行长度
        size_t rowLength = strMatrix.size();
        if (rowLength < 1)
            return false;
        
        // 列长度
        size_t colLength = strMatrix[0].size();
        if (colLength < 1)
            return false;

        dataMatrix.Reset((unsigned int)rowLength, (unsigned int)colLength, 0.0);

        for (size_t row = 0; row < rowLength; row++)
        {
            // 检查每一行中的数据长度是否一致
            if (strMatrix[row].size() != colLength)
            {
                dataMatrix.Reset(0, 0);
                return false;
            }

            for (size_t col = 0; col < colLength; col++)
            {
                dataMatrix[(unsigned int)row][(unsigned int)col] = StringToDouble(strMatrix[row][col]);
            }
        }

        return true;
    }

private:
    bool m_bSkipHeader; ///< 跳过首行
    char m_delimiter; ///< 分隔符
    string m_fileName; ///< 文件名

};

LCSVParser::LCSVParser(IN const char* fileName)
{
    m_pParser = nullptr;
    m_pParser = new CCSVParser(fileName);
}

LCSVParser::~LCSVParser()
{
    if (nullptr != m_pParser)
    {
        delete m_pParser;
        m_pParser = nullptr;
    }
}

void LCSVParser::SetSkipHeader(IN bool skip)
{
    m_pParser->SetSkipHeader(skip);
}

void LCSVParser::SetDelimiter(IN char ch)
{
    m_pParser->SetDelimiter(ch);
}

bool LCSVParser::LoadAllData(OUT LDataMatrix& dataMatrix)
{
    return m_pParser->LoadAllData(dataMatrix);
}
