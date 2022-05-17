
#ifndef _LDOCCLASSIFY_H_
#define _LDOCCLASSIFY_H_

#include <string>
#include <map>
#include <set>
using std::string;
using std::map;
using std::set;

#include "LDataStruct/LArray.h"


/// @文档类别
enum LDOC_CATEGORY
{
    LDOC_CAT_BAD = 0,
    LDOC_CAT_GOOD = 1,
    LDOC_CAT_UNKNOWN
};


/// @brief 文档分类
class LDocCategory
{
public:
    LDocCategory();
    virtual ~LDocCategory();
public:
    /// @brief 增加指定分类的计数
    /// @param[in] cat
    /// @return 参数错误返回false
    bool IncCount(LDOC_CATEGORY cat);

    /// @brief 获取指定分类的计数
    /// @param[in] LDOC_CATEGORY cat
    /// @return 参数错误返回false
    int GetCount(LDOC_CATEGORY cat);

    /// @brief 获取所有分类的总计数
    /// @return 总计数
    int GetTotalCount();

private:
    int m_goodCounts; ///< GOOD分类的计数
    int m_badCounts; ///< BAD分类的计数
private:
    //LDocCategory(const LDocCategory&);
   // LDocCategory& operator = (const LDocCategory&);
};

/// @brief 文档特征
class LDocFeature
{
public:
    LDocFeature();
    virtual ~LDocFeature();

    /// @brief 增加该特征在指定分类中的计数
    /// @param[in] cat 指定分类
    /// @return 参数错误返回false
    bool IncCategoryCount(LDOC_CATEGORY cat);

    /// @brief 获取特征在指定分类中的计数
    /// @param[in] cat 指定分类
    /// @return 
    int GetCategoryCount(LDOC_CATEGORY cat);

    /// @brief 获取特征在所有分类中的总计数
    /// @return 总计数
    int GetCategoryTotalCount();

private:
    LDocCategory m_docCategory; ///< 包含特征的文档分类

private:
    //LDocFeature(const LDocFeature&);
    //LDocFeature& operator = (const LDocFeature&);
};



/// @brief 文档分类器
class LDocClassifier
{
public:
    LDocClassifier();
    virtual ~LDocClassifier();
    
public:
    /// @brief 训练分类器
    /// @param[in] text 文档(要求单词间以空格隔开)
    /// @param[in] cat 文档的类别
    /// @return 参数错误返回false
    bool Train(const string& text, LDOC_CATEGORY cat);

    /// @brief 获取指定文档属于某个分类的概率
    /// @param[in] text 文档
    /// @param[in] cat 分类
    /// @return 概率 即 Pr(Category | Document )
    virtual float GetCatgoryProbInDoc(const string& text, LDOC_CATEGORY cat) = 0;

    /// @brief 对文档进行分类
    /// @param[in] text
    /// @return 文档的类别
    LDOC_CATEGORY Classify(const string& text);

protected:
    /// @brief 从给定的文本中获取特征(即不同的单词)
    /// @param[in] text 文本
    /// @param[in] featuresSet 返回的单词集
    /// @return true
    bool GetFeatures(const string& text, set<string>& featuresSet);

    /// @brief 特征在指定的分类中出现的概率
    /// @param[in] feature 特征
    /// @param[in] cat 分类
    /// @return 概率(范围[0, 1]) 即 Pr(Feature| Category)
    float GetFeatureProbInCat(const string& feature, LDOC_CATEGORY cat);

    // @brief 特征在指定的分类中出现的权重概率
    ///
    /// 所有概率值均以0.5做初始值，训练后允许向其他概率值变化
    /// @param[in] feature 特征
    /// @param[in] cat 分类
    /// @return 概率(范围[0, 1]) 即 Pr(Feature| Category)
    float GetFeatureWeightedProbInCat(const string& feature, LDOC_CATEGORY cat);

protected:
    map<string, LDocFeature> m_featureMap; ///< 特征字典
    LDocCategory m_docCategoryTotal; ///< 总的文档分类

    LArray<LDOC_CATEGORY> m_docCatList; ///< 文档分类列表

private:
    LDocClassifier(const LDocClassifier&);
    LDocClassifier& operator = (const LDocClassifier&);
};

/// @brief 朴素贝叶斯分类器
class LNaiveBayesClassifier : public LDocClassifier
{
public:
    LNaiveBayesClassifier();
    ~LNaiveBayesClassifier();
public:
    /// @brief 获取指定文档属于某个分类的概率
    /// @param[in] text 文档
    /// @param[in] cat 分类
    /// @return 概率 即 Pr(Category | Document )
    virtual float GetCatgoryProbInDoc(const string& text, LDOC_CATEGORY cat);

private:
    /// @brief 获取指定分类下出现某个文档的概率
    /// @param[in] text 文档
    /// @param[in] cat 分类
    /// @return 概率 即 Pr(Document | Category)
    float GetDocProbInCat(const string& text, LDOC_CATEGORY cat);

private:
    LNaiveBayesClassifier(const LNaiveBayesClassifier&);
    LNaiveBayesClassifier& operator = (const LNaiveBayesClassifier&);
};

/// @brief  费舍尔分类器
class LFisherClassifier : public LDocClassifier
{
public:
    LFisherClassifier();
    ~LFisherClassifier();
public:
    /// @brief 获取指定文档属于某个分类的概率
    /// @param[in] text 文档
    /// @param[in] cat 分类
    /// @return 概率 即 Pr(Category | Document )
    virtual float GetCatgoryProbInDoc(const string& text, LDOC_CATEGORY cat);

private:
    /// @brief 获取指定特征属于某个分类的概率
    ///
    /// 归一化计算
    /// @param[in] feature 特征
    /// @param[in] cat 分类
    /// @return 概率 即 Pr(Category | Feature )
    float GetCatgoryProbInFea(const string& feature, LDOC_CATEGORY cat);

    /// @brief 获取指定特征属于某个分类的权重概率
    ///
    /// 归一化计算
    /// @param[in] feature 特征
    /// @param[in] cat 分类
    /// @return 概率 即 Pr(Category | Feature )
    float GetCatgoryWeightProbInFea(const string& feature, LDOC_CATEGORY cat);

public:
    /// @brief
    /// @param[in] chi
    /// @param[in] df
    /// @return 
    float Inchi2(float chi, int df);
private:
    LFisherClassifier(const LFisherClassifier&);
    LFisherClassifier& operator = (const LFisherClassifier&);
};

#endif