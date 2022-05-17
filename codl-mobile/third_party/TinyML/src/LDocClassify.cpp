
#include "LDocClassify.h"

#include <cmath>

#include "LDataStruct/LString.h"


LDocCategory::LDocCategory()
{
    m_badCounts = 0;
    m_goodCounts = 0;
}

LDocCategory::~LDocCategory()
{
    
}

bool LDocCategory::IncCount(LDOC_CATEGORY cat)
{
    switch (cat)
    {
    case LDOC_CAT_BAD:
        ++m_badCounts;
        return true;
        break;
    case LDOC_CAT_GOOD:
        ++m_goodCounts;
        return true;
        break;
    default:
        return false;
        break;
    }

    return false;
}

int LDocCategory::GetCount(LDOC_CATEGORY cat)
{
    switch (cat)
    {
    case LDOC_CAT_BAD:
        return m_badCounts;
        break;
    case LDOC_CAT_GOOD:
        return m_goodCounts;
        break;
    default:
        break;
    }

    return 0;
}

int LDocCategory::GetTotalCount()
{
    return m_goodCounts + m_badCounts;
}

LDocFeature::LDocFeature()
{

}

LDocFeature::~LDocFeature()
{

}


bool LDocFeature::IncCategoryCount(LDOC_CATEGORY cat)
{
    return m_docCategory.IncCount(cat);
}

int LDocFeature::GetCategoryCount(LDOC_CATEGORY cat)
{
    return m_docCategory.GetCount(cat);
}

int LDocFeature::GetCategoryTotalCount()
{
    return m_docCategory.GetTotalCount();
}

LDocClassifier::LDocClassifier()
{
    m_docCatList.Reset(2);
    m_docCatList.Data[0] = LDOC_CAT_BAD;
    m_docCatList.Data[1] = LDOC_CAT_GOOD;
}

LDocClassifier::~LDocClassifier()
{

}



bool LDocClassifier::Train(const string& text, LDOC_CATEGORY cat)
{
    
    set<string> featureSet;

    this->GetFeatures(text, featureSet);
    if (featureSet.size() == 0)
        return false;

    for (auto iter = featureSet.begin(); iter != featureSet.end(); iter++)
    {
        m_featureMap[*iter].IncCategoryCount(cat);
    }

    m_docCategoryTotal.IncCount(cat);
    return true;

}

LDOC_CATEGORY LDocClassifier::Classify(const string& text)
{
    LDOC_CATEGORY bestCat = LDOC_CAT_UNKNOWN;
    float maxProb = 0.0f;
    for (int i = 0; i < m_docCatList.Length; i++)
    {
        float prob = this->GetCatgoryProbInDoc(text, m_docCatList.Data[i]);
        if (prob > maxProb)
        {
            maxProb = prob;
            bestCat = m_docCatList.Data[i];
        }
    }

    return bestCat;
}

bool LDocClassifier::GetFeatures(const string& text, set<string>& featuresSet)
{
    featuresSet.clear();

    // 对文本进行分割
    LStringList featureList;
    StringSplit(' ', text.c_str(), featureList);

    // 消除重复单词
    for (int i = 0; i < featureList.Length; i++)
    {
        if (featureList.Data[i].empty())
            continue;

        featuresSet.insert(featureList.Data[i]);
    }

    return true;
}

float LDocClassifier::GetFeatureProbInCat(const string& feature, LDOC_CATEGORY cat)
{
    if (m_docCategoryTotal.GetCount(cat) == 0)
        return 0.0f;

    if (m_featureMap.count(feature) == 0)
        return 0.0f;

    int featureCount = m_featureMap[feature].GetCategoryCount(cat);
    int categoryCount = m_docCategoryTotal.GetCount(cat);
    return (float)featureCount/(float)categoryCount;
   
}

float LDocClassifier::GetFeatureWeightedProbInCat(const string& feature, LDOC_CATEGORY cat)
{
    float basicProb = this->GetFeatureProbInCat(feature, cat);

    int featureTotalCount = m_featureMap[feature].GetCategoryTotalCount();
   
    // w = 0.5 + totalCount/(1 + totalCount) * (basicProb - 0.5)
    float weightedProb = ((1.0f * 0.5f) + (float)featureTotalCount * basicProb)/(1.0f + (float)featureTotalCount);
    return weightedProb;
}

LNaiveBayesClassifier::LNaiveBayesClassifier()
{
    
}

LNaiveBayesClassifier::~LNaiveBayesClassifier()
{

}

float LNaiveBayesClassifier::GetDocProbInCat(const string& text, LDOC_CATEGORY cat)
{
    set<string> featureSet;
    this->GetFeatures(text, featureSet);
    if (featureSet.size() == 0)
        return 0.0f;

    float prob = 1.0f;
    for (auto iter = featureSet.begin(); iter != featureSet.end(); iter++)
    {
        prob *= this->GetFeatureWeightedProbInCat(*iter, cat);
    }

    return prob;
}

float LNaiveBayesClassifier::GetCatgoryProbInDoc(const string& text, LDOC_CATEGORY cat)
{
    float catProb = (float)m_docCategoryTotal.GetCount(cat)/(float)m_docCategoryTotal.GetTotalCount();
    float docProb = this->GetDocProbInCat(text, cat);
    return docProb * catProb;
}

LFisherClassifier::LFisherClassifier()
{

}

LFisherClassifier::~LFisherClassifier()
{

}

float LFisherClassifier::GetCatgoryProbInFea(const string& feature, LDOC_CATEGORY cat)
{
    // 特征在该分类中出现的概率
    float featureProb = this->GetFeatureProbInCat(feature, cat);
    if (featureProb == 0.0f)
        return 0.0f;

    // 特征在所有分类中的概率和
    float featureTotalProb = 0.0f;
    for (int i = 0; i < m_docCatList.Length; i++)
    {
        float prob = this->GetFeatureProbInCat(feature, m_docCatList.Data[i]);
        featureTotalProb += prob;
    }

    float catProb = featureProb/featureTotalProb;

    return catProb;
}

float LFisherClassifier::GetCatgoryWeightProbInFea(const string& feature, LDOC_CATEGORY cat)
{
    float basicProb = this->GetCatgoryProbInFea(feature, cat);

    int featureTotalCount = m_featureMap[feature].GetCategoryTotalCount();
    float weightedProb = ((1.0f * 0.5f) + (float)featureTotalCount * basicProb)/(1.0f + (float)featureTotalCount);
    return weightedProb;
}

float LFisherClassifier::GetCatgoryProbInDoc(const string& text, LDOC_CATEGORY cat)
{
    set<string> featureSet;
    this->GetFeatures(text, featureSet);
    if (featureSet.size() == 0)
        return 0.0f;

    float prob = 1.0f;
    for (auto iter = featureSet.begin(); iter != featureSet.end(); iter++)
    {
        prob *= this->GetCatgoryWeightProbInFea(*iter, cat);
    }

    float score = -2 * log(prob);

    return this->Inchi2(score, featureSet.size() * 2);
}

float LFisherClassifier::Inchi2(float chi, int df)
{
    float m = chi/2.0f;
    float term = exp(-m);
    float sum = term;
    for (int i = 1; i < df/2; i++)
    {
        term *= m/i;
        sum += term;
    }

    if (sum >= 1.0f)
        return 1.0f;

    return sum;

}