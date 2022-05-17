/// Author:Burnell_Liu Email:674288799@qq.com Date: 2014/08/27
/// Description: 优化算法
/// 
/// 遗传算法优化
/// 爬山算法优化
/// 模拟退火算法优化
///
/// 应用优化算法核心:1.问题的解可以转化为数字列表 2. 对于解而言, 最优解应该接近于其他次优解, 否则没法优化
/// Others: 
/// Function List: 
///
/// History: 
///  1. Date, Author 
///  Modification
///


#ifndef _LOPTIMIZATION_H_
#define _LOPTIMIZATION_H_

#include "LDataStruct/include/LArray.h"

#ifndef IN
#define IN
#endif

#ifndef INOUT
#define INOUT
#endif

#ifndef OUT
#define OUT
#endif

/// @brief 基因范围结构
struct LOGeneDomain
{
    int Min; ///< 最小值
    int Max; ///< 最大值
};

typedef LArray<int> LOGenome; ///< 基因组
typedef LArray<LOGeneDomain>  LOGenomeDomain; ///< 基因组范围
typedef LArray<int> LGOCrossOverSplitPointList; ///< 交叉分割点列表

/// @brief 解结构
struct LOSolution
{
    static const int MAX_COST; ///< 最大成本

    LOGenome* PGenome; ///< 基因组
    int Cost; ///< 解成本
    
    /// @brief 重载<操作符, 使该结构可被排序函数使用
    bool operator < (const LOSolution& other)
    {
        return this->Cost < other.Cost;
    }
};

/// @brief 解变异类
class LOSolutionMutate
{

public:
    LOSolutionMutate();
    virtual ~LOSolutionMutate();

    /// @brief 设置概率
    /// @param[in] prob 变异概率, 范围为0~1的浮点数
    /// @return 参数错误返回false
    bool SetProb(IN float prob);

    /// @brief 设置步长
    /// @param[in] step 变异步长, 步长为大于等于1的整数
    /// @return 参数错误返回false
    bool SetStep(IN int step);

    /// @brief 变异
    /// @param[in] genomeDomain 基因组范围
    /// @param[in] solution 需要编译的解
    /// @return 参数错误返回false
    bool Mutate(IN const LOGenomeDomain& genomeDomain, INOUT LOSolution& solution) const;

private:
    const static int ERROR_STEP; ///< 错误步长
    const static float ERROR_PROB; ///< 错误概率
private:
    int m_step; ///< 变异步长
    float m_prob; ///< 变异概率
private:
    // 禁止默认赋值操作符和拷贝构造函数
    LOSolutionMutate(const LOSolutionMutate&);
    LOSolutionMutate& operator = (const LOSolutionMutate&);
};

/// @解交叉类
class LOSolutionCrossOver
{
public:
    LOSolutionCrossOver();
    virtual ~LOSolutionCrossOver();

    /// @brief 设置交叉切割点列表
    /// 
    /// 请在使用Init()方法前,设置变异步长
    /// @param[in] splitPointList 切割点列表
    /// @return true
    bool SetSplitPointList(IN const LGOCrossOverSplitPointList& splitPointList);

    /// @brief 对两个解进行交叉产生两个新的解
    /// @return true
    bool CrossOver(IN const LOSolution& solutionOld1, IN const LOSolution& solutionOld2, 
        OUT LOSolution& solutionNew1, OUT LOSolution& solutionNew2) const;

private:
    LGOCrossOverSplitPointList m_splitPointList; ///< 交叉切割点列表
private:
    // 禁止默认赋值操作符和拷贝构造函数
    LOSolutionCrossOver(const LOSolutionCrossOver&);
    LOSolutionCrossOver& operator = (const LOSolutionCrossOver&);
};

/// @brief 成本函数接口
class LOCostFunInterface
{
public:
    virtual ~LOCostFunInterface(){};

    /// @brief 计算基因组的成本
    ///
    /// 越差的基因组, 返回的成本越高
    /// @param[in] genome 基因组
    /// @return 成本
    virtual int CalculateGenomeCost(IN LOGenome& genome) = 0;
};

/// @brief 遗传算法优化抽象基类
class LGeneticOptimize
{
public:
    LGeneticOptimize();
    virtual ~LGeneticOptimize();

public:
    /// @brief 设置种群大小
    ///
    /// 请在使用Init()方法前,设置种群大小
    /// @param[in] popSize 种群大小, 范围为大于0的整数, 默认值(200)
    /// @return 参数错误返回false
    bool SetPopSize(IN int popSize);

    /// @brief 设置每一代直接胜出者的比例
    ///
    /// 请在使用Init()方法前,设置胜出者比例
    /// @param[in] elitePer 胜出者比例, 范围为0~1的浮点数, 默认值(0.2)
    /// @return 参数错误返回false
    bool SetElitePercent(IN float elitePer);

    /// @brief 设置变异概率
    ///
    /// 请在使用Init()方法前,设置变异概率
    /// @param[in] mutateProb 变异概率, 范围为0~1的浮点数, 默认值(0.02)
    /// @return 参数错误返回false
    bool SetMutateProb(IN float mutateProb);

    /// @brief 设置变异步长
    ///
    /// 请在使用Init()方法前,设置变异步长
    /// @param[in] mutateStep 变异步长, 步长为大于等于1的整数, 默认值(1)
    /// @return 参数错误返回false
    bool SetMutateStep(IN int mutateStep);

    /// @brief 设置交叉切割点列表
    /// 
    /// 请在使用Init()方法前,设置变异步长
    /// @param[in] splitPointList 切割点列表
    /// @return true
    bool SetCrossOverSplitPointList(IN const LGOCrossOverSplitPointList& splitPointList);

    /// @brief 初始化
    /// @param[in] pCostFun 成本函数
    /// @param[in] genomeDomain 基因组范围
   /// @return 参数错误返回false
    virtual bool Init(IN LOCostFunInterface* pCostFun, IN const LOGenomeDomain& genomeDomain) = 0;

protected:
    /// @brief 初始种群
    /// @return true
    bool InitPopulation();

    /// @brief 种群竞争
    /// @return true
    bool PopulationCompete();

    /// @brief 赌轮选择法
    /// @return 个体索引
    int RouletteWheelSelection();

    /// @brief 清理资源
    /// @return true
    bool CleanUp();

protected:
    typedef LArray<LOSolution> LSolutionList; // 解列表

    LOSolutionMutate m_solutionMutate; ///< 解变异
    LOSolutionCrossOver m_solutionCrossOver; ///< 解交叉

    float m_elitePer; ///< 胜出者比例
    int m_popSize; ///< 种群大小

    int m_genomeLength; ///< 基因组长度
    LSolutionList m_solutionList; ///< 解列表(种群)
    LSolutionList m_solutionListCopy; ///< 解列表副本
    LOGenomeDomain m_genomeDomain; ///< 基因组范围

    LOCostFunInterface* m_pCostFun; ///< 成本函数接口指针

    bool m_bInitSuccess; ///< 初始化成功

private:
    // 禁止默认赋值操作符和拷贝构造函数
    LGeneticOptimize(const LGeneticOptimize&);
    LGeneticOptimize& operator = (const LGeneticOptimize&);
};

/// @brief 开放式遗传算法
///
/// 开放式遗传算法对成本函数接口指针不做要求
/// 如果成本函数接口指针为空时, 那么请在Breed()之前使用GetSolution(int)手动设置成本
class LOpenedGenetic : public LGeneticOptimize
{
public:
    /// @brief 初始化
    /// @param[in] pCostFun 成本函数接口指针, 可以为空
    /// @param[in] genomeDomain 基因组范围, 基因组范围的每个基因都必需Min小于等于Max
    /// @return 参数错误返回false
    virtual bool Init(IN LOCostFunInterface* pCostFun, IN const LOGenomeDomain& genomeDomain);

    /// @brief 种群演化一代
    /// @return 初始化错误返回false
    bool Breed();

    /// @brief 获取种群中的一个解
    /// @param[in] 解的索引, 请注意索引的范围(0 ~ 种群大小-1), 该函数不对索引范围进行检查
    /// @return 解的引用
    LOSolution& GetSolution(IN int index);

};

/// @brief 封闭式遗传算法
class LClosedGenetic : public LGeneticOptimize
{
public:
    /// @brief 初始化
    /// @param[in] pCostFun 成本函数接口指针, 不可以为空
    /// @param[in] genomeDomain 基因组范围, 基因组范围的每个基因都必需Min小于等于Max
    /// @return 参数错误返回false
    virtual bool Init(IN LOCostFunInterface* pCostFun, IN const LOGenomeDomain& genomeDomain);

    /// @brief 种群演化多代
    ///
    /// 该函数可能很耗时(受数据复杂度以及成本函数效率的影响)
    /// @param[in] times 演化代数, 演化代数为大于等于1的整数
    /// @param[out] bestSolution 最优解, 请分配最优解需要的空间
    /// @return 参数错误或者初始化错误返回false
    bool BreedEx(IN int times, OUT LOSolution&bestSolution);
};



/// @brief 优化算法抽象基类
class LOptimize
{
public:
    LOptimize();
    virtual ~LOptimize();
public:
    /// @brief 设置步长
    ///
    /// 请在使用Init()方法前,设置步长
    /// 步长和算法搜索解空间的速度成反比, 但是大的步长可能会跳过最优解
    /// @param[in] mutateStep 步长, 步长为大于等于1的整数, 默认值(1) 
    /// @return 参数错误返回false
    bool SetStep(IN int step);

    /// @brief 初始化
    /// @param[in] pCostFun 成本函数接口, 成本函数接口指针不能为空
    /// @param[in] genomeDomain 基因组范围, 基因组范围的每个基因的必需Min小于等于Max
    /// @return 参数错误返回false
    bool Init(IN LOCostFunInterface* pCostFun, IN const LOGenomeDomain& genomeDomain);

    /// @brief 搜索最优解
    ///
    /// 该函数可能很耗时(受数据复杂度以及成本函数效率的影响)
    /// @param[out] bestSolution 最优解,  请分配最佳优需要的空间, 
    /// @return 参数错误或者初始化错误返回false
    virtual bool Search(INOUT LOSolution& bestSolution) = 0; 

protected:
    int m_step; ///< 步长
    int m_genomeLength; ///< 基因组长度
    LOGenomeDomain m_genomeDomain; ///< 基因组范围
    LOCostFunInterface* m_pCostFun; ///< 成本函数接口指针

    bool m_bInitSuccess; ///< 初始化成功

private:
    // 禁止默认赋值操作符和拷贝构造函数
    LOptimize(const LOptimize&);
    LOptimize& operator = (const LOptimize&);
};

/// @brief 爬山法优化
class LClimbHillOptimize : public LOptimize
{
public:
    /// @brief 搜索最优解
    ///
    /// 该函数可能很耗时(受数据复杂度以及成本函数效率的影响)
    /// @param[out] bestSolution 最优解,  请分配最优解需要的空间, 
    /// @return 参数错误或者初始化错误返回false
    virtual bool Search(OUT LOSolution& bestSolution); 

    /// @brief 多次爬山搜索最优解
    ///
    /// 该函数可能很耗时(受数据复杂度以及成本函数效率的影响)
    /// @param[in] times 搜索次数, 次数为大于等于1的整数
    /// @param[out] bestSolution 最优解, 请分配最优解需要的空间
    /// @return 参数错误或者初始化错误返回false
    bool SearchEx(IN int times, OUT LOSolution&bestSolution);
};

/// @模拟退火算法优化
class LAnnealingOptimize : public LOptimize
{
public:
    LAnnealingOptimize();
    virtual ~LAnnealingOptimize();

    /// @brief 设置初始温度
    /// @param[in] startTemp 初始温度, 要求大于0.1的浮点数数, 默认值(10000.0)
    /// @return 参数错误返回false
    bool SetStartTemperature(IN float startTemp);

    /// @brief 设置温度冷却速度
    ///
    /// 冷却速度越大, 温度降低越快
    /// @param[in] coolSpeed 冷却速度, 要求大于0小于1的浮点数, 默认值(0.05)
    /// @return 参数错误返回false
    bool SetCoolSpeed(IN float coolSpeed);

    /// @brief 搜索最优解
    ///
    /// 该函数可能很耗时(受数据复杂度以及成本函数效率的影响)
    /// @param[out] bestSolution 最优解,  请分配最优解需要的空间, 
    /// @return 参数错误或者初始化错误返回false
    virtual bool Search(OUT LOSolution& bestSolution); 

private:
    float m_coolSpeed; ///< 温度的冷却速度
    float m_startTemp; ///< 初始温度
private:
    // 禁止默认赋值操作符和拷贝构造函数
    LAnnealingOptimize(const LAnnealingOptimize&);
    LAnnealingOptimize& operator = (const LAnnealingOptimize&);
};

#endif