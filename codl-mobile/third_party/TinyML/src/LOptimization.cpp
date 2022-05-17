

#include "LOptimization.h"

#include <math.h>

#include <algorithm>
#include <list>
using std::list;

#include "LLib.h"

const int LOSolution::MAX_COST = (unsigned int)(~0) >> 1;
const int LOSolutionMutate::ERROR_STEP = 0;
const float LOSolutionMutate::ERROR_PROB = 0.0f;

LOSolutionMutate::LOSolutionMutate()
{
    m_step = ERROR_STEP;
    m_prob = ERROR_PROB;
}

LOSolutionMutate::~LOSolutionMutate()
{

}

bool LOSolutionMutate::SetProb(IN float prob)
{
    if (prob < 0.0f || prob > 1.0f)
        return false;

    m_prob = prob;

    return true;
}

bool LOSolutionMutate::SetStep(IN int step)
{
    if (step < 1)
        return false;

    return true;
}

bool LOSolutionMutate::Mutate(IN const LOGenomeDomain& genomeDomain, INOUT LOSolution& solution) const
{
    // 检查参数
    if (m_prob == ERROR_PROB || m_step == ERROR_STEP)
        return false;

    if (solution.PGenome == NULL)
        return false;

    if (genomeDomain.Length != solution.PGenome->Length)
        return false;

    for (int i = 0; i < genomeDomain.Length; i++)
    {
        float prob = LRandom::RandFloat();
        if (prob < m_prob)
        {
            if ((solution.PGenome->Data[i] + m_step) <= 
                genomeDomain.Data[i].Max)
                solution.PGenome->Data[i] += m_step;
        }

        if (prob >= m_prob && prob < 2 * m_prob)
        {
            if ((solution.PGenome->Data[i] - m_step) >= 
                genomeDomain.Data[i].Min)
                solution.PGenome->Data[i] -= m_step;
        }
    }

    solution.Cost = LOSolution::MAX_COST;

    return true;
}

LOSolutionCrossOver::LOSolutionCrossOver()
{

}

LOSolutionCrossOver::~LOSolutionCrossOver()
{

}

bool LOSolutionCrossOver::SetSplitPointList(IN const LGOCrossOverSplitPointList& splitPointList)
{
    m_splitPointList = splitPointList;
    return true;
}

bool LOSolutionCrossOver::CrossOver(
    IN const LOSolution& solutionOld1, 
    IN const LOSolution& solutionOld2, 
    OUT LOSolution& solutionNew1, 
    OUT LOSolution& solutionNew2) const
{
    int genomeLength = solutionOld1.PGenome->Length;

    for (int i = 0; i < genomeLength; i++)
    {
        solutionNew1.PGenome->Data[i] = solutionOld1.PGenome->Data[i];
        solutionNew2.PGenome->Data[i] = solutionOld2.PGenome->Data[i];
    }
    solutionNew1.Cost = LOSolution::MAX_COST;
    solutionNew2.Cost = LOSolution::MAX_COST;
    int geneIndex;
    if (m_splitPointList.Data != NULL)
    {
        int randNum = LRandom::RandInt(0, m_splitPointList.Length-1);
        geneIndex = m_splitPointList[randNum];
    }
    else
    {
        geneIndex = LRandom::RandInt(0, genomeLength-1);
    }

    for (int i = 0; i <= geneIndex; i++)
    {
        solutionNew1.PGenome->Data[i] = solutionOld2.PGenome->Data[i];
        solutionNew2.PGenome->Data[i] = solutionOld1.PGenome->Data[i];
    }

    return true;
}
LGeneticOptimize::LGeneticOptimize()
{
	m_pCostFun = NULL;

	m_popSize = 200;
	m_elitePer = 0.2f;

	m_genomeLength = 0;

    m_bInitSuccess = false;

    m_solutionMutate.SetProb(0.02f);
    m_solutionMutate.SetStep(1);
};

LGeneticOptimize::~LGeneticOptimize()
{
	CleanUp();
}

bool LGeneticOptimize::SetPopSize(IN int popSize)
{
    if (popSize <= 0)
        return false;

	m_popSize = popSize;

    return true;
}

bool LGeneticOptimize::SetElitePercent(IN float elitePer)
{
    if (elitePer <= 0.0f || elitePer >= 1.0f)
        return false;

	m_elitePer = elitePer;

    return true;
}

bool LGeneticOptimize::SetMutateProb(IN float mutateProb)
{
    if (mutateProb <= 0.0f || mutateProb >= 1.0f)
        return false;

    return m_solutionMutate.SetProb(mutateProb);
}

bool LGeneticOptimize::SetMutateStep(IN int mutateStep)
{
    if (mutateStep < 1)
        return false;

    return m_solutionMutate.SetStep(mutateStep);
}

bool LGeneticOptimize::SetCrossOverSplitPointList(IN const LGOCrossOverSplitPointList& splitPointList)
{
    return m_solutionCrossOver.SetSplitPointList(splitPointList);
}



bool LGeneticOptimize::InitPopulation()
{
    LRandom::SRandTime();

    // 构造初始化种群
    m_solutionList.Reset(m_popSize);
    m_solutionListCopy.Reset(m_popSize);
    for (int i = 0; i < m_popSize; i++)
    {
        LOSolution& solution = m_solutionList.Data[i]; // 基因组
        solution.Cost = LOSolution::MAX_COST;
        solution.PGenome = new LOGenome;
        solution.PGenome->Reset(m_genomeLength);

        // 随机构造一个基因组
        for (int j = 0; j < m_genomeLength; j++)
        {
            int min = m_genomeDomain.Data[j].Min;
            int max = m_genomeDomain.Data[j].Max;
            int gene = LRandom::RandInt(min, max);
            solution.PGenome->Data[j] = gene;
        }

        m_solutionListCopy.Data[i].PGenome = new LOGenome;
        m_solutionListCopy.Data[i].PGenome->Reset(m_genomeLength);
    }

    return true;
}

bool LGeneticOptimize::PopulationCompete()
{
    // 计算所有种群成员的成本
    if (m_pCostFun != NULL)
    {
        for (int i = 0; i < m_popSize; i ++)
        {
            if (m_solutionList.Data[i].Cost == LOSolution::MAX_COST)
            {
                LOSolution& solution = m_solutionList.Data[i];
                solution.Cost = m_pCostFun->CalculateGenomeCost(*(solution.PGenome));
            }
        }
    }

    std::sort(&m_solutionList.Data[0], &m_solutionList.Data[m_popSize]);// 按升序排序

    // 临时种群, 从种群中复制
    for (int i = 0; i < m_popSize; i++)
    {
        m_solutionListCopy.Data[i].Cost = m_solutionList.Data[i].Cost;
        (*m_solutionListCopy.Data[i].PGenome) = (*m_solutionList.Data[i].PGenome);
    }

    int eliteNum = (int)(m_elitePer * m_popSize); // 直接胜出者数目

    // 0~eliteNum保留之前的基因组
    // elitNum~m_popsize * 0.8使用交叉来得到基因组
    for (int i = eliteNum; i < m_popSize * 0.8; i += 2)
    {
        // 使用赌轮法选取两个个体进行交叉
        int mutateIndex1 = RouletteWheelSelection(); 
        int mutateIndex2 = RouletteWheelSelection();

        m_solutionCrossOver.CrossOver(m_solutionListCopy.Data[mutateIndex1], m_solutionListCopy.Data[mutateIndex2],
            m_solutionList.Data[i], m_solutionList.Data[i + 1]);

        // 进行变异
        m_solutionMutate.Mutate(m_genomeDomain, m_solutionList.Data[i]);
        m_solutionMutate.Mutate(m_genomeDomain, m_solutionList.Data[i + 1]);
    }

    // m_popSize * 0.8~m_popSize
    for (int i = (int)(m_popSize * 0.8); i < m_popSize; i++)
    {
        int mutateIndex = RouletteWheelSelection();
        LOSolution& solution = m_solutionListCopy.Data[mutateIndex]; // 基因组

        (*m_solutionList.Data[i].PGenome) = (*solution.PGenome);
        m_solutionList.Data[i].Cost = LOSolution::MAX_COST;

        m_solutionMutate.Mutate(m_genomeDomain, m_solutionList.Data[i]);
    }

    return true;
}

int LGeneticOptimize::RouletteWheelSelection()
{
	// 将种群分为4等分, 根据2/8原则使用赌轮法来选择个体
	int subLen = m_popSize/4;
	int selectIndex = LRandom::RandInt(0, subLen - 1);;
	if (LRandom::RandFloat() < 0.8)
	{
		if (LRandom::RandFloat() < 0.8)
			selectIndex += 0 * subLen;
		else
			selectIndex += 1 * subLen;
	}
	else
	{
		if (LRandom::RandFloat() < 0.8)
			selectIndex += 2 * subLen;
		else
			selectIndex += 3 * subLen;
	}

	return selectIndex;
}

bool LGeneticOptimize::CleanUp()
{

	for (int i = 0; i < m_solutionList.Length; i++)
	{
		LOSolution& solution = m_solutionList.Data[i];
        LDestroy::SafeDelete(solution.PGenome);
	}

	for (int i = 0; i < m_solutionListCopy.Length; i++)
	{
		LOSolution& solution = m_solutionListCopy.Data[i]; 
		LDestroy::SafeDelete(solution.PGenome);
	}

    return true;
}

bool LOpenedGenetic::Init(IN LOCostFunInterface* pCostFun, IN const LOGenomeDomain& genomeDomain)
{
    // 检查输入参数
    m_bInitSuccess = false;
    if (genomeDomain.Length == 0)
        return false;
    for (int i = 0; i < genomeDomain.Length; i++)
    {
        if (genomeDomain.Data[i].Min > genomeDomain.Data[i].Max)
            return false;
    }

    m_pCostFun = pCostFun;

     // 初始化基因组范围
    m_genomeDomain = genomeDomain;
    m_genomeLength = genomeDomain.Length;

    this->CleanUp();

    this->InitPopulation();

    m_bInitSuccess = true;

    return true;
}

bool LOpenedGenetic::Breed()
{
    if (m_bInitSuccess == false)
        return false;

    PopulationCompete();

    return true;
}

LOSolution& LOpenedGenetic::GetSolution(IN int index)
{
    return m_solutionList.Data[index];
}

bool LClosedGenetic::Init(IN LOCostFunInterface* pCostFun, IN const LOGenomeDomain& genomeDomain)
{
    // 检查输入参数
    m_bInitSuccess = false;
    if (pCostFun == NULL || genomeDomain.Length == 0)
        return false;
    for (int i = 0; i < genomeDomain.Length; i++)
    {
        if (genomeDomain.Data[i].Min > genomeDomain.Data[i].Max)
            return false;
    }

    m_pCostFun = pCostFun;

    // 初始化基因组范围
    m_genomeDomain = genomeDomain;
    m_genomeLength = genomeDomain.Length;

    this->CleanUp();

    this->InitPopulation();

    m_bInitSuccess = true;

    return true;
}

bool LClosedGenetic::BreedEx(IN int times, OUT LOSolution&bestSolution)
{
    // 检查输入参数
    if (times <= 0 || bestSolution.PGenome == NULL)
        return false;
    if (m_bInitSuccess == false)
        return false;

    for (int i = 0; i < times; i++)
        PopulationCompete();

    bestSolution.Cost = m_solutionList.Data[0].Cost;
    (*bestSolution.PGenome) = (*m_solutionList.Data[0].PGenome);
    return true;
}

LOptimize::LOptimize()
{
    m_pCostFun = NULL;
    m_genomeLength = 0;
    m_step = 1;
    m_bInitSuccess = false;
}

LOptimize::~LOptimize()
{
    m_pCostFun = NULL;
}

bool LOptimize::SetStep(IN int step)
{
    if (step < 1)
        return false;

    m_step = step;

    return true;
}

bool LOptimize::Init(IN LOCostFunInterface* pCostFun, IN const LOGenomeDomain& genomeDomain)
{
    // 检查输入参数
    m_bInitSuccess = false;
    if (pCostFun == NULL || genomeDomain.Length == 0)
        return false;
    for (int i = 0; i < genomeDomain.Length; i++)
    {
        if (genomeDomain.Data[i].Min > genomeDomain.Data[i].Max)
            return false;
    }

    m_pCostFun = pCostFun;
    m_genomeDomain = genomeDomain;
    m_genomeLength = genomeDomain.Length;
    LRandom::SRandTime();

    m_bInitSuccess = true;
    return true;
}

bool LClimbHillOptimize::Search(OUT LOSolution& bestSolution)
{
    // 检查输入参数
    if (bestSolution.PGenome == NULL)
        return false;
    if (m_bInitSuccess == false)
        return false;


    // 创建一个随机基因组
    int bestCost = LOSolution::MAX_COST;
    LOGenome bestGenome(m_genomeLength);
    for (int i = 0; i < m_genomeLength; i++)
    {
        int min = m_genomeDomain.Data[i].Min;
        int max = m_genomeDomain.Data[i].Max;
        int gene = LRandom::RandInt(min, max);
        bestGenome.Data[i] = gene;
    }

    list<LOGenome> nearbyGenomeList; // 附近基因组列表

    while (true)
    {
        // 创建附近基因组列表
        nearbyGenomeList.clear();
        LOGenome nearbyGenome(m_genomeLength);
        for (int i = 0; i < m_genomeLength; i++)
        {
            nearbyGenome = bestGenome;
            nearbyGenome.Data[i] = bestGenome.Data[i] - m_step;

            if (nearbyGenome.Data[i] >= m_genomeDomain.Data[i].Min)
                 nearbyGenomeList.push_back(nearbyGenome);

            nearbyGenome.Data[i] = bestGenome.Data[i] + m_step;
            if (nearbyGenome.Data[i] <= m_genomeDomain.Data[i].Max)
                nearbyGenomeList.push_back(nearbyGenome);
        }

        // 在附近解中寻找最优解
        int currentCost = m_pCostFun->CalculateGenomeCost(bestGenome);
        bestCost = currentCost;

        for (auto iter = nearbyGenomeList.begin(); iter != nearbyGenomeList.end(); iter++)
        {
            LOGenome& genome = *iter;
            int cost = m_pCostFun->CalculateGenomeCost(genome);
            if (cost < bestCost)
            {
                bestCost = cost;
                bestGenome = genome; 
            }
        }

        if (bestCost == currentCost)
            break;

    }

    (*bestSolution.PGenome) = bestGenome;
    bestSolution.Cost = bestCost;

    return true;
}

bool LClimbHillOptimize::SearchEx(IN int times, OUT LOSolution&bestSolution)
{
    // 检查输入参数
    if (times <= 0 || bestSolution.PGenome == NULL)
        return false;
    if (m_bInitSuccess == false)
        return false;

    bestSolution.Cost = LOSolution::MAX_COST;

    LOSolution solution;
    solution.Cost = LOSolution::MAX_COST;
    solution.PGenome = new LOGenome;

    while (times > 0)
    {
        Search(solution);
        if (solution.Cost < bestSolution.Cost)
        {
            bestSolution.Cost = solution.Cost;
            (*bestSolution.PGenome) = (*solution.PGenome);
        }
        times--;
    }

    LDestroy::SafeDelete(solution.PGenome);

    return true;
}

LAnnealingOptimize::LAnnealingOptimize()
{
    m_startTemp = 10000.0f;
    m_coolSpeed = 0.05f;
}

LAnnealingOptimize::~LAnnealingOptimize()
{

}

bool LAnnealingOptimize::SetStartTemperature(IN float startTemp)
{
    if (startTemp <= 0.1f)
        return false;

    m_startTemp = startTemp;

    return true;
}

bool LAnnealingOptimize::SetCoolSpeed(IN float coolSpeed)
{
    if (coolSpeed <= 0.0f || coolSpeed >= 1.0f)
        return false;

    m_coolSpeed = coolSpeed;
    return true;
}

bool LAnnealingOptimize::Search(OUT LOSolution& bestSolution)
{
    // 检查参数
    if (bestSolution.PGenome == NULL)
        return false;
    if (m_bInitSuccess == false)
        return false;

    // 创建一个随机基因组
    int bestCost = LOSolution::MAX_COST;
    LOGenome bestGenome(m_genomeLength);
    for (int i = 0; i < m_genomeLength; i++)
    {
        int min = m_genomeDomain.Data[i].Min;
        int max = m_genomeDomain.Data[i].Max;
        int gene = LRandom::RandInt(min, max);
        bestGenome.Data[i] = gene;
    }
     bestCost = m_pCostFun->CalculateGenomeCost(bestGenome);

    LOGenome nearbyGenome(m_genomeLength);
    nearbyGenome = bestGenome;
    for (float temp = m_startTemp; temp > 0.1f; temp = temp * (1.0f - m_coolSpeed))
    {
        for (int i = 0; i < m_genomeLength; i++)
        {
            int dir = LRandom::RandInt(0, 1);
            if (dir == 1)
                nearbyGenome.Data[i] = bestGenome.Data[i] + m_step;
            else
                nearbyGenome.Data[i] = bestGenome.Data[i] - m_step;

            if (nearbyGenome.Data[i] > m_genomeDomain.Data[i].Max)
                nearbyGenome.Data[i] = m_genomeDomain.Data[i].Max;
            if (nearbyGenome.Data[i] < m_genomeDomain.Data[i].Min)
                nearbyGenome.Data[i] = m_genomeDomain.Data[i].Min;

            int nearbyCost = m_pCostFun->CalculateGenomeCost(nearbyGenome);

            float perRec = exp(-(nearbyCost - bestCost)/temp);
            if (nearbyCost < bestCost || LRandom::RandFloat() < perRec)
            {
                bestGenome.Data[i] = nearbyGenome.Data[i];
                bestCost = nearbyCost;
            }  

            nearbyGenome.Data[i] = bestGenome.Data[i];
        }
    }

    bestSolution.Cost = bestCost;
    (*bestSolution.PGenome) = bestGenome;

    return true;
}