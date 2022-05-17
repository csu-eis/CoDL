
##多分类问题##
多分类问题在生活中很常见。例如，音乐可以被分类为民谣，古典，金属，流行等等。[上一章](http://www.coderjie.com/blog/604c0804dbeb11e7841d00163e0c0e36)介绍的二项逻辑回归可以很好的解决二分类问题，针对多分类问题二项逻辑回归就不太适用，Softmax回归可以很好的解决多分类问题。

##Softmax回归模型##
设特征数量为 $n$ 即，特征为 $x$，特征的参数为 $\theta$ ，我们定义如下：

\begin{align}
\\\
x & =
\begin{bmatrix}
1 \ x\_1 \ \cdots \ x\_n
\end{bmatrix}
\\\
\\\
\theta & =
\begin{bmatrix}
\theta\_0 \\\\
\theta\_1 \\\\
\vdots    \\\\
\theta\_n
\end{bmatrix}
\\\
\\\
x\theta & = \theta\_0 + \theta\_1x\_1 + \theta\_2x\_2 + \cdots + \theta\_nx\_n \\\\
\\\
\end{align}

在在多分类问题中 $y$ 可以取 $k$ 个不同的值，也就是 $y\in\\{0,1,2,...,k-1 \\}$ 。因为每个类别都需要一个特征参数 $\theta$ ，所以有：

\begin{align}
\\\
\Theta & =
\begin{bmatrix}
\theta^{0} \ \theta^{1} \ \cdots \ \theta^{k-1}
\end{bmatrix}
\\\
\end{align}

针对多分类问题，我们的概率函数应该给出每一种分类结果的概率，所以我们的概率函数应该输出一个 $k$ 维的向量，我们定义概率函数如下：

\begin{align}
\\\
h\_\Theta(x) & = 
\begin{bmatrix}
p(y=0|x;\Theta) \\\\
p(y=1|x;\Theta) \\\\
\vdots    \\\\
p(y=k-1|x;\Theta)
\end{bmatrix}
= \frac{1}{\sum\_{l=0}^{k-1}e^{x\theta^l}}
\begin{bmatrix}
e^{x\theta^0} \\\\
e^{x\theta^1} \\\\
\vdots    \\\\
e^{x\theta^{k-1}}
\end{bmatrix}
\\\
\end{align}

注意 $\frac{1}{\sum\_{l=0}^{k-1}e^{x\theta^l}}$ 这一项为对概率分布进行归一化，使得所有概率和为1。

设训练样本数为 $m$，训练样本集为 $X$ ，训练输出集为 $Y$ ，如下：
\begin{align}
X & =
\begin{bmatrix}
x^{0}  \\\\
x^{1}  \\\\
\cdots \\\\
x^{m-1}
\end{bmatrix}
\\\
\\\
Y & =
\begin{bmatrix}
y^{0}       \\\\
y^{1}       \\\\
\vdots        \\\\
y^{m-1}
\end{bmatrix}
\\\
\end{align}

我们的目标是已知 $X$ 和 $Y$ 的情况下得到最优的 $\Theta$。

##似然函数##
哪个 $\Theta$ 是最优的？我们需要先定义似然函数：

\begin{align}
\\\
L(\Theta) &= \prod\_{i=0}^{m-1} p(y^i \mid x^i)
\\\
\\\
L(\Theta) &= \prod\_{i=0}^{m-1} \sum\_{j=0}^{k-1}1 \\{y^i=j\\} \frac{e^{x^i\theta^j}}{\sum\_{l=0}^{k-1}e^{x^i\theta^l}}
\\\
\end{align}

上面的公式中 $1\\{ \cdot \\}$ 是示性函数，其取值规则为：

$$ 1\\{ 值为真的表达式 \\} = 1 $$ 

$$ 1\\{ 值为假的表达式 \\} = 0 $$ 

我们在似然函数中引入自然对数以方便后续的求导，则：

\begin{align}
\\\
L(\Theta) &= \log(\prod\_{i=0}^{m-1} \sum\_{j=0}^{k-1}1 \\{y^i=j\\} \frac{e^{x^i\theta^j}}{\sum\_{l=0}^{k-1}e^{x^i\theta^l}})
\\\
\\\
L(\Theta) &= \sum\_{i=0}^{m-1}\log(\sum\_{j=0}^{k-1}1 \\{y^i=j\\} \frac{e^{x^i\theta^j}}{\sum\_{l=0}^{k-1}e^{x^i\theta^l}})
\\\
\\\
L(\Theta) &= \sum\_{i=0}^{m-1}\sum\_{j=0}^{k-1}1 \\{y^i=j\\} \log(\frac{e^{x^i\theta^j}}{\sum\_{l=0}^{k-1}e^{x^i\theta^l}})
\\\
\\\
L(\Theta) &= \sum\_{i=0}^{m-1}\sum\_{j=0}^{k-1}1 \\{y^i=j\\}(\log e^{x^i\theta^j} - \log \sum\_{l=0}^{k-1}e^{x^i\theta^l})
\\\
\\\
L(\Theta) &= \sum\_{i=0}^{m-1}\sum\_{j=0}^{k-1}1 \\{y^i=j\\}(x^i\theta^j - \log \sum\_{l=0}^{k-1}e^{x^i\theta^l})
\\\
\end{align}

很明显似然函数最大值对应的 $\Theta$ 就是我们求解的目标，所以问题变为：
$$
\max\_\Theta L\_\Theta
$$

##梯度上升法##
使用梯度上升法可以帮助我们找到似然函数的最大值，参数 $\theta^t$ 的梯度为：

\begin{align}
\\\
\frac{\partial L(\Theta)}{\partial \theta^t} &= \frac{\partial}{\partial \theta^t} ( \sum\_{i=0}^{m-1}\sum\_{j=0}^{k-1}1 \\{y^i=j\\}(x^i\theta^j - \log \sum\_{l=0}^{k-1}e^{x^i\theta^l}) )
\\\
\\\
\frac{\partial L(\Theta)}{\partial \theta^t} &= \sum\_{i=0}^{m-1}\frac{\partial}{\partial \theta^t}(\sum\_{j=0}^{k-1}1 \\{y^i=j\\}(x^i\theta^j - \log \sum\_{l=0}^{k-1}e^{x^i\theta^l}) )
\\\
\\\
\frac{\partial L(\Theta)}{\partial \theta^t} &= \sum\_{i=0}^{m-1}\frac{\partial}{\partial \theta^t}(\sum\_{j=0}^{k-1}1 \\{y^i=j\\}x^i\theta^j - \sum\_{j=0}^{k-1}1 \\{y^i=j\\} \log \sum\_{l=0}^{k-1}e^{x^i\theta^l} )
\\\
\\\
\frac{\partial L(\Theta)}{\partial \theta^t} &= \sum\_{i=0}^{m-1}(1 \\{y^i=t\\}(x^i)^T -\frac{\partial}{\partial \theta^t} ( \sum\_{j=0}^{k-1}1 \\{y^i=j\\} \log \sum\_{l=0}^{k-1}e^{x^i\theta^l}) )
\\\
\\\
\frac{\partial L(\Theta)}{\partial \theta^t} &= \sum\_{i=0}^{m-1}(1 \\{y^i=t\\}(x^i)^T - \sum\_{j=0}^{k-1}1 \\{y^i=j\\} \frac{\partial}{\partial \theta^t} (\log \sum\_{l=0}^{k-1}e^{x^i\theta^l}) )
\\\
\\\
\frac{\partial L(\Theta)}{\partial \theta^t} &= \sum\_{i=0}^{m-1}(1 \\{y^i=t\\}(x^i)^T - \sum\_{j=0}^{k-1}1 \\{y^i=j\\} \frac{1}{\sum\_{l=0}^{k-1}e^{x^i\theta^l}} \frac{\partial}{\partial \theta^t} \sum\_{l=0}^{k-1}e^{x^i\theta^l} )
\\\
\\\
\frac{\partial L(\Theta)}{\partial \theta^t} &= \sum\_{i=0}^{m-1}(1 \\{y^i=t\\}(x^i)^T - \sum\_{j=0}^{k-1}1 \\{y^i=j\\} \frac{1}{\sum\_{l=0}^{k-1}e^{x^i\theta^l}} \frac{\partial}{\partial \theta^t} e^{x^i\theta^t} )
\\\
\\\
\frac{\partial L(\Theta)}{\partial \theta^t} &= \sum\_{i=0}^{m-1}(1 \\{y^i=t\\}(x^i)^T - \sum\_{j=0}^{k-1}1 \\{y^i=j\\} \frac{(x^i)^T e^{x^i\theta^t}}{\sum\_{l=0}^{k-1}e^{x^i\theta^l}})
\\\
\\\
\frac{\partial L(\Theta)}{\partial \theta^t} &= \sum\_{i=0}^{m-1}(1 \\{y^i=t\\}(x^i)^T - \frac{(x^i)^T e^{x^i\theta^t}}{\sum\_{l=0}^{k-1}e^{x^i\theta^l}})
\\\
\\\
\frac{\partial L(\Theta)}{\partial \theta^t} &= \sum\_{i=0}^{m-1}(x^i)^T(1 \\{y^i=t\\} - \frac{e^{x^i\theta^t}}{\sum\_{l=0}^{k-1}e^{x^i\theta^l}})
\\\
\\\
\frac{\partial L(\Theta)}{\partial \theta^t} &= \sum\_{i=0}^{m-1}(x^i)^T(1 \\{y^i=t\\} - p(y=t|x^i;\Theta))
\\\
\end{align}

梯度上升法下的 $\theta^t$ 的更新公式为：

$$
\theta^t := \theta^t + \alpha \frac{\partial L(\Theta)}{\partial \theta^t}
$$

##Softmax回归模型参数化的特点##
Softmax回归有一个不寻常的特点：它有一个“冗余”的参数集。假设我们从参数向量 $\theta^j$ 中减去向量 $\psi$ ，则概率函数变为：

\begin{align}
\\\
p(y=j|x;\Theta) &= \frac{e^{x^i(\theta^j-\psi)}}{\sum\_{l=0}^{k-1}e^{x^i(\theta^l-\psi)}}
\\\
\\\
p(y=j|x;\Theta) &= \frac{e^{x^i\theta^j}e^{-x^i\psi}}{\sum\_{l=0}^{k-1}e^{x^i\theta^l}e^{-x^i\psi}}
\\\
\\\
p(y=j|x;\Theta) &= \frac{e^{x^i\theta^j}}{\sum\_{l=0}^{k-1}e^{x^i\theta^l}}
\\\
\end{align}

我们看到从参数向量 $\theta^j$ 中减去向量 $\psi$ 并不影响概率函数的结果。也就是说我们可以得到多组参数值。所以可以设 $\psi = \theta^0$ ，这样的话 $\theta^0$ 就是一个零向量。也就是我们把 $\theta^0$设置为零向量后，我们只需要优化其他 $\theta$ 。 

##C++代码实现##
我们定义如下的接口：

```C++
    
    /// @brief Softmax回归(多分类)
    class LSoftmaxRegression
    {
    public:
        /// @brief 构造函数
        LSoftmaxRegression();
    
        /// @brief 析构函数
        ~LSoftmaxRegression();
    
        /// @brief 训练模型
        /// 如果一次训练的样本数量为1, 则为随机梯度下降
        /// 如果一次训练的样本数量为M(样本总数), 则为梯度下降
        /// 如果一次训练的样本数量为m(1 < m < M), 则为批量梯度下降
        /// @param[in] xMatrix 样本矩阵, 每一行代表一个样本, 每一列代表样本的一个特征
        /// 样本矩阵数据最好归一化(即样本矩阵全部调整为0~1之间的值), 防止数据溢出
        /// @param[in] yMatrix 类标记矩阵, 每一行代表一个样本, 每一列代表样本的一个类别
        /// 如果样本属于该类别则标记为REGRESSION_ONE, 不属于则标记为REGRESSION_ZERO
        /// @param[in] alpha 学习速度, 该值必须大于0.0f
        /// @return 成功返回true, 失败返回false(参数错误的情况下会返回失败)
        bool TrainModel(IN const LRegressionMatrix& xMatrix, IN const LRegressionMatrix& yMatrix, IN float alpha);
    
        /// @brief 使用训练好的模型预测数据
        /// @param[in] xMatrix 需要预测的样本矩阵
        /// @param[out] yMatrix 存储预测的结果矩阵, 每一行代表一个样本, 每一列代表在该类别下的概率
        /// @return 成功返回true, 失败返回false(参数错误的情况下会返回失败)
        bool Predict(IN const LRegressionMatrix& xMatrix, OUT LRegressionMatrix& yMatrix) const;
    
        /// @brief 计算似然值, 似然值为0.0~1.0之间的数, 似然值值越大模型越好
        /// @param[in] xMatrix 样本矩阵, 每一行代表一个样本, 每一列代表样本的一个特征
        /// @param[in] yMatrix 类标记矩阵, 每一行代表一个样本, 每一列代表样本的一个类别
        /// 如果样本属于该类别则标记为REGRESSION_ONE, 不属于则标记为REGRESSION_ZERO
        /// @return 成功返回似然值, 失败返回-1.0f(参数错误的情况下会返回失败)
        float LikelihoodValue(IN const LRegressionMatrix& xMatrix, IN const LRegressionMatrix& yMatrix) const;
    
    private:
        CSoftmaxRegression* m_pSoftmaxRegression; ///< Softmax回归实现对象
    };

```

LMatrix是我们自定义的矩阵类，用于方便机器学习的一些矩阵计算，关于它的详细代码可以查看链接：[猛戳我](https://github.com/BurnellLiu/TinyML/blob/master/Src/LMatrix.h)

我们为LSoftmaxRegression设计了三个方法TrainModel，Predict以及LikelihoodValue，用于训练模型，预测新数据以及计算似然值。

我们看一下TrainModel的实现：

```C++

    // 增加常数项后的样本矩阵
    LRegressionMatrix X;
    Regression::SamplexAddConstant(xMatrix, X);

    // 权重矩阵
    LRegressionMatrix& W = m_wMatrix;

    // 概率矩阵
    LRegressionMatrix P(X.RowLen, m_K, 0.0f);

    // 计算概率矩阵
    this->SampleProbK(X, W, P);

    LRegressionMatrix::SUB(yMatrix, P, P);

    // 权重向量(列向量)
    LRegressionMatrix dwVec(m_N + 1, 1, 0.0f);

    // 第一个权重值不优化, 解决Softmax回归参数有冗余的问题
    for (unsigned int k = 1; k < m_K; k++)
    {
        dwVec.Reset(m_N + 1, 1, 0.0f);
        for (unsigned int row = 0; row < X.RowLen; row++)
        {
            for (unsigned int col = 0; col < X.ColumnLen; col++)
            {
                dwVec[col][0] += X[row][col] * P[row][k];
            }
        }

        LRegressionMatrix::SCALARMUL(dwVec, alpha, dwVec);

        for (unsigned int row = 0; row < m_wMatrix.RowLen; row++)
        {
            m_wMatrix[row][k] += dwVec[row][0];
        }
    }

```

计算概率矩阵函数SampleProbK的代码如下：

```C++

    /// @brief 计算样本属于K个分类的各个概率
    /// @param[in] sampleMatrix 样本矩阵, m * n
    /// @param[in] weightMatrix 权重矩阵, n * k, 每一列为一个分类权重
    /// @param[out] probMatrix 概率矩阵, 存储每个样本属于不同分类的概率
    void SampleProbK(
        IN const LRegressionMatrix& sampleMatrix, 
        IN const LRegressionMatrix& weightMatrix, 
        OUT LRegressionMatrix& probMatrix) const
    {
        LRegressionMatrix::MUL(sampleMatrix, weightMatrix, probMatrix);

        for (unsigned int row = 0; row < probMatrix.RowLen; row++)
        {
            for (unsigned int col = 0; col < probMatrix.ColumnLen; col++)
            {
                probMatrix[row][col] = exp(probMatrix[row][col]);
            }
        }

        for (unsigned int row = 0; row < probMatrix.RowLen; row++)
        {
            float sum = 0.0f;
            for (unsigned int col = 0; col < probMatrix.ColumnLen; col++)
            {
                sum += probMatrix[row][col];
            }

            for (unsigned int col = 0; col < probMatrix.ColumnLen; col++)
            {
                probMatrix[row][col] = probMatrix[row][col]/sum;
            }
        }
    }

```

以上完整的代码可以在链接：[猛戳我](https://github.com/BurnellLiu/TinyML/tree/master/Src)查看，我们的逻辑回归被定义在文件LRegression.h和LRegression.cpp中。