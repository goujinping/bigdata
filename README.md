VAE是基于变分思想的深度学习生成模型，VAE是Variational Auto Encoder的简称。
 
## 一. 生成模型和判别模型

&emsp;&emsp;已知观察变量$X$和隐含变量$z$，判别式模型对$p(z|X)$进行建模，它是根据输入的观察变量$x$得到隐含变量$z$出现的可能性。而生成模型则是反过来，它要对$p(X|z)$进行建模，输入是隐含变量，输出则是观察变量的概率。如果想通过生成模型去解决判别问题，则需要通过贝叶斯公式转换。如果想通过判别模型去生成数据，效率十分低下，而生成模型解决却很简单自然，先确定好$z$的取值，然后根据$p(X|z)$的分布进行随机采样就好了。

## 二. VAE模型推导过程

生成模型一般通过最大化后验概率的形式进行建模优化，根据贝叶斯公式有

$$p(z|X)=\frac{p(X|z)p(z)}{\int_zp(X|z)p(z)dz}$$

这个公式在复杂的模型和大规模数据面前极难求解。为了方便问题的解决，我们用一个新的函数代替后验概率，那么这两个概率分布需要尽可能近，使用$KL$散度衡量两者的相近程度，则

$$
\begin{aligned}
KL(q(z)||p(z|X))&=\int q(z)\log\frac{q(z)}{p(z|X)}dz\\\\
&=\int q(z)\left[\log q(z)-\log p(z|X)\right]dz\\\\
&=\int q(z)\left[\log q(z)-\log \frac{p(X|z)p(z)}{p(X)}\right]dz\\\\
&=\int q(z)\left[\log q(z)-\log p(X|z)-\log p(z)+\log p(X)\right]dz\\\\
&=\int q(z)\left[\log q(z)-\log p(X|z)-\log p(z)\right]dz+\log p(X)\\\\
&=KL(q(z)||p(z))-\int q(z)\log p(X|z)dz+\log p(X)\\\\
\end{aligned}
$$
$$
\begin{array}{c|lcr}
n & \text{Left} & \text{Center} & \text{Right} \\\\
1 & 0.24 & 1 & 125 \\\\
2 & -1 & 189 & -8 \\\\
3 & -20 & 2000 & 1+10i
\end{array}
$$

$$
  \begin{pmatrix}
  1 & a_1 & a_1^2 & \cdots & a_1^n\\\\
  1 & a_2 & a_2^2 & \cdots & a_2^n \\\\
  \vdots & \vdots & \ddots & \vdots \\\\  
  1 & a_n & a_n^2 & \cdots & a_n^n  \\\\
  \end{pmatrix}
$$
