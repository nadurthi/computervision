
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>



<div style="display: none;">

$$
\(
\newcommand{\mbf}[1]{\mathbf{#1}}
\newcommand{\NN}[3]{{\bf \mathcal{N}}(#1:#2,#3)}
\newcommand{\re}{\mathbb R}
\newcommand{\II}{{\bf \mathbb I}}
\newcommand{\bx}{{\bf x}}
\newcommand{\bmx}{{\bf \hat{x}}}
\newcommand{\bmxt}[2]{{\bf \hat{x}}_{#1}^{#2}}
\newcommand{\sigptset}{{\bf \mathbb (X;w)}}
\newcommand{\sigptsetele}{\mathbb (X^i;w^i)}
\newcommand{\xk}{\mathbf{x}_{k}}
\newcommand{\hhxk}{\mathbf{\hat{x}}_{k}}
\newcommand{\hhxkngk}{\mathbf{\hat{x}}_{k+1/k}}
\newcommand{\hhxkgk}{\mathbf{\hat{x}}_{k/k}}
\newcommand{\hhxkngkn}{\mathbf{\hat{x}}_{k+1/k+1}}
\newcommand{\hhxkn}{\mathbf{h}(\mathbf{x}_{k+1})}
\newcommand{\xkn}{\mathbf{x}_{k+1}}
\newcommand{\fxk}{\mathbf{f}(\mathbf{x}_{k})}
\newcommand{\hxkn}{\mathbf{h}(\mathbf{x}_{k+1})}
\newcommand{\ykn}{\mathbf{y}_{k+1}}
\newcommand{\hhykn}{\mathbf{\hat{y}}_{k+1}}
\newcommand{\Yk}{\mathbf{Y}_{k}}
\newcommand{\Ykn}{\mathbf{Y}_{k+1}}
\newcommand{\pderiv}[2]{\frac{\partial #1}{\partial #2}}
\newcommand{\eps}{\boldsymbol{\epsilon}}
\newcommand{\Nu}{\boldsymbol{\nu}}
\newcommand{\Ps}{\boldsymbol{\Psi}}
\newcommand{\xii}{\boldsymbol{\xi}}
\newcommand{\Al}{\boldsymbol{\alpha}}
\newcommand{\x}{\mathbf{x}}
\newcommand{\C}{\mathbf{C}}
\newcommand{\uu}{\mathbf{u}}
\newcommand{\z}{\mathbf{z}}
\newcommand{\q}{\mathbf{q}}
\newcommand{\A}{\mathbf{A}}
\newcommand{\B}{\mathbf{B}}
\newcommand{\f}{\mathbf{f}}
\newcommand{\hc}{\mathbf{H}}
\newcommand{\R}{\mathbf{R}}
\newcommand{\y}{\mathbf{y}}
\newcommand{\Y}{\mathbf{Y}}
\)
$$
</div>

# TEST

<article class="markdown-body">
	<h1>Unicorns</h1>
	<p>All the things</p>
</article>

# Computer vision basics

$$x=a^2+b^2 $$

# Computer vision basics

\begin{align}
x=a^2+1
\end{align}

# Computer vision basics

$$
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix}
$$

## OK

```c++
#include <iostream>

int main(){

}
```
```python
def func1():
	a=a+a
```
# Kalman Filter

1. Initial conditions, Process noise and Measurement noise are <font color="red"> Gaussian</font> distributed and mutually independent processes.

2. State and Measurement model equations are \alert{linear}.

$$
\begin{align}
\mbf{x}_{k+1}&=F_k{x}_k+\nu_k \\
y_{k+1}&=H_{k+1}{x}_{k+1}+\omega_{k+1}
\end{align}
$$


$$
\begin{align}
\mbf{x}_0 &\sim \NN{\mbf{x}_0}{\mbf{\hat{x}}_0}{P_0}\\
\mbf{\nu}_k &\sim \NN{\mbf{\nu}_k}{\mbf{0}}{Q_k}\\
\mbf{\omega}_{k+1} &\sim \NN{\mbf{\omega}_{k+1}}{0}{R_{k+1}}
\end{align}
$$


$$
\begin{align}
p(\mbf{x}_k|Y_k)=\NN{\mbf{x}_k}{\mbf{\hat{x}}_{k|k}}{P_{k|k}}
\end{align}
$$

where \(Y_k\) is set of measurement upto time \(k\),


$$
\begin{align*}
p(\mbf{x}_{k+1}|Y_k)=\int p(\mbf{x}_{k+1}|\mbf{x}_{k})\: . \:p(\mbf{x}_{k}|Y_k)\:d\mbf{x}_{k}
\end{align*}
$$

The state transition probability density \(p(\mbf{x}_{k+1}|\mbf{x}_{k})\) is given as

$$
\begin{align*}
p(\mbf{x}_{k+1}|\mbf{x}_{k})=\NN{\mbf{x}_{k+1}}{F_k\mbf{{x}}_{k}}{Q_k}
\end{align*}
$$

$$
\begin{align*}
p(\mbf{x}_{k+1}|Y_k)=\int \NN{\mbf{x}_{k+1}}{F_k\mbf{{x}}_{k}}{Q_k}\: . \: \NN{\mbf{x}_k}{\mbf{\hat{x}}_{k|k}}{P_{k|k}} \:d\mbf{x}_{k}
\end{align*}
$$

$$
\begin{align*}
p(\mbf{x}_{k+1}|Y_k)&=\frac{1}{\sqrt{(2\pi)^n |P_{k+1|k}|}} \: .\: \exp\Big[-\frac{1}{2}\big(\mbf{x}_{k+1}-\mbf{\hat{x}}_{k+1|k}\big)^T\:\:P_{k+1|k}^{-1}\:\:\big(\mbf{x}_{k+1}\mbf{\hat{x}}_{k+1|k} \big)\Big]
\end{align*}
$$

The prior pdf \(p(\mbf{x}_{k+1}|Y_k)\) has mean \(\mbf{\hat{x}}_{k+1|k}\) and covariance \(P_{k+1|k}\) given as

$$
\begin{align*}
{\mbf{\hat{x}}_{k+1|k}}&={F_k\mbf{\hat{x}}_{k|k}}\\
{P_{k+1|k}}&={F_kP_{k|k}F^T+Q_k}
\end{align*}
$$


As we know the pdf remains Gaussian, an easier approach is to directly compute the mean and covariance from the linear model equations as:

$$
\begin{align*}
\mbf{\hat{x}}_{k+1|k}&=E[\mbf{x}_{k+1}]=E\big[F_kx_k\big]+E[\nu_k]=F_kE\big[x_{k} \big]=F_k\mbf{\hat{x}}_{k|k}\\
P_{k+1|k}&=E\big[ \big(\mbf{x}_{k+1}-F_k\mbf{\hat{x}}_{k|k} \big) \big(\mbf{x}_{k+1}-F_k\mbf{\hat{x}}_{k|k} \big)^T \big]=E\big[ \big(F_kx_k+\nu_k-F_k\mbf{\hat{x}}_{k|k} \big) \big(F_kx_k+\nu_k-F_k\mbf{\hat{x}}_{k|k} \big)^T \big]\\
&=E\big[ F_k(x_k-\mbf{\hat{x}}_{k|k})(x_k-\mbf{\hat{x}}_{k|k})^TF_k^T+\nu_k\nu_k^T \big]=F_kP_{k|k}F^T+Q_k
\end{align*}
$$


Starting with the prior pdf at time step \(k+1\),
$$
\begin{align*}
p(\mbf{x}_{k+1}|Y_k)=\NN{\mbf{x}_{k+1}}{\mbf{\hat{x}}_{k+1|k}\:\:}{P_{k+1|k}}
\end{align*}
$$
the posterior pdf from Bayes' rule at the same time step is given by
$$
\begin{align*}
p(\mbf{x}_{k+1}|Y_{k+1})=\frac{p(\mbf{y}_{k+1}|\mbf{x}_{k+1})p(\mbf{x}_{k+1}|Y_k)}{\int p(\mbf{y}_{k+1}|\mbf{x}_{k+1})p(\mbf{x}_{k+1}|Y_k)\:d\mbf{x}_{k+1}}
\end{align*}
$$


where the measurement likelihood pdf \(p(\mbf{y}_{k+1}|\mbf{x}_{k+1})\) can be derived from the measurement model equations as:

$$
\begin{align*}
p(\mbf{y}_{k+1}|\mbf{x}_{k+1})=\NN{\mbf{y}_{k+1}}{H_{k+1}\mbf{x}_{k+1}}{\:\:R_{k+1}}
\end{align*}
$$

Numerator of Bayes' rule is simplified as:

$$
\begin{align*}
p(\mbf{y}_{k+1}|\mbf{x}_{k+1})p(\mbf{x}_{k+1}|Y_k)&=\NN{\mbf{y}_{k+1}}{H_{k+1}\mbf{x}_{k+1}}{\:\:R_{k+1}}\NN{\mbf{x}_{k+1}}{\mbf{\hat{x}}_{k+1|k}\:\:}{P_{k+1|k}}
\end{align*}
$$

$$
\begin{align*}
&=\frac{1}{\sqrt{|2\pi R_{k+1}|}}\:.\: \frac{1}{\sqrt{|2\pi P_{k+1|k}|}}\: .\: \exp\Big[-\frac{1}{2}\big( \mbf{y}_{k+1}-H_{k+1}\mbf{x}_{k+1}\big)^TR_{k+1}^{-1}\big( \mbf{y}_{k+1}-H_{k+1}\mbf{x}_{k+1}\big) \\
&\hspace{5cm} \: \: -\frac{1}{2}\big( \mbf{x}_{k+1}-\mbf{\hat{x}}_{k+1|k}\big)^T P_{k+1|k}^{-1} \big(\mbf{x}_{k+1}-\mbf{\hat{x}}_{k+1|k}\big) \Big]
\end{align*}
$$


The exponent is:

$$
\begin{align*}
&\Rightarrow-\frac{1}{2}\big( \mbf{y}_{k+1}-H_{k+1}\mbf{x}_{k+1}\big)^TR_{k+1}^{-1}\big( \mbf{y}_{k+1}-H_{k+1}\mbf{x}_{k+1}\big) -\frac{1}{2}\big( \mbf{x}_{k+1}-\mbf{\hat{x}}_{k+1|k}\big)^T P_{k+1|k}^{-1} \big(\mbf{x}_{k+1}-\mbf{\hat{x}}_{k+1|k}\big)\\
&\Rightarrow -\frac{1}{2}\Big[ \mbf{y}_{k+1}^TR_{k+1}^{-1}\mbf{y}_{k+1} + \mbf{x}_{k+1}^T\underbrace{\big( P_{k+1|k}^{-1}+H_{k+1}^T R_{k+1}^{-1}H_{k+1}\big)}_{A}\mbf{x}_{k+1}\\
&\hspace{4cm} -2 \underbrace{\big(\mbf{y}_{k+1}^TR_{k+1}^{-1}H_{k+1} +\mbf{\hat{x}}_{k+1|k}^T P_{k+1|k}^{-1}\big)}_{b^T}\mbf{x}_{k+1}+ \mbf{\hat{x}}_{k+1|k}^T P_{k+1|k}^{-1} \mbf{\hat{x}}_{k+1|k} \Big]\\
&\Rightarrow -\frac{1}{2}\Big[ \mbf{y}_{k+1}^TR_{k+1}^{-1}\mbf{y}_{k+1} + \mbf{x}_{k+1}^TA\mbf{x}_{k+1} -2 b^T\mbf{x}_{k+1}+ \mbf{\hat{x}}_{k+1|k}^T P_{k+1|k}^{-1} \mbf{\hat{x}}_{k+1|k} \Big]\\
\end{align*}
$$


The denominator of Bayes' rule is given as:

$$
\begin{align*}
&\int p(\mbf{y}_{k+1}|\mbf{x}_{k+1})p(\mbf{x}_{k+1}|Y_k)\:d\mbf{x}_{k+1}=\frac{1}{\sqrt{|2\pi R_{k+1}|}}\:.\: \frac{1}{\sqrt{|2\pi P_{k+1|k}|}}\: .\:\\
&\int \exp\Big[ -\frac{1}{2} \mbf{y}_{k+1}^TR_{k+1}^{-1}\mbf{y}_{k+1} -\frac{1}{2} \mbf{x}_{k+1}^TA\mbf{x}_{k+1} +b^T\mbf{x}_{k+1}-\frac{1}{2} \mbf{\hat{x}}_{k+1|k}^T P_{k+1|k}^{-1} \mbf{\hat{x}}_{k+1|k} \Big]\:d\mbf{x}_{k+1}\\
&=\frac{1}{\sqrt{|2\pi R_{k+1}|}}\:.\: \frac{1}{\sqrt{|2\pi P_{k+1|k}|}}\: .\: \exp\Big[ -\frac{1}{2} \mbf{y}_{k+1}^TR_{k+1}^{-1}\mbf{y}_{k+1}-\frac{1}{2} \mbf{\hat{x}}_{k+1|k}^T P_{k+1|k}^{-1} \mbf{\hat{x}}_{k+1|k} \Big]\\
&\hspace{3cm}\: .\: \int \exp\Big[ -\frac{1}{2} \mbf{x}_{k+1}^TA\mbf{x}_{k+1} +b^T\mbf{x}_{k+1}\Big]\:d\mbf{x}_{k+1}\\
&=\frac{\sqrt{|2\pi A^{-1}|}}{\sqrt{ |2\pi R_{k+1}||2\pi P_{k+1|k}|}}\: .\: \exp\Big[ -\frac{1}{2} \mbf{y}_{k+1}^TR_{k+1}^{-1}\mbf{y}_{k+1}-\frac{1}{2} \mbf{\hat{x}}_{k+1|k}^T P_{k+1|k}^{-1} \mbf{\hat{x}}_{k+1|k} +\frac{1}{2}b^TA^{-T}b \Big]
\end{align*}
$$
The posterior pdf from Bayes' rule is given as:
$$
\begin{align*}
p(\mbf{x}_{k+1}|Y_k)&=\frac{1}{\sqrt{|2\pi A^{-1}|}} \: . \: \exp \Big[-\frac{1}{2}\mbf{x}_{k+1}^TA\mbf{x}_{k+1} + b^T\mbf{x}_{k+1} -\frac{1}{2}b^TA^{-T}b \Big]\\
&=\frac{1}{\sqrt{|2\pi A^{-1}|}} \: . \: \exp \Big[-\frac{1}{2}\big(\mbf{x}_{k+1}-A^{-1}b\big)^TA\big(\mbf{x}_{k+1}-A^{-1}b\big) \Big]\\
\end{align*}
$$

$$
\begin{align}
p(\mbf{x}_{k+1}|Y_{k+1})&=\frac{1}{\sqrt{|2\pi A^{-1}|}} \: . \: \exp \Big[-\frac{1}{2}\big(\mbf{x}_{k+1}-A^{-1}b\big)^TA\big(\mbf{x}_{k+1}-A^{-1}b\big) \Big]
\end{align}
$$

$$
\begin{align}
b^T&=\big(\mbf{y}_{k+1}^TR_{k+1}^{-1}H_{k+1} +\mbf{\hat{x}}_{k+1|k}^T P_{k+1|k}^{-1}\big)\\
A&= P_{k+1|k}^{-1}+H_{k+1}^T R_{k+1}^{-1}H_{k+1}\\
A^{-1}&=P_{k+1|k}-\underbrace{P_{k+1|k}H_{k+1}^T\big(R_{k+1} +H_{k+1} P_{k+1|k} H_{k+1}^T \big)^{-1}}_{K_{k+1}} H_{k+1}P_{k+1|k}\\
&=P_{k+1|k}-K_{k+1}H_{k+1}P_{k+1|k}\\
A^{-1}b&=\big( P_{k+1|k}-K_{k+1}H_{k+1}P_{k+1|k} \big)\big(H_{k+1}^TR_{k+1}^{-1}\mbf{y}_{k+1} +P_{k+1|k}^{-1}\mbf{\hat{x}}_{k+1|k}\big)\\
&=\mbf{\hat{x}}_{k+1|k}+K_{k+1}\big(\mbf{y}_{k+1}-H_{k+1}\mbf{\hat{x}}_{k+1|k} \big)
\end{align}
$$
Hence the posterior pdf still remains Gaussian even after Bayes' Rule update, with mean and covariance as
$$
\begin{align}
\mbf{\hat{x}}_{k+1|k+1}&=A^{-1}b=\mbf{\hat{x}}_{k+1|k}+K_{k+1}\big(\mbf{y}_{k+1}-H_{k+1}\mbf{\hat{x}}_{k+1|k} \big)\\
P_{k+1|k+1}&=A^{-1}=P_{k+1|k}-K_{k+1}H_{k+1}P_{k+1|k}
\end{align}
$$

$$
\begin{align}
p(\mbf{x}_{k+1}|Y_k)&=\NN{\mbf{x}_{k+1}}{\mbf{\hat{x}}_{k+1|k}\:\:}{P_{k+1|k}}\\
\mbf{\hat{x}}_{k+1|k}&=F_k\mbf{\hat{x}}_{k|k}\\
P_{k+1|k}&=F_kP_{k|k}F^T+Q_k
\end{align}
$$

$$
\begin{align}
p(\mbf{x}_{k+1}|Y_{k+1})&=\NN{\mbf{x}_{k+1}}{\mbf{\hat{x}}_{k+1|k+1}\:\:}{P_{k+1|k+1}}\\
\mbf{\hat{x}}_{k+1|k+1}&=\mbf{\hat{x}}_{k+1|k}+K_{k+1}\big(\mbf{y}_{k+1}-H_{k+1}\mbf{\hat{x}}_{k+1|k} \big)\\
P_{k+1|k+1}&=P_{k+1|k}-K_{k+1}H_{k+1}P_{k+1|k}\\
K_{k+1}&=P_{k+1|k}H_{k+1}^T\big(R_{k+1} +H_{k+1} P_{k+1|k} H_{k+1}^T \big)^{-1}
\end{align}
$$

<h2>Minimum Variance Estimate</h2>
Starting from the prior pdf $p(\mbf{x}_{k+1}|Y_{k+1})=\NN{\mbf{x}_{k+1}}{\mbf{\hat{x}}_{k+1|k+1}\:\:}{P_{k+1|k+1}}$ at time $k+1$

$$
\begin{align}
\min\limits_{\hhxkngkn} \: Tr\Big\{E[(\xkn-\hhxkngkn)(\xkn-\hhxkngkn)^T]\Big\}
\end{align}
$$

i.e. find an estimate that minimizes the posterior variance

$$
\begin{align}
\hhxkngkn =\hhxkngk+\mbf{K}_{k+1}(\ykn-\hhykn)
\end{align}
$$

The assumed estimator is unbiased:

$$
\begin{align}
E[\hhxkngkn]&=E[\hhxkngk+\mbf{K}_{k+1}(\ykn-\hhykn)]=E[\hhxkngk]+\mbf{K}_{k+1}(E[\ykn]-E[\hhykn])\\
&=F_kE[\mbf{\hat{x}}_{k|k}]+\mbf{K}_{k+1}(H_{k+1}E[\mbf{x}_{k+1}]-H_{k+1}E[\hhxkngk])\\
&=E[\mbf{x}_{k+1}]+\mbf{K}_{k+1}(H_{k+1}F\mbf{\hat{x}}_{k|k}-H_{k+1}F\mbf{\hat{x}}_{k|k})=E[\mbf{x}_{k+1}]
\end{align}
$$

<h2>Variance approach for Kalman Filter</h2>

$$
\begin{align*}
&\min\limits_{\mbf{K}_{k+1}} \: Tr\Big\{E[(\xkn-\hhxkngkn)(\xkn-\hhxkngkn)^T]\Big\}\\
&\textit{with } \quad \hhxkngkn =\hhxkngk+\mbf{K}_{k+1}(\ykn-\hhykn)
\end{align*}
$$

$$
\begin{align*}
&\min\limits_{\mbf{K}_{k+1}} \: Tr\Big\{E[(\xkn-\hhxkngk-\mbf{K}_{k+1}(\ykn-\hhykn))(\xkn-\hhxkngk-\mbf{K}_{k+1}(\ykn-\hhykn))^T]\Big\}\\
&\min\limits_{\mbf{K}_{k+1}} \: Tr\Big\{ E[\big(\xkn-\hhxkngk \big)\big(\xkn-\hhxkngk \big)^T] +\mbf{K}_{k+1}E[(\ykn-\hhykn)(\ykn-\hhykn)^T]\mbf{K}_{k+1}^T \\
&\hspace{1cm} -\mbf{K}_{k+1}E[(\ykn-\hhykn)(\xkn-\hhxkngk)^T]- E[\big(\xkn-\hhxkngk \big)(\ykn-\hhykn)^T]\mbf{K}_{k+1}^T \Big\}\\
&\min\limits_{\mbf{K}_{k+1}} \: Tr\Big\{P_{k+1|k}+ \mbf{K}_{k+1} P^y_{k+1|k} \mbf{K}_{k+1}^T-\mbf{K}_{k+1}P^{yx}_{k+1|k}-P^{xy}_{k+1|k} \mbf{K}_{k+1}^T \Big\}
\end{align*}
$$


The optimal gain $K_{k+1}$ is given by

$$
\begin{align*}
\mbf{K}_{k+1}&=\mbf{P}^{xy}_{k+1}(\mbf{P}^{y}_{k+1/k})^{-1}
\end{align*}
$$

$$
\begin{align*}
\hhykn &=E[\mbf{y}_{k+1}]=E[H_{k+1}x_{k+1}+\omega_{k+1}]=H_{k+1}E[x_{k+1}]=H_{k+1}\mbf{\hat{x}}_{k+1|k}\\
%&=H_{k+1}\int \mbf{x}_{k+1}\:\: \NN{\mbf{x}_{k+1}}{\mbf{\hat{x}}_{k+1|k}\:\:}{P_{k+1|k}}\:d\mbf{x}_{k+1}=H_{k+1}\mbf{\hat{x}}_{k+1|k}\\
{P}^{y}_{k+1/k}&=E[(\ykn-\hhykn)(\ykn-\hhykn)^T]\\
&=E[\big(H_{k+1}\mbf{x}_{k+1}+\omega_{k+1}-H_{k+1}\mbf{\hat{x}}_{k+1|k}\big)\big(H_{k+1}\mbf{x}_{k+1}+\omega_{k+1}-H_{k+1}\mbf{\hat{x}}_{k+1|k}\big)^T]\\
&=H_{k+1}E[\big(\mbf{x}_{k+1}-\mbf{\hat{x}}_{k+1|k} \big)\big(\mbf{x}_{k+1}-\mbf{\hat{x}}_{k+1|k} \big)^T]H_{k+1}^T+E[\omega_{k+1}\omega_{k+1}^T]\\
&=H_{k+1}P_{k+1|k}H_{k+1}^T+R_{k+1}\\
{P}^{xy}_{k+1}&=E[\big(\xkn-\hhxkngk \big)(\ykn-\hhykn)^T]\\
&=E[\big(\xkn-\hhxkngk \big) \big( H_{k+1}(\xkn-\hhxkngk)+\omega_{k+1} \big)^T]\\
&=E[\big(\mbf{x}_{k+1}-\mbf{\hat{x}}_{k+1|k} \big)\big(\mbf{x}_{k+1}-\mbf{\hat{x}}_{k+1|k} \big)^T]H_{k+1}=P_{k+1|k}H_{k+1}^T
\end{align*}
$$

The Kalman Filter gain is then given as:

$$
\begin{align}
\mbf{K}_{k+1}&=\mbf{P}^{xy}_{k+1}(\mbf{P}^{y}_{k+1/k})^{-1}=P_{k+1|k}H_{k+1}^T\big( H_{k+1}P_{k+1|k}H_{k+1}^T+R_{k+1}\big)^{-1}
\end{align}
$$

<h2>Optimization approach - Recursive Least Squares</h2>

$$
\begin{align}
J=\frac{1}{2}(\mbf{x}_0-\hat{\mbf{x}}_0)^TP_0^{-1}(\mbf{x}_0-\hat{\mbf{x}}_0)+\frac{1}{2}\sum_{k=1}^{T}(\mbf{y}_k-H_k\mbf{x}_k)^TR_k^{-1}(\mbf{y}_k-H_k\mbf{x}_k)
\end{align}
$$

subject to:

$$
\begin{align*}
\mbf{x}_{k+1}=F_k\mbf{x}_k
\end{align*}
$$

minimize w.r.t. $[\mbf{x}_0,\mbf{x}_1,\cdots,\mbf{x}_T]$


