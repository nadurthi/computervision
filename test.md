
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<div style="display: none;">
\(
\newcommand{\mbf}[1]{\mathbf{#1}}
\newcommand{\NN}[3]{{\bf \mathcal{N}}(#1:#2,#3)}
\)
</div>

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

# Kalman Filter

1. Initial conditions, Process noise and Measurement noise are \alert{Gaussian} distributed and mutually independent processes.

2. State and Measurement model equations are \alert{linear}.

$$
\begin{align}
x_{k+1}&=F_k{x}_k+\nu_k \\
y_{k+1}&=H_{k+1}{x}_{k+1}+\omega_{k+1}
\end{align}
$$

