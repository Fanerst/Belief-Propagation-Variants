# Belief-Propagation-Variants

## Belief Propagation of cavity fields

For traditional Belief Propagation equation (BPE):
$$
q_{i\rightarrow a}(\sigma) = \frac{
    \psi_i(\sigma) \prod_{b\in\partial i \backslash a} p_{b\rightarrow i}(\sigma)
    }{
        \sum_\sigma\psi_i(\sigma) \prod_{b\in\partial i \backslash a} p_{b\rightarrow i}(\sigma)
    }
\\
p_{a\rightarrow a}(\sigma) = \frac{
    \sum_{\underline\sigma_{\partial a}}\delta(\sigma_i, \sigma) \psi_a(\underline\sigma_{\partial a}) \prod_{j\in\partial a \backslash i} q_{j\rightarrow a} (\sigma_j)
    }{
        \sum_\sigma\sum_{\underline\sigma_{\partial a}}\delta(\sigma_i, \sigma) \psi_a(\underline\sigma_{\partial a}) \prod_{j\in\partial a \backslash i} q_{j\rightarrow a} (\sigma_j)
    }
$$

where $\psi_i(\sigma) = \exp(-\beta E_i(\sigma))$ and $\psi_a(\underline\sigma_{\partial a})=\exp(-\beta E_a(\underline\sigma_{\partial a}))$.

When we consider the Ising model with binary variables $\sigma\in\{-1, 1\}$, its energy form is $E(\vec{\sigma}) = - \sum_i h_i\sigma_i - \sum_{<i, j>} J_{ij}\sigma_i\sigma_j$, the BPE has the form
$$
q_{i\rightarrow a}(\sigma) = \frac{
    e^{\beta h_i\sigma_i} \prod_{b\in\partial i \backslash a} p_{b\rightarrow i}(\sigma)
    }{
        \sum_\sigma\psi_i(\sigma) \prod_{b\in\partial i \backslash a} p_{b\rightarrow i}(\sigma)
    }
\\
p_{a\rightarrow a}(\sigma) = \frac{
    \sum_{\sigma_j} e^{\beta J_{ij}\sigma\sigma_j} q_{j\rightarrow a} (\sigma_j)
    }{
        \sum_\sigma\sum_{\sigma_j} e^{\beta J_{ij}\sigma\sigma_j} q_{j\rightarrow a} (\sigma_j)
    }
$$
transfer to cavity field representation ($p_{a\rightarrow i}=e^{\beta u_{a\rightarrow i}\sigma} / 2\cosh(\beta u_{a\rightarrow i})$ and $q_{i\rightarrow a}=e^{\beta h_{i\rightarrow a}\sigma} / 2\cosh(\beta h_{i\rightarrow a})$), then BPE becomes
$$
h_{i\rightarrow a} = h_i^0 + \sum_{b\in\partial i \backslash a} u_{b\rightarrow i}
\\
u_{a\rightarrow i} = \frac{1}{\beta} \text{arctanh}[\tanh(\beta J_{ij})\tanh(\beta h_{j\rightarrow a})]
$$
combining two equations together we will get
$$
h_{i\rightarrow j} = h_i^0 + \sum_{k\in\partial i \backslash j} \frac{1}{\beta} \text{arctanh}[\tanh(\beta J_{ik})\tanh(\beta h_{k\rightarrow i})]
$$
where $h_i^0 = \frac{1}{2\beta}\ln[\psi_i(+1)/\psi_i(-1)] = h_i$.

Other physical properties can also be described by cavity field $h_{i\rightarrow j}$.

Free energy:
$$
F_0 = -\frac{1}{\beta} \sum_i\ln[\sum_{\sigma_i} e^{\beta h_i \sigma_i} \prod_{j\in\partial i}\sum_{\sigma_j}e^{\beta J_{ij}\sigma_i\sigma_j}q_{j\rightarrow i}(\sigma_j)] + \frac{1}{\beta} \sum_{<i,j>}\ln[\sum_{\sigma_i\sigma_j}e^{\beta J_{ij}\sigma_i\sigma_j}q_{j\rightarrow i}(\sigma_j)q_{i\rightarrow j}(\sigma_i)] \\
= -\frac{1}{\beta} \sum_i\ln[\sum_{\sigma_i} e^{\beta h_i \sigma_i} \prod_{j\in\partial i}\sum_{\sigma_j}e^{\beta J_{ij}\sigma_i\sigma_j} \frac{e^{\beta h_{j\rightarrow i}\sigma_j}}{2\cosh(\beta h_{j\rightarrow i})}] + \frac{1}{\beta} \sum_{<i,j>}\ln[\sum_{\sigma_i\sigma_j}e^{\beta J_{ij}\sigma_i\sigma_j} \frac{e^{\beta h_{j\rightarrow i}\sigma_j}}{2\cosh(\beta h_{j\rightarrow i})} \frac{e^{\beta h_{i\rightarrow j}\sigma_i}}{2\cosh(\beta h_{i\rightarrow j})}] \\
= -\frac{1}{\beta} \sum_i\ln[\prod_{j\in\partial i} \frac{e^{\beta h_i}\cosh\beta(J_{ij} + h_{j\rightarrow i}) + e^{-\beta h_i}\cosh\beta(-J_{ij} + h_{j\rightarrow i})}{\cosh\beta h_{j\rightarrow i}}] + \\
\frac{1}{\beta}\sum_{<i,j>}\ln[\frac{e^{\beta J_{ij}}\cosh\beta(h_{i\rightarrow j} + h_{j\rightarrow i}) + e^{-\beta J_{ij}}\cosh\beta(h_{i\rightarrow j} - h_{j\rightarrow i})}{2\cosh(\beta h_{i\rightarrow j})cosh(\beta h_{j\rightarrow i})}]

$$