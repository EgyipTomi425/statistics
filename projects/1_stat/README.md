# [Rice](https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik) adathalmaz

> A momentumok implementációja videókártyán [Pébay cikkje](https://people.xiph.org/~tterribe/pubs/P%C3%A9bayTerriberryKolla%2B16-_Numerically_Stable%2C_Scalable_Formulas_for_parallel_and_Online_Computation_of_Higher-Order_Multivariate_Central_Moments_with_Arbitrary_Weights.pdf) alapján készült. Ezek a [statistics.cu](https://github.com/EgyipTomi425/statistics/blob/master/statistics_cuda.cu) fájlban találhatóak.

$$
\mu_p = \sum_{k=0}^{p} \binom{p}{k}
\left( \sum_{i=1}^{n} (x_i - \bar{x})^{p-k} \right)
\left( -\frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x}) \right)^k
$$

$$
\mu_4
= S_4 - 4 \, \delta \, S_3 + 6 \, \delta^2 \, S_2 - 3 \, N \, \delta^4
$$

$$
\mu_4 = \sum_{i=1}^{N} (x_i - \bar{x})^4 - 4 \sum_{i=1}^{N} (x_i - \bar{x})^3 \left( \frac{1}{N} \sum_{j=1}^{N} (x_j - \bar{x}) \right) + 6 \sum_{i=1}^{N} (x_i - \bar{x})^2 \left( \frac{1}{N} \sum_{j=1}^{N} (x_j - \bar{x}) \right)^2 - 3 N \left( \frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x}) \right)^4
$$