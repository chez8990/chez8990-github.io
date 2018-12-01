---
layout: post
title: Little fact used in the IWAE paper
---

There's a little lemma used in the Importance Weighted Autoencoder paper <a href='#1'>[1]</a>

Let $I \in \{1,\cdots, k\}$ with $\|I\|=m$ be a uniformly distributed subset of distinct indices. Then 

$$
	\begin{align*}
		\mathbb{E}_{I=\{i_1,\cdots, i_m\}}\left[\frac{a_{i_1}+\cdots+a_{i_1}}{m}\right] = \frac{a_1+\cdots+a_k}{m}
	\end{align*}
$$

proof:

The probability of a random subset $S$ with $m$ distinct indices chosen from $I$ is ${m \choose k}^{-1}$. Therefore

$$
	\begin{align*}
		\mathbb{E}_{I=\{i_1,\cdots, i_m\}}\left[\frac{a_{i_1}+\cdots+a_{i_1}}{m}\right] 
			&= \sum_{I={i_1, \cdots, i_k}} \frac{1}{m \choose k}\frac{a_{i_1}+\cdots+a_{i_1}}{m} \\
			&= \frac{(m-1)!(k-m)!}{k!} \sum_{I={i_1, \cdots, i_k}} a_{i_1} + \cdots + a_{i_m} \\
			&= \frac{(m-1)!(k-m)!}{k!} \cdot {k-1 \choose m-1} \: (a_1 + \cdots + a_k)\\
			&= \frac{a_1 + \cdots + a_k}{k}
	\end{align*}
$$

The part that needs explaining is line 3, the number of times $a_j$ appears in the sum in line 2 is exacly ${k-1\choose m-1}$. This is because once any first index fixed, there are ${k-1\choose m-1}$ combinations such an index appears in. 

### Reference
<div id='1'>[1]</div> Yuri Burda, Roger Grosse & Ruslan Salakhutdinov. 2016. Importance Weighted Autoencoder. arXiv:1509.00519