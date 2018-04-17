---
layout: post
title: Autoencoder
---

### What is an autoencoder
An autoencoder is a deep neural network atchitecture that is used in unsupervised learning, it consists of three parts, an encoder, a decoder and a latent space. The goal of an autoencoder is to encode high dimensional data to a low dimensional subspace by means of feature learning. 
![Autoencoder_model]({{ "https://upload.wikimedia.org/wikipedia/commons/2/28/Autoencoder_structure.png"}})

The formulation is as follows:

Given an input vector \\( X \in \mathbb{R}^n \\), the encoder $$ f:\mathbb{R}^n\rightarrow\mathbb{R}^m $$ does the following 

$$ Z = f(X) \in \mathbb{R}^m$$

where $$m\leq n $$. In other words, the encoder compresses the input to a lower dimensional vector space called the "latent space", correspondingly $$ Z $$ is called the "latent reprepsentation" of the input $$X$$. 

The decoder on the other hand, takes the latent vector $$ Z $$ and tries to reconstruct the original input vector $$ X $$. i.e

$$ X^\prime = f^{-1}(Z) \in \mathbb{R}^n $$

So an autoencoder is essentially an approximation of the identity function on $$\mathbb{R}^n$$.
The network is then trained to minimize the reconstruction loss of $$ X^\prime$$. Often times it will be the mean-squared-error

$$ l(X, X^\prime) = \frac{1}{n} \sum_{i=1}^n ||x_i - x^\prime_i||^2 $$


### Why autoencoder?
The latent space has a surprising linear structure, meaning one can perform linear arithmetic on the latent vectors, and the corresponding decoder output $$ X^\prime $$ will reflect the arithmetic done on the latent vectors. 

Due to this, one can "evolve" from input vector $$ X_1 $$ to $$ X_2 $$ to another by decoding the line segment between latent vectors $$ Z_1 $$ and $$ Z_2 $$.

$$ Z = f^{-1}(Z_1(1-t) + Z_2\:t) \quad t\in [0,1]$$

{: .center}
![face_latent](/images/face_latent.jpeg)

