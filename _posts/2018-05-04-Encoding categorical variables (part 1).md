---
layout: post
title: Feature engineering - Encoding categorical variables (Part 1)
---


### Why u do dis?

Categorical variables are discrete variables that take values from a finite set. Examples of some categorical variables are 

<ul>
	<li> Blood type: A, B, C</li>
	<li> Countries: France, England </li>
	<li> The ranking of an NBA team: 1, 2, 3 </li>
</ul>

While some machine learning algorithms can use these variables directly (tree based algorithms), most algorithms can't handle non-numeric variables. It is then necessary to transform them to numerical variables if we were to use these algorithms. 

### Label encoding and One-hot encoding

**Label encoding** is the act of randomly assigning integer values to categories. E.g

Blood type - A:1, B:2, AB:3

While this encoding scheme solves part of the problem, they introduce numerical structures that are not apparent in the original variable. For instance, we see that an ordering of blood type has emerged $A<B<AB$, moreover airthmetic can now be performed 

$$\frac{A+B+C}{3} = 2 = B$$

So the average of all blood types is $B$, which does not seem very sensical. Also, the order can be randomly permuted so that our little example above could essentialy give non-unique answers. Which is undesirable. 

**One-hot encoding** converts each category to a binary variable, where a 1 indicates that the sample belongs to that category and 0 otherwise. For example, if we have three categories $cat, dog, mouse$, then 
<table><tr><th></th><th>dog</th><th>cat</th><th>mouse</th></tr><tr><td>dog</td><td>1</td><td>0</td><td>0</td></tr><tr><td>mouse</td><td>0</td><td>0</td><td>1</td></tr></table>

This is both very easy to understand and effective when the number of categories is not too much. However, if we have 10000 different categories, then it becomes problematic, namely we will encounter the <a href='https://en.wikipedia.org/wiki/Curse_of_dimensionality'> curse of dimensionality </a> and memory issues. For a 10000 category, the total number of combinations is $2^{10000}$, which is a huge number. For a single decision tree covering all these combinations, this is the number of leaves in the tree. You can imagine how long it will take to learn even the easiest pattern, and so the algorithm might choose to only learn the basic patterns such as random guessing by class probability etc.

### The problem

Our main problems are the high dimensionality and undesirable arithmetic strucutre brought by these encoding methods. Is there a way that we can reduce these dimensions while not introducing arithematics that don't make sense? 

### Target encoding

**Target encoding** <a href='#1'>[1]</a> is a method of converting categorical variables to a numerical variable using information from the target variable (if it's available). So the output will be a single vector instead of a matrix, our high dimensionality problem would have been averted if this works.

The details are quite simple to understand actually. Given $x=c$ where $c$ is a category with $n$ number of samples, the encoding for $x$ is given by 

$$ IC(x) = \begin{cases} 
			\lambda(n)\mathbb{E}[y \mid x] - (1-\lambda(n))\mathbb{E}[y] & \text{if regression} \\
			\\
			\lambda(n)\Pr(y \mid x) - (1-\lambda(n))\Pr(y) & \text{if classification} 
			\end{cases}
$$

$$\lambda:[0, \infty) \rightarrow [0,1]$$ is a monotonically increasing function.

Let me explain what's happening, we want to encode the category $c$ by the class mean/prboability, this is given by the quantities $$\mathbb{E}[y \mid c]$$ and $\Pr(y \mid x)$. This is the posterior distribution that we are estimating from the samples provided, when the per class sample size is small, the estimation diffiates from the true posterior distribution, which would give incorrect information about the class and introduce noise to the variable overall.

Instead, we include a level of belief called $\lambda$ between the estimated posterior and the sample prior $\Pr(y)$/$\mathbb{E}[y]$. The methodology here is, the more sample of category $c$ we can get, the more we beileve in the estimated posterior, so $\lambda$ must be monotonically increasing.

There are many choices of $\lambda$ one can choose. However, the chosen function should be reflective of the data size one's currently working with. A $\lambda$ was suggested by Micci-Barreca in his paper<a href='#1'>[1]</a>. 

$$ \lambda(n) = \frac{1}{1+e^{\frac{n-k}{f}}}$$

$k$ is known as the "threshold" and $f$ is a smoothing parameter that determines the "slope" of the function around $k$. If $f\rightarrow \infty$, $\lambda$ basically becomes a hard-thresholding.

#### What about the bias bro?

Now, including information from the target variable sounds all kinds of wrong in the world of machine learning, as it would cause overfitting if not done carefully. Moreover, when applied to testing data, there are no labels provided, so how do we cope that situation? 

Those are all legit concerns, one way to overcome all of these issues is to use "cross-validation". I put it in quotes because we are not actually evaluating the score of a model, we simply want to use the methodology of cross-validation to reduce bias built up by using target encoding. 


#### Cross-validation in target encoding
The idea is as follows, suppose we have a categorical feature $x$ with samples $x_{ic}$ ($i$-th sample of category $c$)

$$ \begin{pmatrix}
	u_{1c} \\
	\vdots \\
	u_{kc} \\ \\
	\hline\\
	v_{1c}\\
	\vdots \\
	v_{nc} \\
	\end{pmatrix}$$

So we divide the fold into a training and testing set $$\{u_i\}$$, $$\{v_j\}$$. We first calculate $IC(x)$ using $u_i$, and apply the encoding to the $v_j$'s. When trained using this method, the model will simulate the scenario during testing stage.

In practice, the training set inside each fold is further divided into more folds to include more noise to the encoding of each class. 

The implementation is as follows 

{% gist 89ecc153074a5b56edce9836de00c113 %}

(The code only works for a pandas dataframe now, if you have a suggestion to optimize this code or extend it to non-dataframe type objects. Please let me know!)


### Entity embedding

If you are a Kaggler like me, you would have discovered early on that neural networks don't work great on Kaggle problems. This is because categorical data don't have an apparent continuity built into them, and neural networks are only great at approximating continuous/piece-wise continuous functions.

Using one-hot encoding or target encoding can essentially avert this problem. However, as we noted before, one-hot encoding introduces excess dimensions and target encoding only takes the impact of the category towards the target into account. 

We will explore a method known as **Entity embedding** <a href='#2'> [2]</a>, which uses a neural network to learn an embedding for each categorical variable. 

The overall structure is as follows 

![structure](/assets/images/entity_embedding_structure.jpg)

The first layer is called the embedding layer, all categorical variables are converted into a one-hot vector $\delta_{x_i\alpha}$, it is then fed into a fully connected layer, whose weights will be treated as our embedding. So the output of our embedding layer looks like this

$$ x_i = \sum_{\alpha} W_{\alpha\beta}\delta_{x_i\alpha} $$

$\beta$ is the embedding lookup index. 

After the embedding layer, the output vectors are concatenated with continuous variables from the dataset, further feedforward network is then included. The output of this model is the prediction $\hat{y}$, weights can be updated via backpropagation minimizing an appropriate loss function (log loss for classification, MSE for regression etc.)

Below is an implementation in keras. 

{% gist 9764cceccca977168762e78d3ae79119 %}


### Some thoughts...

In other areas such as NLP and facial recognition, embeddings aren't acquired this way. Specifically, embeddings are learned by exploting the internal structure of the features within a dataset, ***not*** by its impact on a target variable. So these embeddings/encodings are learned by means of unsupervised learning (Word2Vec, FastText, PCA, Autoencoder, etc.).

The approaches we have introduced are all problem specific, meaning the embeddings/encodings learned are likely not ready to be reused in a different set of problem (with the potential exception of entitiy embedding). Perhaps one could generalize the architectures to allow unsupervised learning in categorical embeddings too, we shall explore more on this in part 2... 


<!-- Part of the task of encoding categorical variables maximizing the resulting liklihood function $\Pr(y \mid f(c))$. In target encoding, we are maximizing the likelihood function of a binomial(multinomial) dsitribution for each category. Of course, the best estimate of $p_c$, the probability of category $c$ having a positive outcome, is given by 

$$ \tilde{p}_c = \frac{|\{y|y=1, \: x=k\}|}{|\{x| x=k\}|} $$

Note that in this encoding scheme, we have ignored all other variables at our disposal. If we were to include these information, it would be much more difficult to maximize the likelihood function. Instead, we approximate the optimal solution 

The general question of encoding a categorical variable, can be rephrased as finding a function $f:\mathbb{\Omega} \rightarrow V$ such that 

$$ f = \mathop{\arg\, \max}\limits_g \sum_{c\in C}	\Pr(y \mid g(\tilde{c}), x_1,\cdots, x_m) $$

$\tilde{c}$ is the one hot vector for category $c$, $V$ is a vector space, $\Omega$ is the sample space of the categorical variable $x$ and $x_i$ are other variables in the dataset.  -->

### References

<div id='1'>
	[1] Micci-Barreca. A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems.
</div>

<div id='2'>
	[2] Cheng Guo, Felix Berkhahn. Entity Embeddings of Categorical Variabl. arXiv:1604.06737, 2016.
</div>