---
layout: post
title: Automatic Text Summarization
---

### Overview of automatic text summarization
According to Radef et al <a href='#1'>[1]</a>. A summary is defined as “**a text that is produced from one or more texts, that conveys important information in the original text(s), and that is no longer than half of the original text(s) and usually, significantly less than that**”. 

There are numerous applications developed on automatic text summarization, such as headline generation for news, search engine snippets generation and financial news summarization. 

There are two different approaches to text summarization: ***extractive*** and ***abstractive***. 
<ul>
	<li> Extractive methods focus on identifying and selecting important sections of the text, and presenting them in a grammatically sensible ways</li>
	<li> Abstractive methods aim at creating new texts that converys the most critical information from the original texts. This is done by using advance natural language processing techinques.</li>
</ul>

Current research focuses on extractive methods rather than abstractive methods, due to the complication of abstractive methods arising from semantic representation and other NLP areas. 

### Extractive summarization

Extractive methods comprise of three stages
<ol>
	<li> Construct a representation of the input text</li>
	<li> Score the sentences according to a defined metric based on the representations </li>
	<li> Select a summary with n sentences base on the scores </li>
</ol>

There are a four main category of methods available. 
<ul>
	<li> Frequency based </li>
	<li> Topic keywords </li>
	<li> Latent semantic analysis </li>
	<li> Bayseian topic modeling </li>
</ul>

We will primarily focus on the the latter two since they are more interesting and useful in reality. The former two are relatively straight forward and are easy to understand, so we will simply skip those.

### TFIDF
I lied about completely skipping the frequency approach, as I do need to introduce TFIDF in order for me to talk about latent semantic analysis (LSA)

TFIDF stands for Term Frequency Inverse Document Frequency, it is weighting method for words in a document. For a word $w$ in a document $d$ in collection of documents $D$, its TFIDF weight is given as follows

$$ q_d(w) = f_d(w) \log\frac{|D|}{|\{d\in D| w\in D\}|} $$

Where $f_d(w)$ is the occurance frequency of $w$ in $d$. The important take away here is, words that appear frequently across $D$ gets weighted less inside a sentence, so stuff like stop-words would have a lot less weighting then say the phrase "golden shower" (#RealDonaldTrump) in any setting. 

### Latent Semantic Analysis
LSA is a way of decomposing the term-document matrix arising from TFIDF calculations, into components that are lower in rank. Since even a modest sized document could be composed of tens of thousand words, the resulting term-document matrix $C$ could be extremely large as well. LSA seeks to find a low-rank approximator $C_k$ to $C$ by means of singular value decomposition, where $k$ is a predetermined dimension much smaller than the number of words $n$ in the document.

How we do this mathematically, is by singular value decomposition on $C$. Formally, this is stated as follows 

$$ C = U\Sigma V^T$$

Where $U$ is comprised of the eigenvectors of $CC^T$, $V$ is that of $C^TC$ and $\Sigma$ is a diagonal matrix with entries being the singular values of $C$.



### References

<div id='1'>
	[1] z28, 4 (2002), 399–408.
</div>