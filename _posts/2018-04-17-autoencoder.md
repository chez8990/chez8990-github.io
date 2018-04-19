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
Data that appears in nature are usually very high dimensional but sparse, we know from experience that these data are not well suited to be used straight away for machine learning. 
By using an autoencoder as a dimenison reduction estimator, one can in theory compress these data into any desired dimension. 

If the dimension of the latent space $$ m \in \{2, 3\} $$, we can even visualize the latent space and get a glimse on the geometry of the data, which helps us understand our data even better. 

More importantly, the latent space representation is ***dense***, which means we can traverse along different trajectories within the latent space, and interpret the results through the decoder. One such example is linear interpolation between two points in the latent space.

Given input vector $$ X_1 $$, $$ X_2 $$ the line segment between latent vectors $$ Z_1 $$ and $$ Z_2 $$ is as follows

$$ Z_t = Z_1(1-t) + Z_2\:t \quad t\in [0,1] $$

We can then interpret this equation through the decoder output

$$ X^\prime_t = f^{-1}(Z_t) \quad t\in [0,1]$$

{: .center}
![face_latent](/assets/images/face_latent.jpeg)

There's just one more perk autoencoder brings to the table, it allows us to sample from the data's distribution in the latent space and create more data, this is especially important if we are lacking in data quantity. 

### Implementation in Python with Keras

We will now build an autoencoder with Keras and demonstrate it with the MNIST dataset.

The network architecture goes as follows 
<table>
	<tbody>
		<tr>
			<td>&nbsp;<strong>Layer</strong></td>
			<td><strong>Output shape</strong>&nbsp;</td>
			<td><strong>Parameter number</strong>&nbsp;</td>
		</tr>
		<tr>
			<td>&nbsp;Input</td>
			<td>&nbsp;(None, 1, 784)</td>
			<td>&nbsp;0</td>
		</tr>
		<tr>
			<td>Dense&nbsp;</td>
			<td>(None, 1, 512)</td>
			<td>401920&nbsp;</td>
		</tr>
		<tr>
			<td>Dense&nbsp;</td>
			<td>&nbsp;(None, 1, 256)&nbsp;</td>
			<td>131328&nbsp;</td>
		</tr>
		<tr>
			<td>Dense&nbsp;</td>
			<td>&nbsp;(None, 1, 64)&nbsp;&nbsp;</td>
			<td>16448&nbsp;</td>
		</tr>
		<tr>
			<td>Dense&nbsp;</td>
			<td>&nbsp;(None, 1, 2)</td>
			<td>130&nbsp;</td>
		</tr>
		<tr>
			<td>Dense&nbsp;</td>
			<td>&nbsp;(None, 1, 64)</td>
			<td>192&nbsp;</td>
		</tr>
		<tr>
			<td>Dense&nbsp;</td>
			<td>&nbsp;(None, 1, 256)</td>
			<td>16640&nbsp;</td>
		</tr>
		<tr>
			<td>Dense&nbsp;</td>
			<td>&nbsp;(None, 1, 512)</td>
			<td>131584&nbsp;</td>
		</tr>
		<tr>
			<td>Dense&nbsp;</td>
			<td>&nbsp;(None, 1, 784)</td>
			<td>402192&nbsp;</td>
		</tr>
	</tbody>
</table>

Notice the number of filters evolves in the way described above, the input dimension is 784 = 28 x 28 as we have flatten the image, the latent dimension is 2 for easy visualization of the latent space.

{% gist 295f40d5a8532683d6c149d6cff46472 %}

We can see that the latent space structure has been learned 

### Drawbacks of autoencoders

Autoencoders could end up learning nothing but simply copy and paste the input as the output, this is the eventual destiny of any vanilla autoencoder, since it's approximating the identity function; this will be harmful if our goal was to generate more data from our sample (time inefficient and quality deficient). Moreover, autoencoders trained this way are sensitive to the input, any distortion to the data (noise, rotation, reflection) could detriment the model's ability to generate correct output. 

(include examples)


Are there ways that we can ensure variety, stability and quality of the generated data in an autoencoder? Go to [Other types of autoencoders](/) to see more.