---
layout:     post
title:      How to train your Deep Neural Network
date:       2017-01-05 9:00:05
summary:    List of commonly used practices for efficient training of Deep Neural Networks.
categories: ml
comments: true
---

There are certain practices in **Deep Learning** that are highly recommended, in order to efficiently train **Deep Neural Networks**. In this post, I will be covering a few of these most commonly used practices, ranging from importance of quality training data, choice of hyperparameters to more general tips for faster prototyping of DNNs. Most of these practices, are validated by the research in academia and industry and are presented with mathematical and experimental proofs in research papers like [Efficient BackProp(Yann LeCun et al.)](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) and [Practical Recommendations for Deep Architectures(Yoshua Bengio)](https://arxiv.org/pdf/1206.5533v2.pdf).

As you'll notice, I haven't mentioned any mathematical proofs in this post. All the points suggested here, should be taken more of a summarization of the best practices for training DNNs. For more in-depth understanding, I highly recommend you to go through the above mentioned research papers and references provided at the end.

----

### Training data

A lot of ML practitioners are habitual of throwing raw training data in any **Deep Neural Net(DNN)**. And why not, any DNN would(presumably) still give good results, right? But, it's not completely old school to say that - "given the right type of data, a fairly simple model will provide better and faster results than a complex DNN"(although, this might have exceptions). So, whether you are working with **Computer Vision**, **Natural Language Processing**, **Statistical Modelling**, etc. try to preprocess your raw data. A few measures one can take to get better training data:

* Get your hands on as large a dataset as possible(DNNs are quite data-hungry: more is better)
* Remove any training sample with corrupted data(short texts, highly distorted images, spurious output labels, features with lots of null values, etc.)
* Data Augmentation - create new examples(in case of images - rescale, add noise, etc.)

<!-- ### Normalize input vectors

A Deep learning model can converge much faster, if the empirical mean of input vectors lies near `0`. [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) elucidated this in detail. Basically, it boils down to the average polarity(positive/negative) of the product of input vector - `x` and weight -  `W`. Hence, during **backpropagation** all of these weights will either decrease or increase; consequently, loss will be optimized in a zig-zag fashion. -->

### Choose appropriate activation functions

One of the vital components of any Neural Net are [activation functions](https://en.wikipedia.org/wiki/Activation_function). **Activations** introduces the much desired **non-linearity** into the model. For years, `sigmoid` activation functions have been the preferable choice. But, a `sigmoid` function is inherently cursed by these two drawbacks - 1. Saturation of sigmoids at tails(further causing [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)). 2. `sigmoids` are not zero-centered. 

A better alternative is a `tanh` function - mathematically, `tanh` is just a rescaled and shifted `sigmoid`, `tanh(x) = 2*sigmoid(x) - 1`.
 Although `tanh` can still suffer from the **vanishing gradient problem**, but the good news is - `tanh` is zero-centered. Hence, using `tanh` as activation function will result into faster convergence. I have found that using `tanh` as activations generally works better than sigmoid.

 You can further explore other alternatives like [`ReLU`](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)), `SoftSign`, etc. depending on the specific task, which have shown to ameliorate some of these issues.


### Number of Hidden Units and Layers

Keeping a larger number of hidden units than the optimal number, is generally a safe bet. Since, any regularization method will take care of superfluous units, at least to some extent. On the other hand, while keeping smaller numbers of hidden units(than the optimal number), there are higher chances of underfitting the model. 

Also, while employing **unsupervised pre-trained representations**(describe in later sections), the optimal number of hidden units are generally kept even larger. Since, pre-trained representation might contain a lot of irrelevant information in these representations(for the specific supervised task). By increasing the number of hidden units, model will have the required flexibility to filter out the most appropriate information out of these pre-trained representations.

Selecting the optimal number of layers is relatively straight forward. As [@Yoshua-Bengio](https://www.quora.com/profile/Yoshua-Bengio) mentioned on Quora - "You just keep on adding layers, until the test error doesn't improve anymore". ;)

### Weight Initialization

Always initialize the weights with small `random numbers` to break the symmetry between different units. But how small should weights be? What's the recommended upper limit? What probability distribution to use for generating random numbers? Furthermore, while using `sigmoid` activation functions, if weights are initialized to very large numbers, then the sigmoid will **saturate**(tail regions), resulting into **dead neurons**. If weights are very small, then gradients will also be small. Therefore, it's preferable to choose weights in an intermediate range, such that these are distributed evenly around a mean value.

Thankfully, there has been lot of research regarding the appropriate values of initial weights, which is really important for an efficient convergence. 
To initialize the weights that are evenly distributed, a `uniform distribution` is probably one of the best choice. Furthermore, as shown in the [paper(Glorot and Bengio, 2010)](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf), units with more incoming connections(fan_in) should have relatively smaller weights. 

Thanks to all these thorough experiments, now we have a tested formula that we can directly use for weight initialization; i.e. - weights drawn from `~ Uniform(-r, r)` where `r=sqrt(6/(fan_in+fan_out))` for `tanh` activations, and `r=4*(sqrt(6/fan_in+fan_out))` for `sigmoid` activations, where `fan_in` is the size of the previous layer and `fan_out` is the size of next layer.

### Learning Rates

This is probably one of the most important hyperparameter, governing the learning process. Set the learning rate too small and your model might take ages to converge, make it too large and within initial few training examples, your loss might shoot up to sky. Generally, a learning rate of `0.01` is a safe bet, but this shouldn't be taken as a stringent rule; since the optimal learning rate should be in accordance to the specific task. 

In contrast to, a fixed learning rate, gradually decreasing the learning rate, after each epoch or after a few thousand examples is another option. Although this might help in faster training, but requires another manual decision about the new learning rates. Generally, **learning rate can be halved after each epoch** - these kinds of strategies were quite common a few years back. 

Fortunately, now we have better `momentum based methods` to change the learning rate, based on the curvature of the error function. It might also help to set different learning rates for individual parameters in the model; since, some parameters might be learning at a relatively slower or faster rate.

Lately, there has been a good amount of research on optimization methods, resulting into `adaptive learning rates`. At this moment, we have numerous options starting from good old `Momentum Method` to `Adagrad`, `Adam`(personal favourite ;)), `RMSProp` etc. Methods like `Adagrad` or `Adam`, effectively save us from manually choosing an `initial learning rate`, and given the right amount of time, the model will start to converge quite smoothly(of course, still selecting a good initial rate will further help).


### Hyperparameter Tuning: Shun Grid Search - Embrace Random Search

**Grid Search** has been prevalent in classical machine learning. But, Grid Search is not at all efficient in finding optimal hyperparameters for DNNs. Primarily, because of the time taken by a DNN in trying out different hyperparameter combinations. As the number of hyperparameters keeps on increasing, computation required for Grid Search also increases exponentially.

There are two ways to go about it:
 1. Based on your prior experience, you can manually tune some common hyperparameters like learning rate, number of layers, etc. 
 2. Instead of Grid Search, use **Random Search/Random Sampling** for choosing optimal hyperparameters. A combination of hyperparameters is generally choosen from a **uniform distribution** within the desired range. It is also possible to add some prior knowledge to further decrease the search space(like learning rate shouldn't be too large or too small). Random Search has been found to be way more efficient compared to Grid Search.

### Learning Methods

Good old **Stochastic Gradient Descent** might not be as efficient for DNNs(again, not a stringent rule), lately there have been a lot of research to develop more flexible optimization algorithms. For e.g.: `Adagrad`, `Adam`, `AdaDelta`, `RMSProp`, etc. In addition to providing **adaptive learning rates**, these sophisticated methods also use **different rates for different model parameters** and this generally results into a smoother convergence. It's good to consider these as hyper-parameters and one should always try out a few of these on a subset of training data.


### Keep dimensions of weights in the exponential power of 2

Even, when dealing with **state-of-the-art** Deep Learning Models with latest hardware resources, **memory management** is still done at the byte level; So, it's always good to keep the size of your parameters as `64`, `128`, `512`, `1024`(all powers of `2`). This might help in sharding the matrices, weights, etc. resulting into slight boost in learning efficiency. This becomes even more significant when dealing with **GPUs**.


### Unsupervised Pretraining

Doesn't matter whether you are working with NLP, Computer Vision, Speech Recognition, etc. **Unsupervised Pretraining** always help the training of your supervised or other unsupervised models. **Word Vectors** in NLP are ubiquitous; you can use [ImageNet](http://image-net.org/) dataset to pretrain your model in an unsupervised manner, for a 2-class supervised classification; or audio samples from a much larger domain to further use that information for a speaker disambiguation model. 

### Mini-Batch vs. Stochastic Learning

Major objective of training a model is to learn appropriate parameters, that results into an optimal mapping from inputs to outputs. These parameters are tuned with each training sample, irrespective of your decision to use **batch**, **mini-batch** or **stochastic learning**. While employing a stochastic learning approach, gradients of weights are tuned after each training sample, introducing noise into gradients(hence the word 'stochastic'). This has a very desirable effect; i.e. - with the introduction of **noise** during the training, the model becomes less prone to overfitting. 

However, going through the stochastic learning approach might be relatively less efficient; since now a days machines have far more computation power. Stochastic learning might effectively waste a large portion of this. If we are capable of computing **Matrix-Matrix multiplication**, then why should we limit ourselves, to iterate through the multiplications of individual pairs of **Vectors**? Therefore, for greater throughput/faster learning, it's recommended to use mini-batches instead of stochastic learning.

But, selecting an appropriate batch size is equally important; so that we can still retain some noise(by not using a huge batch) and simultaneously use the computation power of machines more effectively. Commonly, a batch of `16` to `128` examples is a good choice(exponential of `2`). Usually, batch size is selected, once you have already found more important hyperparameters(by **manual search** or **random search**). Nevertheless, there are scenarios when the model is getting the training data as a stream([online learning](https://en.wikipedia.org/wiki/Online_machine_learning)), then resorting to Stochastic Learning is a good option.

### Shuffling training examples

This comes from **Information Theory** - "Learning that an unlikely event has occurred is more informative than learning that a likely event has occurred". Similarly, randomizing the order of training examples(in different epochs, or mini-batches) will result in faster convergence. A slight boost is always noticed when the model doesn't see a lot of examples in the same order. 

### Dropout for Regularization

Considering, millions of parameters to be learned, regularization becomes an imperative requisite to prevent **overfitting** in DNNs. You can keep on using **L1/L2** regularization as well, but **Dropout** is preferable to check overfitting in DNNs. Dropout is trivial to implement and generally results into faster learning. A default value of `0.5` is a good choice, although this depends on the specific task,. If the model is less complex, then a dropout of `0.2` might also suffice. 

Dropout should be turned off, during the test phase, and weights should be scaled accordingly, as done in the [original paper](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf). Just allow a model with Dropout regularization, a little bit more training time; and the error will surely go down.

### Number of Epochs/Training Iterations

"Training a Deep Learning Model for multiple epochs will result in a better model" - we have heard it a couple of times, but how do we quantify "many"?
Turns out, there is a simple strategy for this - Just keep on training your model for a fixed amount of examples/epochs, let's say `20,000` examples or `1` epoch. After each set of these examples compare the **test error** with **train error**, if the gap is decreasing, then keep on training. In addition to this, after each such set, save a copy of your model parameters(so that you can choose from multiple models once it is trained).

### Visualize 

There are a thousand ways in which the training of a deep learning model might go wrong. I guess we have all been there, when the model is being trained for hours or days and only after the training is finished, we realize something went wrong. In order to save yourself from bouts of hysteria, in such situations(which might be quite justified ;)) - **always visualize the training process**. Most obvious step you can take is to **print/save logs** of `loss` values, `train error` or `test error`, etc. 

In addition to this, another good practice is to use a visualization library to plot histograms of weights after few training examples or between epochs. This might help in keeping track of some of the common problems in Deep Learning Models like **Vanishing Gradient**, **Exploding Gradient** etc.


### Multi-Core machines, GPUs

Advent of GPUs, libraries that provide vectorized operations, machines with more computation power, are probably some of the most significant factors in the success of Deep Learning. If you think, you are patient as a stone, you might try running a DNN on your laptop(which can't even open 10 tabs in your Chrome browser) and wait for ages to get your results. Or you can play smart(and expensively :z) and get a descent hardware with at least **multiple CPU cores** and a **few hundred GPU cores**. GPUs have revolutionized the Deep Learning research(no wonder Nvidia's stocks are shooting up ;)), primarily because of their ability to perform Matrix Operations at a larger scale. 

So, instead of taking weeks on a normal machine, these parallelization techniques, will bring down the training time to days, if not hours.


### Use libraries with GPU and Automatic Differentiation Support

Thankfully, for rapid prototyping we have some really descent libraries like [Theano](http://deeplearning.net/software/theano/), [Tensorflow](https://www.tensorflow.org/), [Keras](https://keras.io/), etc. Almost all of these DL libraries provide **support for GPU computation** and **Automatic Differentiation**. So, you don't have to dive into core GPU programming(unless you want to - it's definitely fun :)); nor you have to write your own differentiation code, which might get a little bit taxing in really complex models(although you should be able to do that, if required). Tensorflow further provides support for training your models on a **distributed architecture**(if you can afford it).

This is not at all an exhaustive list of practices, to train a DNN. In order to include just the most common practices, I have tried to exclude a few concepts like Normalization of inputs, Batch/Layer Normalization, Gradient Check, etc. Although feel free to add anything in the comment section and I'll be more than happy to update it in the post. :)

### References:
1. [Efficient BackProp(Yann LeCun et al.)](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
2. [Practical Recommendations for Deep Architectures(Yoshua Bengio)](https://arxiv.org/pdf/1206.5533v2.pdf)
3. [Understanding the difficulty of training deep feedforward neural networks(Glorot and Bengio, 2010)](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
4. [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
5. [Andrej Karpathy - Yes you should understand backprop(Medium)](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b#.yd17cx8ml)

{% include disqus.html %}
