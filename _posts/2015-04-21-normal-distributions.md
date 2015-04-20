---
layout:     post
title:      Normal/Gaussian Distributions
date:       2015-04-21 04:00:01
summary:    This is first blog post of the series "Statistical Distributions". We are starting with the most commonly used Normal Distributions.
categories: statistics
comments: true
---

Normal Distributions are the most common distributions in statistics primarily because they describe a lot of natural phenomena. Normal distributions are also known as 'Gaussian distributions' or 'bell curve', because of the bell shaped curve.

![bell](../../../../../images/normal_distributions.png)

Samples of heights of people, size of things produced by machines, errors in measurements, blood pressure, marks in an examination, wages payed to employees by a company, life span of a species, all of these follows a normal or nearly normal distribution.

I don't intend to cover a lot of mathematical background regarding normal distributions, still it won't hurt to know just a few simple mathematical properties of normal distributions:

<blockquote>
	<ul>
		<li>Bell curve is symmetrical about mean(which lies at the center)
		<li>mean = median = mode
		<li>Only determining factors of normal distributions are its mean and standard deviation
	</ul>
</blockquote>

We can also get a normal distribution from a lot of datasets using [Central Limit Theorem](http://en.wikipedia.org/wiki/Central_limit_theorem)(CLT). In layman's language CLT states that if we take a large number of samples from a population, multiple times and go on plotting these then it will result in a normal distribution(which can be used by a lot of statistical and machine learning models).

A lot of machine learning models assumes that data fed to these models follows a normal distribution. So, after you have got your data cleaned, you should definitely check what distribution it follows. Some of the machine learning and Statistical models which assumes a normally distributed input data are:

<blockquote>
	<ul>
		<li>Gaussian naive Bayes
		<li>Least Squares based (regression)models
		<li>LDA
		<li>QDA
	</ul>
</blockquote>
It is also quite common to transform non-normal data to normal form by applying log, square root or similar transormations. 

If plotting the data results in a skewed plot, then it is probably a log-normal distribution(as shown in figure below), which you can transform into normal form, simply by applying a log function on all data points.

![log-normal](../../../../../images/log-normal.png)

Once it is transformed into normal distributions, you are free to use this dataset with models assuming a normal input data(as listed in above section). 

As a general approach, <b>Always look at the statistical/probability distiributions</b> as your first step in data analysis.

<!-- % if page.comments % -->

<div id="disqus_thread"></div>
<script type="text/javascript">
    /* * * CONFIGURATION VARIABLES * * */
    var disqus_shortname = 'rishabhshukla';
    
    /* * * DON'T EDIT BELOW THIS LINE * * */
    (function() {
        var dsq = document.createElement('script'); dsq.type = 'text/javascript'; dsq.async = true;
        dsq.src = '//' + disqus_shortname + '.disqus.com/embed.js';
        (document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(dsq);
    })();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript" rel="nofollow">comments powered by Disqus.</a></noscript>

<!-- % endif % -->