---
layout: post
title: Regression Model Selection - Statistics to the Rescue
author: Artan Zandian
date: Dec 07, 2021
excerpt: 'When it comes to Regression model selection the two main questions that arise are, first, which model fits the data the best, and second, which features within that model could be trimmed off because of no added value. This post tries to use statistics to answer the questions.'
---

When it comes to Regression model selection the two main questions that arise are, first, which model fits the data the best, and second, which features within that model could be trimmed off because of no added value. This post tries to use statistics to answer the questions.  

We will be breaking down the problem into two main categories depending on the availability of a test data, and then multiple scenarios within each catetory to discuss the potential solutions:

<br>

# Prediction - Test data is available
With access to the test data the question that we are trying to answer in a prediction scenario is whether all the features help predict the response of new test data or only a subset of features can do the work. To compare out-of-sample predictions from different models the preferred metric to use is a **cross-validation MSE** (Mean Squared Error). The reason we are using a cross-validation (or validation) set separate from the train data is that the set that has been used to select can not be used (to prevent data leakage) to evaluate the prediction of a selection model. This is a well-known practice and I won't go through details.

<br>

# Inference - Test data is not available
Since a cross-validation study is not always feasible, we use different measures to approximate the cross-validation MSE. However, depending on the scenario, we might even have better alternatives:  

<br>

## Models of the same size  
When using models of the same size on the same data, our best friend is the good old **Coefficient of determination** ($R^2$) defined as the amount of variation of the response that is explained by feature variables in a model:

$$R^2=1 - \frac{\text{RSS}}{\text{TSS}}$$

Where 
- $RSS$ is Residual Sum of Squares: $$RSS=\sum_{i=1}^n(y_i - \hat{y}_i)^2$$
- $TSS$ is Total Sum of Squares: $$TSS=\sum_{i=1}^n(y_i-\bar{y})^2$$

The most important thing to remember is that as the number of features in a model increase, $R^2$ increases. Therefore, this metric cannot be used for for comparing models of different sizes.

<br>

## Models of varying sizes - Nested
As the name suggests, “nested” means that one model is a subset of another. For example, assuming that we have four features in our base model, we could derive a nested model from any combination of 1-3 features from the original four.
### Nested ANOVA
Also called a hierarchical ANOVA (ANalysis Of VAriance) is an extension of a simple ANOVA for experiments where each group is divided into two or more random subgroups. Under the hood, this test uses **F-test** to see if there is any variation between groups, or within nested subgroups of the attribute variable. $p$ in the formula below is the number of variable.  

$$F=\frac{(TSS - RSS)/p}{RSS/(n-p-1)}$$

In R this could be evaluated using the base `anova()` function:

```R
anova(subset_model, full_model)
```  

> If the ultimate goal was to do *feature selection* by testing contribution of *individual features* on explaining the response one at a time, an alternative is performing a **t-test** which could be tested on the fitted models using the `tidy()` function from the `broom` package in R.
>```R
> broom::tidy(model, model_plus_one)
>```

<br>

## Models of varying sizes - Not Nested
This is probably the most prevalent scenario in our analysis of models, and therefore, we have more possible test flavours. For this scenario we want to discourage overfitting which is a key goal in model selection. We do this by penalizing inclusion of more features in the model. Here are the most commonly used tests for models of varying sizes:

<br>

### Adjusted $R^2$
As mentioned above $R^2$ is not a suitable metric when it comes to models of varying sizes. This is mainly due to the fact that $RSS$ decreases as more variables are included in the model. To overcome this problem we penalize $R^2$ by number of variables in the model ($p$).

$$ \text{adjusted } R^2 = 1- \frac{RSS/(n-p-1)}{TSS/(n-1)} $$

<br>

### Mallow's $C_p$
[Mallow's $C_p$](https://en.wikipedia.org/wiki/Mallows%27s_Cp) is built on the calculation of Oridinary Least Squares ([OLS](https://en.wikipedia.org/wiki/Ordinary_least_squares)) to address the issue of overfitting by also penalizing the number of features. 
The model with the smallest $C_p$ is the best model with this definition:

$$ C_{p}={\frac {1}{n}}(\operatorname {RSS} +2p{\hat {\sigma }}^{2})$$

Where:
- $p$ is the number of estimated features in the model
- $RSS$ is the residual sum of squares as described above
- ${\sigma }^{2}$ is an estimate of the variance associated with each response

<br>

### Akaike Information Criterion (AIC)
Another way of penalizing inclusion of more features in the model is [AIC](https://en.wikipedia.org/wiki/Akaike_information_criterion). This discourages overfitting which is key in model selection. We select that model with the smallest AIC.

AIC uses [information theory](https://en.wikipedia.org/wiki/Information_theory) and the [likelihood function](https://en.wikipedia.org/wiki/Likelihood_function) to make the comparison. The AIC value of the model is:
$$ AIC = 2p - 2\ln{(\hat{L})} $$  

where:
- $p$ is the number of estimated features in the model
- $\hat{L}$ is the maximum value of likelihood function for the model

<br>

### Bayesian information criterion (BIC)
[BIC](https://en.wikipedia.org/wiki/Bayesian_information_criterion) is closely related to AIC in a sense that both use the likelihood function and try to penalize the number of features (k). Models with lower BIC are preferred. The formal definition of BIC is as follows: 

$${BIC} =p\ln(n)-2\ln({\widehat {L}})$$
where:  
- $p$ is the number of estimated features in the model
- $\hat{L}$ is the maximum value of likelihood function for the model
- $n$ is the number of observations

<br>


## Conclusion and How-to

All the above mentioned test values for the models of varying sizes can be generated by feeding the fitted model to the `glance()` function in R. Note that we will compare the results of multiple models by calling `glance()` on each individutal model.  
```R
broom::glance(fitted_model)
```

<br>

<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/glance_results.PNG?raw=True"></center>

<br>


## References
