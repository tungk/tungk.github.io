+++ 
draft = false
date = 2024-09-16T03:09:12+02:00
title = "Combining One-Class Classifiers via Meta Learning"
description = ""
slug = ""
authors = []
tags = []
categories = []
externalLink = ""
series = []
+++

## Abstract

Selecting the best classifier among the available ones is a difficult task, especially when only instances of one class exist. In this work we examine the notion of combining one-class classifiers as an alternative for selecting the best classifier. In particular, we propose two one-class classification performance measures to weigh classifiers and show that a simple ensemble that implements these measures can outperform the most popular one-class ensembles. Furthermore, we propose a new one-class ensemble scheme, **TUPSO**, which uses meta-learning to combine one-class classifiers. Our experiments demonstrate the superiority of **TUPSO** over all other tested ensembles and show that the **TUPSO** performance is statistically indistinguishable from that of the hypothetical best classifier.

## Introduction

The one-class classification task is of particular importance to information retrieval tasks. Consider, for example, trying to identify documents of interest to a user, where the only information available is the previous documents that this user has read (i.e. positive examples), yet another example is citation recommendation, in which the system helps authors in selecting the most relevant papers to cite, from a potentially overwhelming number of references.

While there are plenty of learning algorithms to choose from, identifying the one that performs best in relation to the problem at hand is difficult. This is because evaluating a one-class classifier’s performance is problematic. By definition, the data collections only contain one-class examples and thus, performance metrics, such as false-positive (*FP*), and true negative (*TN*), cannot be computed. In the absence of *FP* and *TN*, derived performance metrics, such as classification accuracy, precision, among others, cannot be computed. Moreover, prior knowledge concerning the classification performance on some previous tasks may not be very useful for a new classification task because classifiers can excel in one dataset and fail in another, i.e., there is no consistent winning algorithm.

In this paper we search for a new method for combining one-class classifiers. We begin by presenting two heuristic methods to evaluate the classification performance of one-class classifiers. We then introduce a simple heuristic ensemble that uses these heuristic methods to select a single base-classifier. Later, we present **TUPSO**, a general meta-learning based ensemble, roughly based on the Stacking technique and incorporates the two classification performance evaluators. We then experiment with the discussed ensemble techniques on forty different datasets. The experiments show that **TUPSO** is by far the best option to use when multiple one-class classifiers exist. Furthermore, we show that **TUPSO**’s classification performance is strongly correlated with that of the actual best ensemble-member.

## Background

### One-Class Ensemble

The main motivation behind the ensemble methodology is to weigh several individual classifiers and combine them to obtain a classifier that outperforms them all. 

Compared to supervised ensemble learning, progress in the one-class ensemble research field is limited. Specifically, the Fix-rule technique was the only method which was considered for combining one-class classifiers. In this method, the combiner regards each participating classifier’s output as a single vote upon which it applies an aggregation function (a combining rule), to produce a final classification. In the following few years, further research was carried out and presently there are several applications reaching domains, such as information security (intrusion detection), remote sensing, image retrieval, image segmentation, on-line signature verification, and fingerprint matching.

Fixed-rule ensemble techniques, however, are not optimal as they use combining rules that are assigned statically and independently of the training data. As a consequence, as we will show later, the fixed rule ensembles produce inferior classification performance in comparison to the best classifier in the ensemble.

In the following lines we use the notation $P_k(x|\omega_{Tc})$ as the estimated probability of instance $x$ given the target class $\omega_{Tc}$, $fr_{(T,k)}$ as the fraction of the target class, which should be accepted for classifier $k = 1, \ldots , R$, $N$ as number of features, and $\theta_k$ notates the classification threshold for classifier $k$. A list of fixed combining rules is presented.


* Majority voting $\displaystyle y(x) = I_{>k/2}(\sum_k(I(P_k(x|\omega_{Tc})\ge\theta_k))$
* Mean vote $\displaystyle y(x) = \frac{1}{R}\sum_{k=1}^RI(P_k(x|\omega_{Tc})\ge\theta_k)$
* Weighted mean vote $\displaystyle y(x) = \frac{1}{R}\sum_{k=1}^R[f_{T,k}I(P_k(x|\omega_{Tc})\ge\theta_k) + (1-f_{T,k})I(P_k(x|\omega_{Tc})\ge\theta_k)]$
* Avg. rule $\displaystyle y(x) = \frac{1}{R}\sum_{k=1}^RP_k(x|\omega_{Tc})$
* Max rule $\displaystyle y(x) = \mathrm{arg}\max_k[P_k(x|\omega_{Tc})]$
* Product rule $\displaystyle y(x) = \prod_{k=1}^R[P_k(x|\omega_{Tc})]$
* Exclusive voting $\displaystyle y(x)=I_1(\sum_kI(P_k(x|\omega_{Tc})\ge\theta_k))$
* Weighted vote product $\displaystyle y(x) = \frac{\prod_{k=1}^R[fr_{(T,k)}I(P_k(x|\omega_{Tc})\ge\theta_k)]}{\prod_{k=1}^R[f_{(T,k)}I(P_k(x|\omega_{Tc})\ge\theta_k)] + \prod_{k=1}^R[(1-f_{(T,k)})I(P_k(x|\omega_{Tc})\ge\theta_k)]}$

Instead of using the fix-rule (e.g., weighting methods), technique to combine one-class classifiers, the meta-learning approach can be used.

### Meta Learning

Meta-learning is the process of learning from basic classifiers (ensemble members); the inputs of the meta-learner are the outputs of the ensemble-member classifiers. The goal of meta-learning ensembles is to train a meta-model (meta-classifier), which will combine the ensemble members’ predictions into a single prediction. 

To create such an ensemble, both the ensemble members and the meta-classifier need to be trained. Since the meta-classifier training requires already trained ensemble members, these must be trained first. The ensemble members are then used to produce outputs (classifications), from which the meta-level dataset (meta-dataset) is created. The basic building blocks of meta-learning are the meta-features, which are measured properties of the ensemble members output, e.g., the ensemble members’ predictions. A vector of meta-features and classification $k$ comprise a meta-instance, i.e., meta-instance $<f^{meta}_1, \ldots , f^{meta}_k, y >$, where $y$ is the real classification of the meta-instance that is identical to the class of the instance used to produce the ensemble members’ predictions. A collection of meta-instances comprises the meta-dataset upon which the meta-classifier is trained.

## Estimating the Classification Quality

$accuracy = (TP + \mathbf{TN})/(TP + \mathbf{TN} + \mathbf{FP} + FN)$, $Precision = TP/(TP + \mathbf{FP})$ and $F-score = 2 * \mathbf{P} * R/(\mathbf{P} + R)$, where $P$ is precision and $R$ is recall cannot be computed. Instead of computing the aforementioned metrics, we propose heuristic methods for estimating, rather than actually measuring, the classifier’s accuracy and F-score, respectively. Next, we describe the two performance estimators.

### Heuristic based Classification Performance Measures

By rewriting the error probability, one can estimate the classification error-rate, in the one-class paradigm, given a prior on the target-class:

$$
Pr[f(x) \ne y] = Pr[f(x) = 1] - Pr[Y = 1] + 2Pr[f(x) = 0|Y = 1]Pr[Y = 1]
$$

where $f(x)$ is the classifier’s classification result for the examined example $x$, $Pr[f(x) = 1]$ is the probability that the examined classifier will classify *Positive*, $Pr[f(x) = 0|Y = 1]$ is the probability that the classifier will classify *Negative* when given a *Positive* example, and lastly, $Pr[Y = 1]$ is the prior on the target-class probability.

Naturally, we define the one-class accuracy (OCA), estimator as follows: $OCA = 1 - Pr[f(x) \ne y]$. Note that the probabilities $Pr[f(x) = 1]$ and $[f(x) = 0|Y = 1]$ can be estimated for any one-class problem at hand using a standard cross-validating procedure.

An additional performance criteria, $\frac{r^2}{Pr[f(x)=1]}$, denoted as One-Class F-score (OCF), is given in. Using this criteria, one can estimate the classifier’s F-score in the semi-supervise paradigm. However, when only positive-labeled instances exist, the recall, $r = Pr[f(x) = 1|y = 1]$, equals to $Pr[f(x) = 1]$ (because $Pr[y = 1] = 1$), which only measures the fraction of correct classifications on positive test examples, i.e., true-positive rate (TPR). Using the TPR to measure the classification performance makes sense, because the TPR is strongly correlated with the classification accuracy when negative examples are very rare, such as in the case of most one-class problems.

### Best-Classifier By Estimation

Using the discussed classification performance estimators, we define a new and very simple ensemble: Estimated Best-Classifier Ensemble (ESBE). This ensemble is comprised of an arbitrary number of one-class ensemble-members (classifiers). During the prediction phase, the ensemble’s output is determined by a single ensemble-member, denoted as the dominant classifier. The ensemble’s dominant member is selected during the training phase. This is achieved by evaluating the performance of the participating ensemble-members using a $5\times 2$ cross-validation procedure, as described in. During this procedure only the training-set’s instances are used, and the metric used to measure the ensemble-members‘ performance is either OCA or OCF.

## TUPSO

![](https://i.imgur.com/DxMXjrY.png)

Figure 1: The **TUPSO** ensemble scheme.

The main principle of **TUPSO** is combining multiple and possibly diverse one-class classifiers using the meta-learning technique. **TUPSO** is roughly based on the Stacking technique, and as so, it uses a single meta-classifier to combine the ensembles‘ members. As opposed to Stacking, however, where the meta-classifier trains directly from the ensemble-members‘ outputs, **TUPSO**’s meta-classifier trains on a series of aggregations from the ensemble-members‘ outputs. To elevate the effectiveness of some of the aggregations used by **TUPSO**, and with that improve the ensemble’s over-all performance, during the training phase, the ensemble-members are evaluated using the aforementioned one-class performance evaluators. The performance estimates are then translated into static weights, which the meta-learning algorithm uses during the training of the meta-classifier, and during the prediction phases. 

The **TUPSO** ensemble, as shown in Figure 1, is made up of four major components: (1) Ensemble-members, (2) Performance evaluator, (3) Meta-features extractor, and (4) Meta-classifier. Next, we describe each component.

### Ensemble Members 

In **TUPSO**, the ensemble members are one-class, machine-learning-based, classifiers. **TUPSO** regards its ensemble members as black boxes, in order to avoid any assumption regarding their inducing algorithm, data structures or methods for handling missing values and categorical features. During the ensemble’s training phase, the ensemble-members are trained several times, as part of a cross-validation process, which is required for generating the meta-classifier’s dataset.

### Performance Evaluator 

The Performance Evaluator estimates the ensemble members’ classification performance during the ensemble’s training phase. To fulfill its task, the Performance Evaluator uses one of the available classification performance estimators, i.e., OCA or OCF. 

### Meta-Features Extractor 

The meta-features are measured properties of one or more ensemble-members’ output. A collection of meta features for a single instance makes a meta-instance. A collection of meta-instances is called a meta-dataset. The meta-dataset is used to train the meta-classifier. The Meta Features Extractor computes the meta-features by using multiple aggregations of the ensemble-members‘ output. Let $P_m = < p(m_1), \ldots, p(m_k) >$ be the vector containing the ensemble-members‘ outputs $p(m_1), \ldots , p(m_k)$, where $k$ is the number of members in the ensemble. A set of aggregate features is computed for each instance in the training set. A single set makes a single meta-instance, which will later be used either as a training instance for the meta-learner or as a test meta-instance.

* Sum-Votes $\displaystyle f_1(P_m) = \sum_{k=1}^k1_{\{p_{m_i} \ge 0.5\}}(P_{m_i})$
* Sum-Predictions $\displaystyle f_2(P_m) = \sum_{k=1}^k P_{m_i}$
* Sum-Weighted-Predictions $\displaystyle f_3(P_m) = \sum_{k=1}^k \alpha_i * P_{m_i}$
* Sum-Power-Weighted-Predictions $\displaystyle f_4(P_m) = \sum_{k=1}^k \alpha_i * (P_{m_i})^2$
* Sum-Log-Weighted-Predictions $\displaystyle f_5(P_m) = \sum_{k=1}^k \alpha_i * \log(P_{m_i})$
* Var-Votes $\displaystyle f_6(P_m) = \textrm{VAR}(1_{\{p_{m_i} \ge 0.5\}}(P_{m_i}))$
* Var-Predictions $\displaystyle f_7(P_m) =  \textrm{VAR}(P_{m_i})$
* Var-Weighted-Predictions $\displaystyle f_8(P_m) = \textrm{VAR}(\alpha_i * P_{m_i})$

The aggregate functions $f_2 \ldots f_5$ and $f_6 \ldots f_8$ are based on the first and second moments, respectively. The first moment computes the“average”ensemble-members‘ prediction, whereas the second moment computes the variability among the ensemble-members‘ predictions. The first moment based aggregation, a subtle version of the mean voting rule, is motivated by Condorcet’s Jury Theorem, and is used in several supervised-learning ensembles, e.g., Distribution-Summation [4]. Furthermore, the second moment based aggregation is motivated by the knowledge it elicits over the first moment, i.e., the level of consent among the ensemble-members. From this information, unique high-level patterns of ensemble members’ predictions can be learned by the meta-learner, and thereafter be at the disposal of the meta-classifier.

![](https://i.imgur.com/9wmwMLG.png)

Table 3: The training-set of the meta-classifier. Each column represents one aggregate feature over the ensemble members’ predictions and $ma_{i,j}$ denotes the value of meta-feature $j$ for meta-instance $i$.

## Meta-Classifier 

The meta-classifier is the ensemble’s combiner, thus, it is responsible for producing the ensemble’s prediction. Similar to the ensemble-members, the meta-classifier is a one-class classifier; it learns a classification model from meta-instances, which consist of meta-features. Practically, the meta-features used in training the meta-classifier can be either aggregate features, raw ensemble-members‘ predictions or their combination. However, preliminary experiments showed that training the meta-classifier using the raw ensemble-members‘ predictions alone or alongside the aggregate meta-features yielded less accurate ensembles. 

### Training Process 

The training process of **TUPSO** begins with training the ensemble-members followed by training the meta-classifier. The ensemble-members and the meta-classifier are trained using an inner $k$-fold cross-validation training process. First, the training-set is partitioned into k splits. Then, in each fold, the ensemble-members are trained on $k-1$ splits. Afterwards, the trained ensemble-members classify the remaining split to produce the instances for training the meta-classifier. The meta-instances in each fold are added to a meta-dataset. After k iterations, the meta-dataset will contain the same amount of instances as the original dataset. Lastly, the ensemble-members are re-trained using the entire training-set and the meta-classifier is trained using the meta-dataset. 

### Weighting the Ensemble Members

In order to calculate certain meta-features, e.g., $f_3$, the ensemble-members‘ predictions have to be weighed. To do so, a set of weights, one per ensemble-member, are learned as part of the ensemble training process. During the meta-classifier training, the ensemble-members predict the class of the evaluated instances. The predictions are fed to the Performance Evaluator, which calculates either OCA or OCF estimations for each of the ensemble-members, $Perf_{vect} =< Perf_1, \ldots , Perf_m >$, where $Perf_i$ is the estimated performance of ensemble-member $i$. Finally, a set of weights, $\alpha_1, \alpha_2, \ldots , \alpha_m$, is computed as follows:

$$
\alpha_i = \frac{Perf_i}{\sum_{j=1}^m Perf_j}, \forall i = 1 \ldots m
$$

## Method

### One-Class Learning Algorithms 

For evaluation purposes, we made use of four, one-class algorithms: OC-SVM [15], OC-GDE, OC-PGA, and ADIFA [14]. We selected these ensemble-members because they represent the prominent families of one-class classifiers, i.e., nearest-neighbor (OC-GDE, OC-PGA), density (ADIFA), and boundary (OC-SVM). The first two algorithms are our adaptations of two well-known supervised algorithms to one class learning. 

We used a static pool of six ensemble-members for all the evaluated ensembles: (i) ADIFA~HM~, (ii) ADIFA~GM~, (iii) OC-GDE, (iv) OC-PGA, OC-SVM~1~, and (vi) OC-SVM~2~.

### Ensemble Combining Methods 

The following evaluation includes several ensemble combining methods from three groups of algorithms: Heuristic-Ensemble: estimated best-classifier ensemble (ESBE); Fixed-rules: majority voting, mean-voting, max-rule and product-rule; and Meta-learning-based: TUPSO. The learning algorithm used for inducing the meta-classifier in TUPSO was ADIFA, as it outperformed the other three mentioned learning algorithms on the evaluation set. 

### Datasets 

During the evaluation we used a total of 40 distinct datasets from two different collections, UCI and KDD-CUP99. All datasets are fully labeled and binary-class. We selected 34 datasets from the widely used UCI dataset repository [3]. The datasets vary across dimensions, number of target classes, instances, input features, and feature type (nominal or numeric). So as to have only two classes in the UCI datasets, a pre-process was completed where only the instances of the two most prominent classes were selected. The other instances were filtered out. The KDD CUP 1999 dataset contains a set of instances that represent connections to a military computer network. The dataset contains 41 attributes, 34 of which are numerical and 7 of which are categorical. The original dataset contained 4,898,431 multi-class data instances. In order to divide the dataset into multiple binary-class sets, we followed the method performed in [20]. Compared with the UCI datasets, the KDD99-CUP are much more natural one-class datasets, as they are highly imbalanced (instances of the network’s normal state make the lion’s share of the de-rived binary datasets).

### Evaluation Methodology 

During the training phase, only the examples of one-class were available to the learning algorithms and to the classification performance estimators. During the testing phase, however, both positive and negative examples were available, to evaluate the classifiers in real-life conditions. The generalized classification accuracy was measured by performing a 5x2 cross-validation procedure [6]. We used the area under the ROC curve (AUC) metric to measure the classification performance of the individual classifiers and ensemble methods.

![](https://i.imgur.com/NUYsE0T.png)

Table 5: Ensembles classification AUC results.

![](https://i.imgur.com/wazcGny.png)

Figure 2: Classification performance: ensembles vs. actual best classifier.

