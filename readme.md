# Naïve Bayes Classifier

This project is a coursework for the Artificial Intelligence module of first year Computer Science at the University of Bath.

For my implementation of the classifier, I have chosen to use the Naïve Bayes approach as the solution is both satisfyingly elegant and produces high accuracy results.

## Overview of the Classifier

The underlying principle that the classifier is based on is Bayes Theorem. 

- Bayes Theorem says that if event X has occured, then we can find the probability of event Y given event X
- **P(Y | X)**

The problem with Bayes Theorem is that for an n-dimensional vector called X, there are 2^X possible combinations for it. And, for a binary variable Y, there are 2^(n+1) combinations for P(X1, X2, ... , Xn | Y).

In our classifier, we have 54-dimensional vector called message and two options for our classifier in the spam filtering section. This would result in 2^(54+1) combinations for P(message | Class = class) and is completely unrealisitic for real-world applications.

However, we can use Naïve Bayes to solve this problem. The main idea behind Naïve Bayes is to assume that each input vector is conditionally independent of other features. In our spam filtering example, we assume conditional independence of keywords (from the feature set) given the class.

- This means that we assume the words in a message were drawn independently from a bag of k different words (where k = 54 in our case)

As a result of applying Naïve Bayes, we can reduce the number of values we have to find from 2^(n+2) to (2n+2) which drastically improves the efficiency of the process, making the algorithm feasible.

- Where n is the dimensionality of the feature vector (also known as the query vector)

Using Naïve Bayes, we can get the probability of a class given a feature vector. We can then compare the probabilities of each class and choose the class that has the greatest probability to be the one classified for the given feature vector. The rule for choosing the class is know as the `maximum a posterior` decision rule.

The classifiers used in this assignment are the same for the Spam Filtering and Digit Classification sections as I made the class 'MyClassifier' generalised such that it can perform the machine learning for any number of classes. This is enabled as a result of the assumption that the data is modelled by a multinomial distritution. _To change the number of classes which are to be identified, pass the an argument 'k' with an integer value denoting the number of classes when you instantiate the MyClassifier class._

## Part 1 - Spam Filtering

## Part 2 - Digit Classification and Feature Engineering
