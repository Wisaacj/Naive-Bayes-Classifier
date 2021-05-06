# Naïve Bayes Classifier

This project is a coursework for the Artificial Intelligence module of first year Computer Science at the University of Bath.

For our implementation of the classifier, we have chosen to use the Naïve Bayes approach as the solution is both satisfyingly elegant and produces high accuracy results. Our classifier is able to predict classes for **both** the Spam Filtering system and the Digit Recognition system.

---

## Overview of the Classifier

The underlying principle that the classifier is based on is Bayes Theorem. 

- Bayes Theorem says that if event X has occurred, then we can find the probability of event Y given event X
- **P(Y | X)**

The problem with Bayes Theorem is that for an n-dimensional vector called X, there are 2^n possible combinations of it. And, for a binary variable Y, there are 2^(n+1) combinations for P(X1, X2, ... , Xn | Y).

In our spam classifier, we have a 54-dimensional vector called message and two classes. This would result in 2^(54+1) combinations for P(message | Class = class) and is completely unrealisitic for real-world applications.

However, we can use Naïve Bayes to solve this problem. The main idea behind Naïve Bayes is to assume that each input vector is conditionally independent of other features. In our spam filtering example, we assume conditional independence of keywords (from the feature set) given the class.

- This means that we assume the words in a message were drawn independently from a bag of k different words (where k = 54 in our case)

As a result of applying Naïve Bayes, we can reduce the number of values we have to find from 2^(n+2) to (2n+2) which drastically improves the efficiency of the process, making the algorithm feasible.

- Where n is the dimensionality of the feature vector (also known as the query vector)

Using Naïve Bayes, we can get the probability of a class given a feature vector. We can then compare the probabilities of each class and choose the class that has the greatest probability to be the one classified for the given feature vector. The rule for choosing the class is known as the `maximum a posterior` decision rule.

The classifiers used in this assignment are the same for btoh the Spam Filtering and Digit Classification sections as we made the class 'MyClassifier' generalised such that it can perform the machine learning for any number of classes. This is enabled as a result of the assumption that the data is modelled by a multinomial distribution. _To change the number of classes which are to be identified, pass the an argument 'k' with an integer value denoting the number of classes when you instantiate the **MyClassifier** class._

---

## Function - _estimate_log_class_priors()_

This function takes an input of an n-length numpy array and outputs the log of the priors (probabilities) of each class.

In layman terms, this means that it counts the number of times each class occurs in the dataset provided and divides that value by the total length of the dataset. This gives the probability of a class, so we simply perform a log function over the array to get the log of the probabilities (_log_class_priors_).

## Function - _estimate_log_class_conditional_likelihoods()_

This function takes three inputs:

1. _input_data_ --- which is a dataset of n samples and m features (i.e. a feature could be: does a word occur in a message, or is a pixel white or black in this location of the image).
2. _labels_ --- which is a dataset of n samples corresponding to the class labels of each row in _input_data_.
3. _alpha_ --- which is a hyper-parameter for tuning the effect of the **laplace smoothing**.

For each of the classes, the function calculates **log(P( w_i | c ))**. This means that the function calculates the log of the probability of a feature given a class label.

To calculate this, we first isolate _input_data_ into sub-datasets, of which each sub-dataset corresponds to the rows of _input_data_ which are given a specific label (e.g. separate _input_data_ into the rows with the label 'ham' and into the rows with the label 'spam').

We next calculate the class conditional likelihoods of each feature by counting the number of times a feature appears for a given class, adding alpha (the hyper-parameter) and then dividing the result by the total number of rows with said class label plus alpha times the number of features in the dataset.

- The use of alpha enables **laplace smoothing**.
- **Laplace smoothing** is necessary because of the use of the multinomial distribution.
- For example, if one of our features does not ever occur in our dataset, without the use of laplace smoothing it would have a probability of zero.
- This is very unlikely to be accurate and is more likely to be a result of not using a large enough dataset for the training of the model.
- Using **laplace smoothing** allows us to change the probability from zero to a very small probability (e.g. 0.001) and stops the program from essentially eliminating one of the features, which could result in a class being given a probability of zero in the prediction method.

Finally, we find the log of each of the class conditional likelihoods and return the array containing the values, called theta.

## Function - _train()_

This function simply calls the two above-mentioned functions with the training data and sets the class fields (of MyClassifier) '_log_class_priors_' and '_theta_' equal to the respective outputs of the functions _estimate_log_class_priors_ and _estimate_log_conditional_likelihoods_.

This sets the class up with the necessary data to perform predictions in the next function.

## Function - _predict()_

The predict function is a given an input of test data with a shape of (n_samples, m_features). It then loops through each row in the test data and finds the features which are present in the row and selects the respective probabilties of those features.

All the probabilities for the row are then summed together (called the _row_class_conditional_likelihoods_sum_), before being added to _log_class_priors_ to get the probability that row _i_ corresponds to a class.

Finally, the class which has the highest probability (argmax from the above calculations) is added to a _class_predictions_ array and then the next row is inspected.

The beauty of using logs of probabilities in each of the steps of the machine learning algorithm is that we only need to add the probabilities when it comes to calculating the predictions (as opposed to getting the products of them) due to the laws of logarithms.

---

## Discussion of the results

With our implemenation of a Naïve Bayes classifier and setting alpha equal to 100 (the hyper-parameter used in **laplace smoothing**), we were able to achieve an accuracy of 0.89, or 89% for the spam filtering system and 0.493, or 49.3% for the digit recognition system using the test data provided.

We believe that these are high results and the only way for them to be improved upon (in our opinion) is by implementing a better feature extractor in the digit recognition system. 

# Conclusion

In conclusion, this project was yet another interesting journey into machine learning and furthering our education. Although we managed to implement a classifier for both the Spam Filtering system and the Digit Recognition system, I would have liked to have implemented a better feature extractor if time had permitted.

Nonetheless, the skills gained during the process of researching and coding this project will be extremely useful going forward and we are excited to see what's to come next.