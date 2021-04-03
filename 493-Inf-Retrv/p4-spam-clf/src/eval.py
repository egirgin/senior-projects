import os
import math
import random

def calculate_metrics(groundTruth, preds):

    """
        For the given GT and predictions calculates
        the metrics wanted.

    :@param groundTruth: A list of integers representing the labels
    :@param preds: A list of integers representing the predictions
    :@return result: A dict storing the metrics calculated
    """
    
    tp = 0
    fn = 0
    fp = 0
    tn = 0

    # Build confusion matrix
    for idx in range(len(preds)):
        label = groundTruth[idx]
        pred = preds[idx]

        if label == 1 and pred == 1:
            tp += 1
        elif label == 1 and pred == 0:
            fn += 1
        elif label == 0 and pred == 1:
            fp += 1
        elif label == 0 and pred == 0:
            tn += 1

    
    precision_spam = tp/(tp + fp) if (tp + fp) != 0 else 0
    precision_legit = tn/(tn + fn) if (tn + fn) != 0 else 0

    recall_spam = tp/(tp + fn) if (tp + fn) != 0 else 0
    recall_legit = tn/(tn + fp) if (tn + fp) != 0 else 0


    f1_spam = (2*precision_spam*recall_spam) / (precision_spam + recall_spam) if (precision_spam + recall_spam) != 0 else 0

    f1_legit = (2*precision_legit*recall_legit) / (precision_legit + recall_legit) if (precision_spam + recall_spam) != 0 else 0


    result = {
        "PrecisionSpam" : precision_spam,
        "PrecisionLegitimate" : precision_legit,
        "RecallSpam" : recall_spam,
        "RecallLegitimate" : recall_legit,
        "F1Spam" : f1_spam,
        "F1Legitimate" : f1_legit,
        "MacroAvgPrecision" : (precision_spam + precision_legit ) / 2,
        "MacroAvgRecall" : (recall_spam + recall_legit ) / 2,
        "MacroAvgF1" : (f1_spam + f1_legit ) / 2,
        "Accuracy" : (tp + tn) / (tp + fp + tn + fn)
    }

    return result


def randomization_test(groundTruth, preds1, preds2, R=1000):
    """
        Applies randomization test to given outputs of two models

    :@param groundTruth: A list of integers representing the labels
    :@param preds1: A list of integers representing the predictions from model1
    :@param preds2: A list of integers representing the predictions from model2
    :@param R: The number of times the test will applied.
    :@return p_value: A value representing the strength of the hypothesis (two models are the same)
    """


    result1 = calculate_metrics(groundTruth, preds1)
    result2 = calculate_metrics(groundTruth, preds2)

    f1_1 = result1["MacroAvgF1"]
    f1_2 = result2["MacroAvgF1"]

    s_default = abs(f1_1 - f1_2)

    counter = 0

    for _ in range(R):
        
        # Change the elements of the preds randomly
        for idx in range(len(groundTruth)):
            if random.random() > 0.5:
                elem1 = preds1[idx]
                elem2 = preds2[idx]

                preds1[idx] = elem2
                preds2[idx] = elem1

        # Calculate macro averaged F1 scores for both shuffled preds
        result1 = calculate_metrics(groundTruth, preds1)
        result2 = calculate_metrics(groundTruth, preds2)

        f1_1 = result1["MacroAvgF1"]
        f1_2 = result2["MacroAvgF1"]

        s_prime = abs(f1_1 - f1_2)


        if s_prime >= s_default:
            counter += 1

    p_value = (counter+1) / (R+1)

    return p_value

        
        



