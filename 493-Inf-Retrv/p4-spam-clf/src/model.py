import os
import json
import pickle
import random

from naiveBayes import NaiveBayes
from feature_extractor import MutualInformation
from eval import calculate_metrics, randomization_test


def load_dataset():
    with open("./dataset.json", "r") as datasetFile:
        dataset = json.load(datasetFile)

    return dataset

def create_load_model(dataset, model_path, mi=None):
    # Load the existing model or create a new one
    if os.path.exists(model_path):
        with open(model_path, "rb") as readModel:
            model = pickle.load(readModel)
    else:
        model = NaiveBayes(alpha = 1)

        if mi:
            model.mi_train(dataset,mi)
        else:
            model.train(dataset)

        with open(model_path, "wb+") as modelFile:
            pickle.dump(model, modelFile)

    return model


def main():

    # Load Dataset
    dataset = load_dataset()

    # Extract MI
    mi_selector = MutualInformation(dataset["train"], k=100)

    mi_features = mi_selector.generate()

    # Train model
    model = create_load_model(dataset["train"], "./model.pth")
    model_mi = create_load_model(dataset["train"], "./model_mi.pth", mi=mi_features)
    
    # Create Test Data
    test_set = dataset["test"]["spam"] + dataset["test"]["legitimate"]

    spam_length = len(dataset["test"]["spam"])
    legit_length = len(dataset["test"]["legitimate"])
    
    groundTruths = [1] * spam_length + [0] * legit_length

    # Test Model
    preds_mi = model_mi.mi_test(test_set)
    preds = model.test(test_set)

    # Calculate Metrics
    result = calculate_metrics(groundTruth=groundTruths, preds=preds)
    result_mi = calculate_metrics(groundTruth=groundTruths, preds=preds_mi)

    print("\n")
    print(" --> Model 1 : W/O Feature Selection")
    print(" - Macro Averaged Precision: {:.2f}".format(result["MacroAvgPrecision"]))
    print(" - Macro Averaged Recall: {:.2f}".format(result["MacroAvgRecall"]))
    print(" - Macro Averaged F1: {:.2f}".format(result["MacroAvgF1"]))
    print("\n")
    print(" - Spam Precision: {:.2f}".format(result["PrecisionSpam"]))
    print(" - Spam Recall: {:.2f}".format(result["RecallSpam"]))
    print(" - Spam F1: {:.2f}".format(result["F1Spam"]))
    print("\n")
    print(" - Legitimate Precision: {:.2f}".format(result["PrecisionLegitimate"]))
    print(" - Legitimate Recall: {:.2f}".format(result["RecallLegitimate"]))
    print(" - Legitimate F1: {:.2f}".format(result["F1Legitimate"]))

    print("\n")

    print(" --> Model 2 : W/ Feature Selection")
    print(" - Macro Averaged Precision: {:.2f}".format(result_mi["MacroAvgPrecision"]))
    print(" - Macro Averaged Recall: {:.2f}".format(result_mi["MacroAvgRecall"]))
    print(" - Macro Averaged F1: {:.2f}".format(result_mi["MacroAvgF1"]))
    print("\n")
    print(" - Spam Precision: {:.2f}".format(result_mi["PrecisionSpam"]))
    print(" - Spam Recall: {:.2f}".format(result_mi["RecallSpam"]))
    print(" - Spam F1: {:.2f}".format(result_mi["F1Spam"]))
    print("\n")
    print(" - Legitimate Precision: {:.2f}".format(result_mi["PrecisionLegitimate"]))
    print(" - Legitimate Recall: {:.2f}".format(result_mi["RecallLegitimate"]))
    print(" - Legitimate F1: {:.2f}".format(result_mi["F1Legitimate"]))


    p_value = randomization_test(groundTruths, preds, preds_mi)

    print("\n")
    if p_value <= 0.05:
        print("The models are different (P-Value : {})".format(p_value))

    else:
        print("The models are the same (P-Value : {})".format(p_value))



if __name__ == "__main__":
    main()


