
def precision(groundTruths, predictions):
    groundTruths = set(groundTruths)
    predictions = set(predictions)

    return len(groundTruths.intersection(predictions))/len(predictions)

def average_precision(groundTruths, predictions):
    
    groundTruths = set(groundTruths)
    predictions = set(predictions)

    m = len(groundTruths.intersection(predictions))

    groundTruths = list(groundTruths)
    predictions = list(predictions)

    summ = 0

    for i in range(0, len(predictions)):
        if predictions[i] in groundTruths:
            summ += precision(groundTruths[:i+1], predictions[:i+1])

    return summ/m if m != 0 else 0