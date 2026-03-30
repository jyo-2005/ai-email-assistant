def grade(predictions, labels):
    total = len(labels)
    correct = 0

    for p, l in zip(predictions, labels):
        if p == l:
            correct += 1

    score = correct / total
    return score
