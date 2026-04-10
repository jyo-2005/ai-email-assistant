def grade(predictions, labels):
    correct = sum([p == l for p, l in zip(predictions, labels)])
    return correct / len(labels)


# Task-specific graders
def grade_easy(predictions, labels):
    return grade(predictions, labels)


def grade_medium(predictions, labels):
    return grade(predictions, labels)


def grade_hard(predictions, labels):
    return grade(predictions, labels)
