import numpy as np
from collections import Counter


class Feature(object):
    def __init__(self, labels):
        self.val = None
        self.labels = labels

    def fit(self, val):
        self.val = np.array(val)

    def __add__(self, other):
        labels = self.labels + other.labels
        val = np.concatenate([self.val, other.val], axis=-1)
        out = Feature(labels)
        out.val = val
        return out


class CountTokenizer(Feature):
    def __init__(self, vocab, sep=","):
        self.labels = vocab
        self.sep = sep

    def fit(self, val):
        val = [s.split(self.sep) for s in val]
        counts = [Counter(s) for s in val]
        vals = [[count[v] for v in self.labels] for count in counts]
        self.val = np.array(vals)


def demo():
    plant_or_animal = Feature(labels=["animal", "reptile", "plant", "mammal"])
    plant_or_animal.fit([[0, 0, 1, 0], [1, 1, 1, 0]])
    print(plant_or_animal.val)
    # [[0 0 1 0]
    #  [1 1 1 0]]
    print(plant_or_animal.labels)
    #  ['animal', 'reptile', 'plant', 'mammal']
    dog_cats = CountTokenizer(vocab=["dog", "cat"])
    dog_cats.fit(["dog,dog,dog,cat,cat,dog", "dog"])
    print(dog_cats.val)
    # [[4 2]
    #  [1 0]]
    print(dog_cats.labels)
    # ['dog', 'cat']
    combined = plant_or_animal + dog_cats
    print(combined.labels)
    # ['animal', 'reptile', 'plant', 'mammal', 'dog', 'cat']
    print(combined.val)
    # [[0 0 1 0 4 2]
    #  [1 1 1 0 1 0]]


if __name__ == "__main__":
    demo()
