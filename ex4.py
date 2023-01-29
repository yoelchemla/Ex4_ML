import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split


def knn(train, query_point, order, ans):
    """
    This function calculates the distance between a single point and all the points in the training set. The chosen
    distance metric (L2-norm) and the dimensionality of the point are determined by inputs passed to the function.
    The result is a list of tuples with the distances and labels of each point in the training set.
    """
    # credit: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm#Algorithm

    list_of_distance = []
    if not ans:
        first_point = np.array([query_point.x, query_point.y])
    else:
        first_point = np.array([query_point.x, query_point.y, query_point.z])

    for point in train.itertuples():
        if not ans:
            second_point = np.array([point.x, point.y])
        else:
            second_point = np.array([point.x, point.y, point.z])
            
        D = np.linalg.norm(first_point - second_point, ord=order)
        list_of_distance.append((D, point.label))
    list_of_distance.sort(key=lambda tup: tup[0])
    return list_of_distance


def round_value(value: float) -> str:
    return format(round(value, 4), '.4f')


def get_prediction(train, test, odd_index, index_p, ans):
    """
    This function is a K-nearest neighbor algorithm implementation that predicts the class label of a sample in the
    test set based on the majority class of its K nearest neighbors in the training set. The algorithm requires the
    input of the training set, test set, number of nearest neighbors (K), the distance metric to be used, and a flag
    variable. The output is an array of predictions for each sample in the test set.
    """
    prediction = np.zeros(test.shape[0])

    index = 0
    positive_index = 1
    negative_index = 0

    # run on test:
    for point in train.itertuples():
        dist = knn(train, point, index_p, ans)
        negative_positive = [0, 0]
        for i in range(odd_index):
            if dist[i][1] == -1:
                negative_positive[negative_index] += 1
            else:
                negative_positive[positive_index] += 1

        if negative_positive[negative_index] <= negative_positive[positive_index]:
            prediction[index] = 1
        else:
            prediction[index] = -1
        index += 1
    return prediction


def run_knn(points, times, max_k, ans):
    train_errors, test_errors, different = np.zeros((5, 3)), np.zeros((5, 3)), np.zeros((5, 3))
    for i in range(times):
        train, test = train_test_split(points, test_size=0.5, stratify=points['label'])
        K = [1, 3, 5, 7, 9][:max_k//2 + 1]
        P = [1, 2, np.inf]
        train_errors_final, test_errors_final = np.zeros((5, 3)), np.zeros((5, 3))
        for j, k in enumerate(K):
            for p in range(len(P)):
                test_prediction = get_prediction(train, test, k, P[p], ans)
                train_prediction = get_prediction(train, train, k, P[p], ans)
                train_errors_final[j][p] += sum(train['label'] != train_prediction) / int(len(points) / 2)
                test_errors_final[j][p] += sum(test['label'] != test_prediction) / int(len(points) / 2)
        train_errors += train_errors_final / times
        test_errors += test_errors_final / times
    different = np.abs(train_errors - test_errors)

    print("true mistake:")
    print("k\t\tp = 1\t\tp = 2\t\tp = inf")
    for i in range(len(K)):
        print(f"{K[i]}\t\t{train_errors[i][0]:.4f}\t\t{train_errors[i][1]:.4f}\t\t{train_errors[i][2]:.4f}")
    print("\nempirical mistake:")
    print("k\t\tp = 1\t\tp = 2\t\tp = inf")
    for i in range(len(K)):
        print(f"{K[i]}\t\t{test_errors[i][0]:.4f}\t\t{test_errors[i][1]:.4f}\t\t{test_errors[i][2]:.4f}")
    print("\nThe difference between empirical and true mistake:")
    print("k\t\tp = 1\t\tp = 2\t\tp = inf")
    for i in range(len(K)):
        print(f"{K[i]}\t\t{different[i][0]:.4f}\t\t{different[i][1]:.4f}\t\t{different[i][2]:.4f}")
    print("Each row present the k value (1,3,5,7,9), each column presents the p value (1,2,inf)")


if __name__ == '__main__':
    haberman = str('haberman.data')
    squares = str('squares.txt')
    ans = False
    if not ans:
        points = pd.read_csv(squares, sep=" ", header=None, names=["x", "y", "label"])
        points.loc[points["label"] == 0, "label"] = -1
    else:
        points = pd.read_csv(haberman, sep=",", header=None, names=["x", "y", "z", "label"])
        points.loc[points["label"] == 2, "label"] = -1

    run_knn(points, 5, 9, ans)
