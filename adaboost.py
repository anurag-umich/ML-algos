#  Adaboost
#
# Version 0.1
#



import math
import numpy as np
from assignment_two_svm \
import evaluate_classifier, print_evaluation_summary



# Remember to return a function, not the
# sign, feature, threshold triple
def weak_learner(instances, labels, dist):

    """ Returns the best 1-d threshold classifier.

    A 1-d threshold classifier is of the form

    lambda x: s*x[j] < threshold

    where s is +1/-1,
          j is a dimension
      and threshold is real number in [-1, 1].

    The best classifier is chosen to minimize the weighted misclassification
    error using weights from the distribution dist.

    """
    n = len(instances[1])
    theta = np.empty(n)
    class_point = np.empty(n)
    errors = np.empty(n)

    for j in xrange(n):

        vars = sorted(set(instances[:, 0]))
        vars.append(float('inf'))
        thresh_list = [float('-inf')] + vars

        num_thresh_list = len(thresh_list)
        class_Error = np.zeros((2, num_thresh_list))

        for k in xrange(num_thresh_list):
           class_Error[0, k] += ((-1 * instances[:, j] <
                           thresh_list[k]) != labels).dot(dist)
           class_Error[1, k] += ((instances[:, j] <
                           thresh_list[k]) != labels).dot(dist)

           thresh_min_index = class_Error.argmin()

           theta[j] = thresh_list[np.unravel_index(class_Error.argmin(),
                                     class_Error.shape)[1]]

           if thresh_min_index < num_thresh_list:
                class_point[j] = -1
           else:
                class_point[j] = 1

           errors[j] = class_Error.flatten()[thresh_min_index]

    min_err_index = errors.argmin()

    return lambda x: (class_point[min_err_index] * x[min_err_index]) < \
        theta[min_err_index]



def compute_error(h, instances, labels, dist):

    """ Returns the weighted misclassification error of h.
    Compute weights from the supplied distribution dist.
    """
    n = len(instances)
    error_vector = np.empty(n)

    for i in xrange(n):
        error_vector[i] = (h(instances[i]) != labels[i]) * 1

    return dist.dot(error_vector)



# Implement the Adaboost distribution update
# Make sure this function returns a probability distribution
def update_dist(h, instances, labels, dist, alpha):

    """ Implements the Adaboost distribution update. """

    n = len(instances)
    dist_update = np.empty(n)

    for i in xrange(n):
        if h(instances[i]) == labels[i]:
            dist_update[i] = dist[i] * np.exp(-alpha)
        else:
            dist_update[i] = dist[i] * np.exp(alpha)

    dist_update=dist_update / sum(dist_update)

    return dist_update


def run_adaboost(instances, labels, weak_learner, num_iters=20):

    n, d = instances.shape
    n1 = labels.size

    if n1 != n:
        raise Exception('Expected same number of labels as no. of rows in \
                        instances')

    alpha_h = []

    dist = np.ones(n)/n

    for i in range(num_iters):

        print "Iteration: %d" % i
        h = weak_learner(instances, labels, dist)

        error = compute_error(h, instances, labels, dist)

        if error > 0.5:
            print "error is " + str(error)
            break

        alpha = 0.5 * math.log((1-error)/error)

        dist = update_dist(h, instances, labels, dist, alpha)

        alpha_h.append((alpha, h))


    # return a classifier whose output
    # is an alpha weighted linear combination of the weak
    # classifiers in the list alpha_h
    def classifier(point):
        """ Classifies point according to a classifier combination.
        The combination is stored in alpha_h.
        """
        alpha = np.array(alpha_h)[:, 0]
        hvector = np.array(alpha_h)[:, 1]

        n = len(alpha)
        hvector_arr = np.empty(n)

        for i in xrange(n):
            hvector_arr[i] = hvector[i](point)

        return (alpha.dot(2 * hvector_arr - 1) > 0) * 1

    return classifier


def main():
    data_file = 'ionosphere.data'

    data = np.genfromtxt(data_file, delimiter=',', dtype='|S10')
    instances = np.array(data[:, :-1], dtype='float')
    labels = np.array(data[:, -1] == 'g', dtype='int')

    n, d = instances.shape
    nlabels = labels.size

    if n != nlabels:
        raise Exception('Expected same no. of feature vector as no. of labels')

    train_data = instances[:200]  # first 200 examples
    train_labels = labels[:200]  # first 200 labels

    test_data = instances[200:]  # example 201 onwards
    test_labels = labels[200:]  # label 201 onwards

    print 'Running Adaboost...'
    adaboost_classifier = run_adaboost(train_data, train_labels, weak_learner)
    print 'Done with Adaboost!\n'

    confusion_mat = evaluate_classifier(adaboost_classifier, test_data,
                                        test_labels)
    print_evaluation_summary(confusion_mat)

if __name__ == '__main__':
    main()