import csv
import numpy as np

from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score

TAG = '[EVALUATION]'


class Evaluation:

    def __init__(self, data, targets, labels):
        self.data = data
        self.targets = targets
        self.labels = labels
    
    def Accuracy(self):
        
        print(TAG, 'Calculating Accuracy.')

        labels = np.array(self.labels)
        classes = np.array(self.targets)

        n = len(labels)
        vec = np.array(range(n))
        ones = np.ones(n)
        ones = np.array(list(map(int, ones)))

        cat = coo_matrix((ones.T, (labels, vec))).todense()
        cls = coo_matrix((ones.T, (classes, vec))).todense()

        cat = np.delete(cat, 0, 0)
        cls = np.delete(cls, 0, 0)

        cmat = np.matmul(cat, cls.T)
        cmat = cmat.T

        row_ind, col_ind = linear_sum_assignment(-cmat)
        cost = cmat[row_ind, col_ind].sum() 

        return cost / n

    def NMI(self):
        
        print(TAG, 'Calculating NMI.')

        A = self.targets
        B = self.labels
        eps = 2.2204e-16
        
        if (len(A) != len (B)):
            print(TAG, 'Error in nmi()')
            return 0.0
    
        total = len(A)
        A_ids = np.unique(A)
        B_ids = np.unique(B)
    
        MI = 0.0
    
        for idA in A_ids:
            for idB in B_ids:
    
                idAOccur = np.where(A == idA)[0]
                idBOccur = np.where(B == idB)[0]
                idABOccur = np.intersect1d(idAOccur, idBOccur)
    
                px = len(idAOccur) / total
                py = len(idBOccur) / total
                pxy = len(idABOccur) / total
    
                MI = MI + pxy * np.log2(pxy / (px * py) + eps)
    
        Hx = 0
        for idA in A_ids:
            idAOccurCount = len( np.where(A == idA)[0] )
            Hx = Hx - (idAOccurCount / total) * np.log2(idAOccurCount / total + eps)
    
        Hy = 0
        for idB in B_ids:
            idBOccurCount = len( np.where(B == idB)[0] )
            Hy = Hy - (idBOccurCount / total) * np.log2(idBOccurCount / total + eps)
    
        return 2 * MI / (Hx + Hy)
    
    def Purity(self): 
        print(TAG, 'Calculating Purity.')
        contingency_matrix = metrics.cluster.contingency_matrix(self.targets, self.labels)
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
    
    def ARI(self):
        print(TAG, 'Calculating ARI.')
        return adjusted_rand_score(self.targets, self.labels)

    def save_labels(self, data_name, idx):

        path = 'ResultLabels/labels' + data_name + str(idx)

        print(TAG, 'Writing result labels to: ' + path)

        with open( path + '.csv', mode='w') as csv_file:
            writer = csv.writer(csv_file)    
            writer.writerow(self.labels)

            print(TAG, 'Done writing labels.')