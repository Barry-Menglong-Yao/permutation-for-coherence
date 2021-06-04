# from https://github.com/DaoD/ConstraintGraph4NSO/blob/main/SecondPhase/Evaluate.py

def evaluate_lcs(y_pred, y_label, story_len):
    def lcs(X , Y): 
        m = len(X) 
        n = len(Y) 

        L = [[None]*(n+1) for i in range(m+1)] 

        for i in range(m+1): 
            for j in range(n+1): 
                if i == 0 or j == 0 : 
                    L[i][j] = 0
                elif X[i-1] == Y[j-1]: 
                    L[i][j] = L[i-1][j-1]+1
                else: 
                    L[i][j] = max(L[i-1][j] , L[i][j-1]) 
        return L[m][n] 
    num = len(y_pred)
    all_lcs = 0.0
    count = 0
    for i in range(num):
        pred = y_pred[i][:story_len[i]]
        label = y_label[i][:story_len[i]]
        LCS = lcs(pred, label)
        all_lcs += LCS / story_len[i]
        count += 1
    return all_lcs / count

def evaluate_tau(y_pred, y_label, story_len, return_list=False):
    def kendall_tau(porder, gorder):
        pred_pairs, gold_pairs = [], []
        for i in range(len(porder)):
            for j in range(i+1, len(porder)):
                pred_pairs.append((porder[i], porder[j]))
                gold_pairs.append((gorder[i], gorder[j]))
        common = len(set(pred_pairs).intersection(set(gold_pairs)))
        uncommon = len(gold_pairs) - common
        tau = 1 - (2*(uncommon/len(gold_pairs)))
        return tau
    num = len(y_pred)
    all_tau = 0.0
    count = 0
    tau_list = []
    for i in range(num):
        pred = y_pred[i][:story_len[i]]
        label = y_label[i][:story_len[i]]
        if len(pred) == 1 and len(label) == 1:
            TAU = 1
        else:
            TAU = kendall_tau(pred, label)
        all_tau += TAU
        tau_list.append(TAU)
        count += 1
    if return_list:
        return all_tau / count, tau_list
    else:
        return all_tau / count

def evaluate_first_last_accuracy(y_pred, y_label, story_len):
    count = 0
    first = 0
    last = 0
    num = len(y_pred)
    for i in range(num):
        pred = y_pred[i][:story_len[i]]
        label = y_label[i][:story_len[i]]
        if pred[0] == label[0]:
            first += 1
        if pred[-1] == label[-1]:
            last += 1
        count += 1
    return first / count, last / count