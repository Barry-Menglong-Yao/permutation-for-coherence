# from https://github.com/DaoD/ConstraintGraph4NSO/blob/main/SecondPhase/Evaluate.py


import zope.interface
import numpy as np
import torch 

class MetricHolder():
    def __init__(self,primary_metric ):
        metric_dic={}
        metric_dic["acc"]=Acc()
        metric_dic["pmr"]=PMR()
        metric_dic["first_acc"]=FirstAcc()
        metric_dic["last_acc"]=LastAcc()
        metric_dic["tau"]=Tau()
        metric_dic["lcs"]=Lcs()
        self.metric_dic=metric_dic
        self.primary_metric=primary_metric
        


    def compute(self,y_pred, y_true, story_len ):
        cat_pred = ranks(y_pred.to("cpu"))
        cat_pred = cat_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        
        for key,metric in self.metric_dic.items() :
            metric.compute_one_step(cat_pred,y_true , story_len ) 
    def avg_step_score_str(self,steps):
        score_str=""
        for key,metric in self.metric_dic.items():
            score=metric.get_avg_step_score(steps)
            score_str+= type(metric).__name__ + ":"+str(score ) +"; "
        return score_str
    def epoch_reset(self):
        for key,metric in self.metric_dic.items():
            metric.epoch_reset()    

    def update_epoch_score(self,epoch,steps):
        score_str=""
        for key,metric in self.metric_dic.items():
            score,best_epoch=metric.update_epoch_score(epoch,steps)
            if best_epoch==epoch:
                score_str+= "best "+type(metric).__name__ + ":"+str(score  ) +"; "
            else:
                score_str+= type(metric).__name__ + ":"+str(score  ) +"; "
        best_score,best_epoch=self.metric_dic[self.primary_metric].get_best_epoch_score()
        return best_score,best_epoch,score_str



def ranks(inputs, axis=-1):
	  
    return torch.argsort(torch.argsort(inputs, descending=False, dim=1), dim=1) 


class Metric( ):
    def __init__(self ):
        self.best_epoch_score  = -np.inf
        self.best_epoch=0
        self.over_score=0
        self.cur_step_score = -np.inf
    def epoch_reset(self):
        self.over_score=0
        self.cur_step_score = -np.inf
    def compute_one_step(self, y_pred, y_label, story_len ):
        pass
    def get_best_epoch_score(self):
        return self.best_epoch_score,self.best_epoch
    def get_avg_step_score(self,step):
        return self.over_score/step
    def update_epoch_score(self,epoch,step):
        score=self.over_score/step
        if score>self.best_epoch_score:
            self.best_epoch_score=score 
            self.best_epoch=epoch
        return score,self.best_epoch
 
class PMR(Metric):
    def compute_one_step(self, y_pred, y_label, story_len ):
        num = len(y_pred)
        all_acc = 0.0
        count = 0
        for i in range(num):
            pred = y_pred[i][:story_len[i]]
            label = y_label[i][:story_len[i]]
            acc = 1 if sum(pred == label) == story_len[i] else 0
            all_acc += acc
            count += 1
        self.cur_step_score=all_acc / count
        self.over_score+=self.cur_step_score
        return self.cur_step_score



 
 
class Acc(Metric):
    
    def compute_one_step(self, y_pred, y_label, story_len ):
        num = len(y_pred)
        all_acc = 0.0
        count = 0
        for i in range(num):
            pred = y_pred[i][:story_len[i]]
            label = y_label[i][:story_len[i]]
            acc = sum(pred == label) / story_len[i]
            all_acc += acc
            count += 1
        self.cur_step_score=all_acc / count
        self.over_score+=self.cur_step_score
        return self.cur_step_score
 
class Tau(Metric):
    
    def compute_one_step(self, y_pred, y_label, story_len ):
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
 
        for i in range(num):
            pred = y_pred[i][:story_len[i]]
            label = y_label[i][:story_len[i]]
            if len(pred) == 1 and len(label) == 1:
                TAU = 1
            else:
                TAU = kendall_tau(pred, label)
            all_tau += TAU
            count += 1
        self.cur_step_score=all_tau / count
        self.over_score+=self.cur_step_score
        return self.cur_step_score
 
 
class Lcs(Metric):
     
    def compute_one_step(self, y_pred, y_label, story_len ):
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
        self.cur_step_score=all_lcs / count
        self.over_score+=self.cur_step_score
        return self.cur_step_score
 
class FirstAcc(Metric):
     
    def compute_one_step(self, y_pred, y_label, story_len ):
        count = 0
        first = 0
        last = 0
        num = len(y_pred)
        for i in range(num):
            pred = y_pred[i][:story_len[i]]
            label = y_label[i][:story_len[i]]
            if pred[0] == label[0]:
                first += 1
 
            count += 1
 
        self.cur_step_score=first / count
        self.over_score+=self.cur_step_score
        return self.cur_step_score
 
 
class LastAcc(Metric):
     
    def compute_one_step(self, y_pred, y_label, story_len ):
        count = 0
        first = 0
        last = 0
        num = len(y_pred)
        for i in range(num):
            pred = y_pred[i][:story_len[i]]
            label = y_label[i][:story_len[i]]
 
            if pred[-1] == label[-1]:
                last += 1
            count += 1
 
        self.cur_step_score=last / count
        self.over_score+=self.cur_step_score
        return self.cur_step_score
  