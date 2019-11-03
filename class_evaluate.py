import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc,roc_auc_score,roc_curve,confusion_matrix
from pprint import pprint

class ClassEval:
    def __init__(self,*args,cfs = True,ks_auc = True ,roc_plt = True, rep = True, bins = 20):
        '''
           [y_true,y_pre] : 最多只能两组传入,训练先于测试数据进入,预测值以单列为1的概率值传入
           cfs,ks_auc,roc_plt,rep : 混淆矩阵，ks auc值，roc曲线，报告， 默认都为true显示，可修改为false不展示
        '''
        self.cfs = cfs
        self.ks_auc = ks_auc
        self.roc_plt = roc_plt
        self.rep = rep
        self.bins = bins
        self.data = args
    
    def popp(self):
        print(self.cfs)
        print(self.ks_auc)
        print(self.data)
    
    def evaluate(self):
        '''
            y_true : y_true在前
            y_pre_pro : 传入的必须是预测概率值
        '''
        if len(self.data) == 1:
            y_tr,y_tr_pro = self.data[0][0],self.data[0][1]
            fpr,tpr,ks_bd = self.ks_auc_score(y_tr,y_tr_pro)
            self.confusion_matrix_own(y_tr,y_tr_pro,ks_bd)
            self.roc_curve_plot([fpr,tpr])
            self.report_table(y_tr,y_tr_pro)
        else:
            y_tr,y_tr_pro = self.data[0][0],self.data[0][1]
            y_te,y_te_pro = self.data[1][0],self.data[1][1]
            
            print('训练集')
            fpr_tr,tpr_tr,ks_bd = self.ks_auc_score(y_tr,y_tr_pro)
            self.confusion_matrix_own(y_tr,y_tr_pro,ks_bd)
            
            print('\n\n测试集')
            fpr_te,tpr_te,ks_bd = self.ks_auc_score(y_te,y_te_pro)
            self.confusion_matrix_own(y_te,y_te_pro,ks_bd)
            self.roc_curve_plot([fpr_tr,tpr_tr],[fpr_te,tpr_te])
            self.report_table(y_te,y_te_pro)

    def confusion_matrix_own(self,y_true,y_pro,ks_bd):
        if self.cfs:
            print('阈值为 : {}'.format(ks_bd))
            y_pre = np.array([1 if x >= ks_bd else 0 for x in y_pro])
            tp = ((y_true == 1) & (y_pre == 1)).sum()
            fp = ((y_true == 0) & (y_pre == 1)).sum()
            fn = ((y_true == 1) & (y_pre == 0)).sum()
            tn = ((y_true == 0) & (y_pre == 0)).sum()
            print('混淆矩阵 : \n{}'.format(pd.DataFrame([[tn,fp],[fn,tp]] ,columns = [['predict','predict'],[0,1]],index = [['true','true'],[0,1]])))

    def ks_auc_score(self,y_true,y_pro):
        fpr,tpr,_ = roc_curve(y_true,y_pro)
        ks = (tpr - fpr).max()
        auc_val = auc(fpr,tpr)
        
        if self.ks_auc:
            print('\nks : {}'.format(ks))
            print('auc : {}'.format(auc_val))
        ks_lst = list(tpr - fpr)
        ks_idx = ks_lst.index(max(ks_lst))
        ks_bd = sorted(list(y_pro),reverse=True)[ks_idx]
        return fpr,tpr,ks_bd
        

    def roc_curve_plot(self,*args):
        if self.roc_plt:
            print('\n\nRoc curve:')
            if len(args) == 2:
                fpr_tr,tpr_tr = args[0]
                fpr_te,tpr_te = args[1]
                plt.plot(fpr_tr,tpr_tr,label = 'train_roc')
                plt.plot(fpr_te,tpr_te,label = 'test_roc')
            else:
                fpr_tr,tpr_tr = args[0]
                plt.plot(fpr_tr,tpr_tr,label = 'roc')
            plt.plot([0,1],[0,1],'k--')
            plt.xlabel('fpr')
            plt.ylabel('tpr')
            plt.title('ROC Curve')
            plt.legend(loc = 'best')
            plt.show()
            
    def report_table(self,y_true,y_pre):
        '''
            y_true传入真实01值，y_pre传入为1的概率值
        '''
        if self.rep:
            y_true,y_pre = list(y_true),list(y_pre)
            lis = [(y_true[i],y_pre[i]) for i in range(len(y_true))]

            lis.sort(key=lambda x : x[1],reverse=True)

            bad_num = sum(y_true)
            good_num = len(y_true) - bad_num
            sample_num = len(lis)
            bin_sample = (sample_num // self.bins) + 1

            bad_add = 0
            good_add = 0
            bad_pct = 0
            good_pct = 0
            df_rep = pd.DataFrame(columns=['KS','BAD','GOOD','BAD_CNT','GOOD_CNT','BAD_PCTG','GOOD_PCTG','BADRATE','GOODRATE'])

            for i in range(self.bins):
                ds = lis[i*bin_sample : min(bin_sample*(i+1),sample_num)]
                bad = int(sum([x[0] for x in ds]))
                good = int(len(ds) - bad)
                bad_add = int(bad_add + bad)
                good_add = int(good_add + good)
                bad_pct += (bad / bad_num)
                good_pct += (good / good_num)
                bad_rate = bad / len(ds)
                good_rate = good / len(ds)
                ks = round((bad_add / bad_num) - (good_add / good_num),3)

                df_rep = df_rep.append([{'KS':ks,'BAD':bad,'GOOD':good,'BAD_CNT':bad_add,'GOOD_CNT':good_add,'BAD_PCTG':round(bad_pct,3),'GOOD_PCTG':round(good_pct,3),'BADRATE':round(bad_rate,3),'GOODRATE':round(good_rate,3)}],ignore_index=True)

            pprint(df_rep)