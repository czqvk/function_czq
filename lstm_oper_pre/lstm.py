import pandas as pd
import numpy as np
import tensorflow
import logging
from sklearn.metrics import roc_auc_score
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LSTM,Dense,Dropout,Activation,SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model

import sys
sys.path.append(r'D:\python\tools\class_label_evaluate')
from class_evaluate import ClassEval

#这份代码的目的是，基于用户操作的数据，将每个用户申请贷款的前200次操作编码成（200,140）的矩阵向量，然后进行违约欺诈客户进行有监督建模
#后面将其中有关lstm模型构建部分和分类结果分析部分提取出来，保存成此份代码

class LossHistory(keras.callbacks.Callback):
    ## 设置回调函数记录每次epoch训练后的训练集测试集损失
    def on_train_begin(self, logs={}):
        self.losses = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}

    def on_epoch_end(self, batch, logs=[]):
        self.lossesp['loss'].append(logs.get('loss'))
        self.lossesp['acc'].append(logs.get('acc'))
        self.lossesp['val_loss'].append(logs.get('val_loss'))
        self.lossesp['val_acc'].append(logs.get('val_acc'))

class roc_callback(keras.callbacks.Callback):
    ## 设置回调函数记录每次epoch训练后的训练集测试集auc
    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self,logs={}):
        self.losses = {'auc':[],'val_auc':[]}

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)

        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        self.losses['auc'].append(roc)
        self.losses['val_auc'].append(roc_val)

        print('\rauc: %s --auc_val: %s'% ((str(round(roc,4))),str(round(roc_val,4))),end=100*' '+'\n')


class model_run:
    def __init__(self,train_vec,train_label,test_vec,test_label,log_file,params,model_save=True,model_load=False,model_name=None,logging = True):
        self.params = params
        self.log_file = log_file
        self.logger = self.logger()
        self.train_vec = train_vec
        self.train_label = train_label
        self.test_vec = test_vec
        self.test_label = test_label
        self.model_save = model_save
        self.model_load = model_load
        self.model_name = model_name

    def model_design(self):
        '''
        :return: 设计模型的结构
        '''
        model = Sequential()
        model.add(Dropout(self.params.get('drop_out'), input_shape=(90,10)))
        ## 加入正则化项
        model.add(LSTM(self.params.get('LSTM_layers'),input_shape = (90,10)))
        model.add(Dense(1,activation = 'sigmoid', kernel_regularizer = regularizers.l2(0.0003)))
        return model

    def run(self):
        ## 读取或编译模型
        train_epochs = 0
        if self.model_load:
            self.load_model(det_name = self.model_name)
            train_epochs = self.model_name.split('_')[-1].split('.')[0]
        else:
            self.model  = self.model_design()
            self.model.compile(loss = self.params.get('loss'),metrics = ['accuracy'],optimizer=self.params.get('optimizer'))

        # history =LossHistory()
        # 设置回调函数并跑模型
        auc_cal = roc_callback(training_data=[self.train_vec, self.train_label], validation_data= [self.test_vec, self.test_label])

        self.model.fit(self.train_vec,
                       self.train_label,
                       batch_size = self.params.get('batch_size'),
                       epochs = self.params.get('epochs'),
                       validation_data = (self.test_vec, self.test_label),
                       callbacks = [auc_cal])

        ## 保存模型和训练过程的评估结果
        if self.model_save:
            new_model_name = 'model_111'
            self.save_model(new_model_name)
            print('save model {} success'.format(new_model_name))

        ks,auc_val,ks_bd,df_confusion = self.model_eva()
        self.logger.info('params: {}\n epoch_eva(train_auc,val_auc) : {}\n ks : {}\n 混淆矩阵 : {}'.format(self.params,auc_cal.losses,ks,auc_val,df_confusion))

    def logger(self):
        '''
        :return: 设置日志
        '''
        logger = logging.getLogger('example')
        ch = logging.basicConfig(level=logging.INFO,
                                 format = '%(asctime)s -- %(filename)s -- %(levelname)s -- %(message)s',
                                 filename = self.log_file,
                                 filemode = 'a+')
        return logger

    def model_eva(self):
        '''
        :return: 评估分类效果
        '''
        test_pre_pro = self.model.predict_proba(self.test_vec)
        test_pre_pro_ls = [t[0] for t in test_pre_pro]
        model_eva = ClassEval([np.array(self.test_label),np.array(test_pre_pro_ls)])
        ks,auc_val,ks_bd,df_confusion = model_eva.evaluate()
        return ks,auc_val,ks_bd,df_confusion

    ## 模型保存和读取
    def save_model(self, det_name):
        self.model.save(r'model/{}'.format(det_name))

    def load_model(self, det_name):
        self.model = load_model(r'model/{}'.format(det_name))


if __name__ == '__main__':
    x_mat = np.random.randint(8, size=(100, 90, 10))
    y_mat = np.random.randint(2, size=100)

    x_test = np.random.randint(8, size=(96, 90, 10))
    y_test = np.random.randint(2, size=96)

    param = {
        'drop_out': 0.4,
        'LSTM_layers': 50,
        'loss': 'binary_crossentropy',
        'optimizer': 'adam',
        'batch_size': 64,
        'epochs': 2,
    }
    log_file = 'logger/log_test'

    model = model_run(x_mat, y_mat, x_test, y_test, log_file, param, model_save=False)
    model.run()