import os
import sys
import pickle
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import tensorflow as tf
import keras
from keras import backend as K
from keras import optimizers
from Up_Lower_Bound import UpperLowerBound
from tensorflow.keras import callbacks

# To generate two random datasets in comparison 1
from Generate_DATA_1 import generate_data1
from Generate_DATA_2 import generate_data2

from CustomEarlyStopping import CustomEarlyStopping

tf.keras.utils.set_random_seed(2)
tf.config.experimental.enable_op_determinism()
# tf.random.set_seed(2)

class Run:

    epochs = 2000
    batch_size = 256

    def __init__(self, No_PCGrad, PCGrad):

        self.No_PCGrad = No_PCGrad
        self.PCGrad = PCGrad

        self.No_PCGrad_model = No_PCGrad.model 
        self.PCGrad_model = PCGrad.model
        
        # To keep the results 
        self.result = []
        self.opt = tf.keras.optimizers.legacy.Adam()

    @classmethod
    def set_epochs(cls, epoches):
        cls.epochs = epoches

    @classmethod
    def set_batch_size(cls, batch_size):
        cls.batch_size = batch_size

    def run_no_pcgrad(self):
        self.No_PCGrad_model.init_arguments()
        self.No_PCGrad_model.compile(optimizer=self.opt,
        loss = [self.No_PCGrad_model.selective_up, self.No_PCGrad_model.selective_low, self.No_PCGrad_model.up_penalty, self.No_PCGrad_model.low_penalty, self.No_PCGrad_model.coverage_penalty],
        metrics = [self.No_PCGrad_model.coverage, self.No_PCGrad_model.mpiw] )
        
        def lr_scheduler(epoch):
            learning_rate = 0.001
            lr_drop = 2000
            return learning_rate * (0.5 ** (epoch // lr_drop))
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        history_no_pcgrad = self.No_PCGrad_model.fit(self.No_PCGrad.X_train, self.No_PCGrad.y_train, 
        validation_data = (self.No_PCGrad.X_val, [self.No_PCGrad.y_val[:,0], self.No_PCGrad.y_val[:,1]]),
        batch_size=self.batch_size, 
        epochs= self.epochs,
        callbacks=[reduce_lr, CustomEarlyStopping(patience= 600)], 
        verbose=1)

        # # Save the training history for analysis 
        name = self.No_PCGrad.dataset.split('.')[0]
        with open(f'{name}_history.pkl', 'wb') as handle:
            pickle.dump(history_no_pcgrad.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # To plot and save .png
        self.plot_training(f'{name}_history.pkl')

        # Predicted results
        no_pcgrad_pred = self.No_PCGrad_model.predict(self.No_PCGrad.X_test)

        # Transformed to original y for comparison 1
        # no_pcgrad_pred = self.No_PCGrad.reversed_norm(no_pcgrad_pred)

        return no_pcgrad_pred

    def run_pcgrad(self):

        self.PCGrad_model.init_arguments(method = 'PCGrad')
        self.PCGrad_model.compile(optimizer=self.opt,
        loss = [self.PCGrad_model.selective_up, self.PCGrad_model.selective_low, self.PCGrad_model.up_penalty, self.PCGrad_model.low_penalty, self.PCGrad_model.coverage_penalty],
        metrics = [self.PCGrad_model.coverage, self.PCGrad_model.mpiw])

        def lr_scheduler(epoch):
            learning_rate = 0.001
            lr_drop = 2000
            return learning_rate * (0.5 ** (epoch // lr_drop))
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        history_pcgrad = self.PCGrad_model.fit(self.PCGrad.X_train, self.PCGrad.y_train, 
        validation_data = (self.PCGrad.X_val, [self.PCGrad.y_val[:,0], self.PCGrad.y_val[:,1]]),
        batch_size=self.batch_size, 
        epochs= self.epochs,
        callbacks=[reduce_lr, CustomEarlyStopping(patience= 600)],
        verbose=1)

        # # Save the training history 
        name = self.PCGrad.dataset.split('.')[0]
        with open(f'{name}_pcgrad_history.pkl', 'wb') as handle:
            pickle.dump(history_pcgrad.history, handle, protocol=pickle.HIGHEST_PROTOCOL)     
        #model.save_weights("checkpoints/{}".format(self.filename))

        self.plot_training(f'{name}_pcgrad_history.pkl')

        pcgrad_pred = self.PCGrad_model.predict(self.PCGrad.X_test)

        # Transfored to original y
        # pcgrad_pred = self.PCGrad.reversed_norm(pcgrad_pred)
        return pcgrad_pred

    def save_pred_results(self, model, predicitons, name):
        df = pd.DataFrame(predicitons, columns = ['Lowerbound', 'Upperbound'])
        # df['y_true'] = model.reversed_norm(model.y_test[:,0].reshape(-1,1))
        df['y_true'] = model.y_test[:,0]
        df['Width'] = (df['Upperbound']-df['Lowerbound'])
        df['MPIW'] = np.mean(df['Width'])
        #df['NMPIW'] = df['MPIW']/self.No_PCGrad.range

        df['Flag']= np.where((df['Upperbound'] >= df['y_true']) & (df['Lowerbound'] <= df['y_true']), 1, 0)  
        df['PICP'] =  np.mean( (df['Upperbound'] >= df['y_true']) & (df['Lowerbound'] <= df['y_true']) )
        # To sort the columns 
        df = df[['PICP','Lowerbound','y_true','Upperbound','Flag','MPIW','Width']] # 'NMPIW',
        # Save all predicted value
        dataset_name = model.dataset.split('.')[0]
        df.to_csv( f'{dataset_name}_{name}_pred.csv', header=True, index = False)
        self.result.append({'PICP':np.mean(df['PICP']),'MPIW':np.mean(df['MPIW'])}) # ,'NMPIW':np.mean(df['NMPIW'])

    def print_comparison(self):
        res = pd.DataFrame(self.result, index = ['No_PCGrad','PCGrad'])
        print(res)
        return res

    def plot_training(self, filename):
        name = filename.split('.')[0]
        dict_data = pd.read_pickle(filename)  
        df = pd.DataFrame(dict_data)
        fig = plt.figure(figsize=(10,6))
        plt.ylim(0, 1.25)
        sns.set_style("ticks")
        plt.title(name)
        plt.xlabel("Epochs")
        sns.lineplot(data=df[ ['coverage', 'mpiw','val_coverage', 'val_mpiw','val_loss']])
        plt.savefig(f'{name}.png', dpi = 300)
        plt.clf()
        #plt.show()

def main():
    
    # For comparison 1
    # datasets1 = ['1_constant_noise.csv', '2_nonconstant_noise.csv', '4_Concrete_Data.xls','5_BETAPLASMA.csv', 
    #             '6_Drybulbtemperature.xlsx' ,'7_moisture content of raw material.xlsx',
    #             '8_steam pressure.xlsx','9_main stem temperature.xlsx','10_reheat steam temperature.xlsx']
    # targets1 = ['y', 'y','Concrete compressive strength(MPa, megapascals) ','BETAPLASMA', 
    #             'dry bulb temperature', 'moisture content of raw material','steam pressure',
    #             'main stem temperature', 'reheat steam temperature']

    # For datasets2
    datasets2  = ['1_Boston_Housing.csv', '2_Concrete_Data.xls',
    '3_Energy Efficiency.csv', '4_kin8nm.csv', '5_Naval Propulsion.csv', 
    '6_Power.csv', '7_Protein.csv', '8_Wine Quality.csv', '9_Yacht.csv','10_Song_Year.csv']
    
    targets2 = ['MEDV','Concrete compressive strength(MPa, megapascals) ','Y1','y',
    'gt_t_decay','Net hourly electrical energy output','y','quality','Residuary resistance per unit weight of displacement',
    'Year']
    

    def training_data(datasets, targets):

        for index, (dataset, target) in enumerate(zip(datasets, targets)):

            # For datasets2 
            if dataset == '7_Protein.csv':
                times = 5
            elif dataset == '10_Song_Year.csv':
                times = 1
            else:
                times = 20
                
            temp = []

            for i in range(times):

                # Split the data identically each time for No_PCGrad and PCGrad
            
                # seed = 3
                seed = np.random.randint(100)

                # # To generate dataset1 and dataset2 
                if dataset == '1_constant_noise.csv':
                    generate_data1()
                if dataset == '2_nonconstant_noise.csv':
                    generate_data2()

                #Default setting to 'big' data 
                No_PCGrad = UpperLowerBound(dataset, target, seed = seed)

                # To calculate the data size 
                if No_PCGrad.y_test.shape[0] < 40.0:
                    No_PCGrad = UpperLowerBound(dataset, target, drop_out = 0.5, flag = True, seed = seed)
                    initial_weights = No_PCGrad.model.get_weights()
                    
                    PCGrad = UpperLowerBound(dataset, target, drop_out = 0.5, flag = True, seed = seed)
                else:
                    initial_weights = No_PCGrad.model.get_weights()
                    PCGrad = UpperLowerBound(dataset, target, seed = seed)
                # To ensure PCGrad and No_PCGrad with same set of initial weights
                PCGrad.model.set_weights(initial_weights)

                obj = Run(No_PCGrad, PCGrad)
                No_PCGrad_Pred = obj.run_no_pcgrad()
                obj.save_pred_results(No_PCGrad, No_PCGrad_Pred, '')

                PCGrad_Pred = obj.run_pcgrad()
                obj.save_pred_results(PCGrad, PCGrad_Pred, 'PCGrad')

                res = obj.print_comparison()
                temp.append(res)

            output = pd.concat(temp)
            name = dataset.split('.')[0]
            output.to_csv(f'{name}_Outputs.csv')

    # training_data(datasets1, targets1)
    
    training_data(datasets2, targets2)

if __name__ == "__main__":
    
    # set the epochs 
    Run.epochs = 6000
    
    # set the batch_size
    # Run.batch_size = 6000
    main()
