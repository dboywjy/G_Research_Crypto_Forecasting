import numpy as np
import pandas as pd
import talib 
from talib import abstract
from joblib import delayed,Parallel,cpu_count
import math
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor
from joblib import dump, load
from tqdm import tqdm

import numpy as np
import pandas as pd
import talib 
from talib import abstract
from joblib import delayed,Parallel,cpu_count
import math
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor
from joblib import dump, load
import math

import numpy as np
import pandas as pd
import talib 
from talib import abstract
from joblib import delayed,Parallel,cpu_count
cpu_nums=cpu_count()
slice_windows=30

ta_factors=['HT_DCPERIOD','HT_DCPHASE','HT_PHASOR','HT_SINE','HT_TRENDMODE','ADD',
    'DIV','APO','BOP','CCI','CMO','MACD','MFI','MOM','PPO','ROC','RSI','TRIX','WILLR',
    'BBANDS','DEMA','EMA','HT_TRENDLINE','KAMA','MA','MAMA','MIDPOINT','MIDPRICE','SAR',
    'SAREXT','SMA','T3','TEMA','TRIMA','WMA','AVGPRICE','MEDPRICE','TYPPRICE','WCLPRICE',
    'LINEARREG','LINEARREG_ANGLE','LINEARREG_INTERCEPT','LINEARREG_SLOPE','STDDEV','TSF',
    'TRANGE','AD','ADOSC','OBV']

def generate_talib_factor(df, ta_factor):
    target_fun=abstract.Function(ta_factor)
    ta_values=target_fun(df)
    #if-else below to cope with different result that some talib functions will return multi-columns while others only one
    #eg. MAXMIN return MAX and MIN two values
    if len(ta_values.shape)==1:
        ta_values=pd.DataFrame(ta_values,columns=[f'{ta_factor}'])
    else:
        ta_values=pd.DataFrame(ta_values)
        colname=ta_values.columns
        new_colname={}
        for col in colname:
            new_colname.update( {col:f'{col}'})
        ta_values.rename(columns=new_colname,inplace=True)
    return ta_values

class MsData():
    def __init__(self, df, method, target='Target', timestamp_fill = False, timestamp_val = 'timestamp', spine_order=3):
        '''
        df: dataframe, should be sorted
        timestamp_fill: True, fill the gap timestamp, else False
        method: ['None', '0', 'mean', 'median' ,'ffill', 'bfill', 'linear', 'spline', 'drop']
        '''
        self.df = df
        self.target = target
        self.timestamp_fill = timestamp_fill
        self.timestamp_val = timestamp_val
        self.method = method
        self.spine_order = spine_order
        
    def fillin(self):
        if self.timestamp_fill:
            self.df = self.df.set_index(self.timestamp_val)
            self.df = self.df.reindex(range(self.df.index[0],self.df.index[-1]+60,60),method=None)
            self.df['timestamp'] = self.df.index
            self.df.reset_index(drop=True, inplace=True)
        else:
            pass
        
        if self.method == 'None':
            pass
        elif self.method == "0":
            self.df[self.target].fillna(0, inplace=True)
        elif self.method == "mean":
            self.df[self.target].fillna(df.mean(), inplace=True)
        elif self.method == "median":
            self.df[self.target].fillna(df.median(), inplace=True)
        elif self.method == "ffill":
            self.df[self.target].fillna(method='ffill', inplace=True)
        elif self.method == "bfill":
            self.df[self.target].fillna(method='bfill', inplace=True)
        elif self.method == "linear":
            self.df[self.target].interpolate(method='linear', inplace=True)
        elif self.method == "spline":
            self.df[self.target].interpolate(method='spline', order=self.spine_order, inplace=True)
        elif self.method == "drop":
            self.df = self.df[self.df[self.target].isnull() == False]
        else:
            pass
    def main(self):
        self.fillin()
        return self.df

class CVsample():
    def __init__(self, df, ival, cv, plength, ptrain, pgap):
        '''
        df: Dataframe
        ival: variable used to cut the df
        length: how long each fold takes in [0, 1]
        cv: how many folds you need
        '''
        self.ival = ival
        self.df = df.sort_values(by=[ival])
        self.cv = cv
        self.plength = plength
        self.ptrain = ptrain
        self.pgap = pgap
        
    def get_intervals(self):
        '''
        p_train: how long you need for train in each fold in [0, 1]
        gap: the gap between train and validation in [0, 1]
        '''
        num = self.df.shape[0]
        train_starts = self.df.iloc[:num - math.floor(num*self.plength)].sample(self.cv, random_state=0)[self.ival].to_list()
        train_ends = [self.df[self.df[self.ival]>train_start].iloc[:math.floor(num*self.plength*self.ptrain)][self.ival].max() for train_start in train_starts]
        gap_ends = [self.df[self.df[self.ival]>train_end].iloc[:math.floor(num*self.plength*self.pgap)][self.ival].max() for train_end in train_ends]
        validation_ends = [self.df[self.df[self.ival]>gap_end].iloc[:math.floor(num*(self.plength-self.plength*self.ptrain-self.plength*self.pgap))][self.ival].max() for gap_end in gap_ends]
        self.train_starts = train_starts
        self.train_ends = train_ends
        self.gap_ends = gap_ends
        self.validation_ends = validation_ends
        
    def get_df(self, i):
        '''
        ith cv
        '''
        train = self.df[(self.df[self.ival] > self.train_starts[i]) & (self.df[self.ival] <= self.train_ends[i])]
        gap = self.df[(self.df[self.ival] > self.train_ends[i]) & (self.df[self.ival] <= self.gap_ends[i])]
        validation = self.df[(self.df[self.ival] > self.gap_ends[i]) & (self.df[self.ival] <= self.validation_ends[i])]
        return [train, gap, validation]
    
    def main(self):
        self.get_intervals()





class talib():
    def __init__(self, df):
        self.df = df

    def main(self):        
        train_data=self.df
        train_data['amount']=train_data['Volume']*train_data['VWAP']
        train_data=train_data.rename(columns={'Close':'close','Open':'open','High':'high','Low':'low','Volume':'volume'})

        panel=pd.concat( [generate_talib_factor(train_data,factor) for factor in ta_factors],axis=1)
        panel=(panel-panel.rolling(slice_windows).mean())/panel.rolling(slice_windows).std()
        panel=pd.concat([panel,train_data[['Target','timestamp']]],axis=1)
#         panel.to_csv(self.savePath, encoding='gbk')
        return panel




class lgb_forcast():
    def __init__(self, train, validation, lgb_params):
        self.train = train
        self.validation = validation
        self.lgb_params = lgb_params
        
    def lgb_train(self):
        feacols = [col for col in self.train.columns if col not in ['Target', 'timestamp']]

        lgr = LGBMRegressor(**self.lgb_params)
        lgr.fit(self.train[feacols], self.train['Target'])
        
        predict = lgr.predict(self.validation[feacols])
        self.out_params = {
            'cov':np.cov(predict, self.validation['Target']),
            'train_time_min':self.train['timestamp'].min(),
            'train_time_max':self.train['timestamp'].max(),
            'test_time_min':self.validation['timestamp'].min(),
            'test_time_max':self.validation['timestamp'].max(),
            }
        self.lgr = lgr
        
    def save(self):
        dump(self.lgr, f'../data/saved_model/model_lgb{str(self.Asset_ID).zfill(2)}.pkl')
        dump(self.out_params, f'../data/saved_model/model_lgb{str(self.Asset_ID).zfill(2)}_out_params.pkl')
     
    def main(self):
        self.lgb_train()
        self.save()
        
        
        