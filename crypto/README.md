### features
using ta-lib

### kfold
In this competition I used 3-fold, walk-forward, grouped cross validation. 
The group key was the timestamp. In a typical setup, train, gap, validation and test takes certain porpotions.
The folds could have some interconnects.

### forcast
Ta-lib can extracts features from the origin time series, each timestamp has many features extracted by ta-lib.
And the target is the log-return.

### evaluate

wPCC can be used and be treated as the key model selection index.
