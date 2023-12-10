### G_Research_Crypto_Forecasting

##### Data
Millions of rows of minute-by-minute cryptocurrency trading data dating back to 2018 

- timestamp: Timestamps in this dataset are multiple of 60, indicating minute-b minute data.
- Asset_ID: The asset ID corresponding to one of the cryptocurrencies.
- Count: Total number of trades in the time interval (last minute).
- Open: Opening price of the time interval (in USD).
- High: Highest price reached during time interval (in USD).
- Low: Lowest price reached during time interval (in USD).
- Close: Closing price of the time interval (in USD).
- Volume: Quantity of asset bought or sold, displayed in base currency USD.
- VWAP: The average price of the asset over the time interval, weighted by volume. 
- Target: Residual log-returns for the asset over a 15 minute horizon

##### My job
Predict price returns across 14 major cryptocurrencies, in the time scale of minutes to hours.

Your predictions will be evaluated by how much they correlate with real market data collected during the future three-month evaluation period.

In advance, you are encouraged to perform additional statistical analyses to have a stronger grasp on the dataset, including autocorrelation, time-series decomposition and stationarity tests.

##### Predict

[refer1][refer1]

Evaluation metrics: weighted Pearson Correlation Coefficient [(wPCC)][(wPCC)] of your prediction and the real data value.

- The danger of overfitting should be considerable.

- The volatility and correlation structure in the data are likely to be highly nonstationary.

- Changes in prices between different cryptocurrencies are highly interconnected. For example, Bitcoin has historically been a major driver of price changes across cryptocurrencies but other coins also impact the market.不同加密货币之间的价格变动高度相互关联。例如，比特币在历史上一直是影响整个加密货币市场价格变动的重要因素，但其他币种也会对市场产生影响。


[submission](https://www.kaggle.com/code/sohier/basic-submission-template/notebook)
```
import gresearch_crypto
env = gresearch_crypto.make_env()
iter_test = env.iter_test()
for (test_df, sample_prediction_df) in iter_test:
    sample_prediction_df['Target'] = 0
    env.predict(sample_prediction_df)
```

[API](https://www.kaggle.com/code/sohier/detailed-api-introduction/notebook)

[ARIMA](https://www.kaggle.com/code/girishkumarsahu/g-research-crypto-forecasting-v2)

##### Ta-lib

```
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz

tar -xzf ta-lib-0.4.0-src.tar.gz

cd ta-lib/
```

[IF NO ROOT ][1]， ./configure --prefix=/home/jwangiy/crypto_forecast/ is the path

```
./configure --prefix=/home/jwangiy/crypto_forecast/
make
make install
```

setup.py:


```

include_dirs = [
    '/usr/include',
    '/usr/local/include',
    '/opt/include',
    '/opt/local/include',
    '/opt/homebrew/include',
    '/opt/homebrew/opt/ta-lib/include',
    '/home/jwangiy/crypto_forecast/include',
]

library_dirs = [
    '/usr/lib',
    '/usr/local/lib',
    '/usr/lib64',
    '/usr/local/lib64',
    '/opt/lib',
    '/opt/local/lib',
    '/opt/homebrew/lib',
    '/opt/homebrew/opt/ta-lib/lib',
    '/home/jwangiy/crypto_forecast/lib',
]
```

Lastly, 

```
python setup.py build
python setup.py install
```



[1]: https://zhuanlan.zhihu.com/p/647474788#:~:text=%E5%8F%82%E8%80%83%20%E4%B8%AD%E6%96%87%E6%95%99%E7%A8%8B%20%E5%92%8C%20%E5%AE%98%E6%96%B9%E6%95%99%E7%A8%8B%20%EF%BC%8C%E4%B8%8B%E8%BD%BD%20ta-lib-0.4.0-src.tar.gz%20%E5%8C%85%EF%BC%8C%E4%B8%8B%E8%BD%BD%E5%90%8E%E8%A7%A3%E5%8E%8B%EF%BC%8C%E5%BE%97%E5%88%B0%20ta-lib%E6%96%87%E4%BB%B6%E5%A4%B9%E5%A6%82%E6%9E%9C%E6%9C%89root%E6%9D%83%E9%99%90%E5%B0%B1%E7%94%A8sudo%EF%BC%8C%E5%86%99%E5%88%B0%2Fusr%2F%E4%B8%8B%EF%BC%8C%E6%B2%A1%E6%9C%89root%E7%9A%84%E8%AF%9D%E4%B9%9F%E5%8F%AF%E4%BB%A5%EF%BC%8C%E8%AE%BE%E7%BD%AEprefix%E4%B8%BA%E8%87%AA%E5%B7%B1%E6%9C%89%E6%9D%83%E9%99%90%E7%9A%84%E7%9B%AE%E5%BD%95%E4%B8%8B%EF%BC%8C%E4%BE%8B%E5%A6%82%EF%BC%9A%2Fdata%2Fjarvix%2Fusr%2F%20sudo.%2Fconfigure%20--prefix%3D%2Fusr%20sudo%20make%20sudo%20make%20install

[refer1]: https://www.kaggle.com/code/cstein06/tutorial-to-the-g-research-crypto-competition/notebook

[(wPCC)]: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient


