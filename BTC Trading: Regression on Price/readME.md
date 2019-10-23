## Lognormal Regression of Bitcoin daily price

### Equation:
![equation](https://latex.codecogs.com/gif.latex?%5Clog_%7B10%7D%28Price%29%20%3D%20m%20*%20%5Clog_%7B10%7D%28%5Ctext%7Bdays%20since%20inception%7D%29%20&plus;%20b)

Where m: slope of line
      b: intercept

### Data
You will need daily BTC prices since inception of Bitcoin. There is good amount of information available in [blockchain.info](https://www.blockchain.com/charts/market-price?timespan=all)
which can be downloaded in csv format.

If you want minute or even second level data, I recommend [Bitcoincharts API Dataset](http://api.bitcoincharts.com/v1/csv/)

### Results
Modelling above equation will give you below graph

` ![title](Images/example1.png)
