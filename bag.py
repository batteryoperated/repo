# Python wrapper for R forecast stuff
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from collections import defaultdict
from rpy2.rinterface import RRuntimeError
import holtwinters as hw
import es

def inv_distances(dists):
    return 1 / (0.0001 + dists)



def dt_parse(date, hour):
    date_string = date + ' ' + hour
    dt = pd.datetime.strptime(date_string, '%Y-%m-%d %H')
    return dt 

def calculate_squareLogError(prediction, target):
    return np.square(np.log(prediction + 1) - np.log(target + 1))

def calculate_rmsle(square_log_errors):
    return np.sqrt(np.mean(square_log_errors))  

print 'Start importing R.'
import rpy2
from rpy2 import robjects 
from rpy2.robjects.packages import importr
from rpy2.robjects.numpy2ri import numpy2ri
rpy2.robjects.numpy2ri.activate()
#robjects.conversion.py2ri = numpy2ri

base = importr('base')
forecast = importr('forecast')
stats = importr('stats')
ts = robjects.r['ts']
print 'Finished importing R.'

#dimnames = robjects.r.list(robjects.StrVector([str(i) for i in range(100)]),
#                          robjects.StrVector(['c'+str(i) for i in range(2)]))
                          
def nparray2rmatrix(x):
  nr, nc = x.shape
  #xvec = robjects.FloatVector(x.reshape((x.size)))
  xr = robjects.r.matrix(x, nrow=nr, ncol=nc, dimnames=dimnames, byrow=True)
  #print xr.dimnames
  return xr

def nparray2rmatrix_alternative(x):
  nr, nc = x.shape
  xvec = robjects.FloatVector(x.reshape((x.size)))
  xr = robjects.r.matrix(xvec, nrow=nr, ncol=nc, dimnames=dimnames, byrow=True)
  return xr

def do_forecast(series, frequency=None, horizon=30, summary=False, exog=None):
  if frequency:
    #series = ts(series, frequency=frequency)
    series = forecast.msts(series, seasonal_periods=frequency)
  else:
    series = ts(series)
  if exog:
    exog_train, exog_test = exog
    r_exog_train = nparray2rmatrix(exog_train)
    r_exog_test = nparray2rmatrix(exog_test)
    order = robjects.IntVector([1, 0, 2]) # c(1,0,2) # TODO find right model
    fit = forecast.Arima(series, order=order, xreg=r_exog_train)
    forecast_result = forecast.forecast(fit, h=horizon, xreg=r_exog_test)
  else:
    # fit = forecast.auto_arima(series)
    #robjects.r.plot(series)
    #fit = stats.HoltWinters(series)
    fit = forecast.tbats(series, use_box_cox=True, use_trend=True, use_damped_trend=True, num_cores=8)
    forecast_result = forecast.forecast(fit, h=horizon)
    #m = 168
    #fit = forecast.Arima(y, order=c(2,0,1), xreg=forecast.fourier(1:n,4,m))
    #plot(forecast(fit, h=2*m, xreg=fourier(n+1:(2*m),4,m)))
  if summary:
    modsummary = base.summary(fit)
    print modsummary
  forecast_values = np.array(list(forecast_result.rx2('mean')))
  return forecast_values
  
      
def holtwinters(y, alpha, beta, gamma, c, debug=True):
    """
    y - time series data.
    alpha , beta, gamma - exponential smoothing coefficients 
                                      for level, trend, seasonal components.
    c -  extrapolated future data points.
          4 quarterly
          7 weekly.
          12 monthly
 
 
    The length of y must be a an integer multiple  (> 2) of c.
    """
    #Compute initial b and intercept using the first two complete c periods.
    ylen =len(y)
    if ylen % c !=0:
        return None
    fc =float(c)
    ybar2 =sum([y[i] for i in range(c, 2 * c)])/ fc
    ybar1 =sum([y[i] for i in range(c)]) / fc
    b0 =(ybar2 - ybar1) / fc
    if debug: print "b0 = ", b0
 
    #Compute for the level estimate a0 using b0 above.
    tbar  =sum(i for i in range(1, c+1)) / fc
    print tbar
    a0 =ybar1  - b0 * tbar
    if debug: print "a0 = ", a0
 
    #Compute for initial indices
    I =[y[i] / (a0 + (i+1) * b0) for i in range(0, ylen)]
    if debug: print "Initial indices = ", I
 
    S=[0] * (ylen+ c)
    for i in range(c):
        S[i] =(I[i] + I[i+c]) / 2.0
 
    #Normalize so S[i] for i in [0, c)  will add to c.
    tS =c / sum([S[i] for i in range(c)])
    for i in range(c):
        S[i] *=tS
        if debug: print "S[",i,"]=", S[i]
 
    # Holt - winters proper ...
    if debug: print "Use Holt Winters formulae"
    F =[0] * (ylen+ c)   
 
    At =a0
    Bt =b0
    for i in range(ylen):
        Atm1 =At
        Btm1 =Bt
        At =alpha * y[i] / S[i] + (1.0-alpha) * (Atm1 + Btm1)
        Bt =beta * (At - Atm1) + (1- beta) * Btm1
        S[i+c] =gamma * y[i] / At + (1.0 - gamma) * S[i]
        F[i]=(a0 + b0 * (i+1)) * S[i]        
        print "i=", i+1, "y=", y[i], "S=", S[i], "Atm1=", Atm1, "Btm1=",Btm1, "At=", At, "Bt=", Bt, "S[i+c]=", S[i+c], "F=", F[i]
        print i,y[i],  F[i]
    #Forecast for next c periods:
    for m in range(5):
        print "forecast:", (At + Bt* (m+1))* S[ylen + m]


# Example
#series = np.arange(100)
#exog_train = np.ones((100,2))
#exog_test = np.ones((100,2))
#horizon = 100
#res = do_forecast(series, horizon=horizon, exog=(exog_train, exog_test))
#res = do_forecast(series, horizon=horizon)
#print res

data = pd.read_csv('hour.csv',  parse_dates = {'datetime':[1, 5]}, index_col='datetime',
                        date_parser=dt_parse, keep_date_col=True)


date_index = pd.date_range(pd.datetime(2011, 1, 1, 0, 0, 0), pd.datetime(2012, 12, 31, 23, 0, 0, 0), freq='H')
data = data.reindex(date_index)
data.drop('instant', 1, inplace=True)
data.rename(columns={'cnt':'count', 'hum':'humidity', 'yr':'year', 'mnth':'month', 'hr':'hour', 'atemp':'airtemp'}, inplace=True)

#fill in missing values
data['casual'].fillna(0, inplace=True)
data['registered'].fillna(0, inplace=True)
data['count'].fillna(0, inplace=True)
data['hour'] = data.index.hour
data['weekday'] = data.index.weekday
data.fillna(method='ffill', inplace=True)

# for season and weathersit we use one-hot encoding
one_hot_season = pd.get_dummies(data['season'], 'season', '_')
data.drop('season', axis=1, inplace=True)
data = pd.concat([data, one_hot_season], axis=1)

one_hot_weathersit = pd.get_dummies(data['weathersit'], 'weathersit', '_')
data.drop('weathersit', axis=1, inplace=True)
data = pd.concat([data, one_hot_weathersit], axis=1)

y = data['count'].values
v = np.where(y>0, np.log10(y), 0)
start = 0
end = 673

results = pd.DataFrame(data.ix[end-1::, 'count'])
results.rename(columns={'count':'target'}, inplace=True)
results['prediction'] = 0.0
results['squared_logarithmic_error'] = 0.0

features = data
features = features.drop({'dteday', 'holiday', 'airtemp', 'registered', 'casual', 'count'}, 1)
features.reset_index(inplace=True)
features = features.drop('index', 1)

#bestfit = np.inf
#for i in range(1,25):

#  fit = forecast.auto_arima(gas, xreg=fourier(gas, K=i), seasonal=FALSE)
#  if fit.rx2('aicc') < bestfit
#        bestfit = fit
#  else break;

#fc <- forecast(bestfit, xreg=fourierf(gas, K=12, h=104))




for i in range(len(results)):
#for i in range(30):   

        test_y = y[end-1:end]

        knn_x = features[0:end-1]
        test_knn_x = features[end-1:end]
        #rexogtrain = nparray2rmatrix(X)
        #rexogtest = nparray2rmatrix(testX)
        k_params =  [3, 6]
        knn_predictions = defaultdict(list)
        knn_y = v[0:end-1]
        for k in k_params:
            for j in range(0,3):
                estimator = KNeighborsRegressor(n_neighbors = k, weights = inv_distances)
                estimator.fit(knn_x, knn_y)
                knn_predictions[k] = estimator.predict(test_knn_x)

        ar_y = v[start:end:168]
        train = ar_y[0:-1]
        #train = y[start:end-1]
        #freq = robjects.IntVector([24, 168]) 
        #if len(train_y) >= 2016 :
        #    freq = robjects.IntVector([24, 168, 672, 2016]) 
        #if len(train_y) >= 8760 :
        #    freq = robjects.IntVector([24, 168, 672, 2016, 8760]) 
        #elif len(train_y) >= 8820 :
        #    freq = robjects.IntVector([24, 168, 672, 2016, 8784])
            
        #res = do_forecast(train_y, frequency=freq, horizon=1)
        
        
        #r_exog_train = nparray2rmatrix(exog_train)
        #r_exog_test = nparray2rmatrix(exog_test)
        #order = robjects.IntVector([1, 0, 2]) # c(1,0,2) # TODO find right model
        #fit = forecast.Arima(series, order=order, xreg=r_exog_train)
        #forecast_result = forecast.forecast(fit, h=horizon, xreg=r_exog_test)
        
        
        
        # new_x is a vector of newly observed daily data with length h
        ry = ts(train)
        #k=1
        #print(y.r_repr())
        #for k in range(1,25):
        #z = forecast.fourier(ts(train, frequency=168), K=k)
    #print(z.r_repr())
        #zf = forecast.fourierf(ts(train, frequency=168), K=k, h=2)
    
        #fitarima = forecast.auto_arima(ry,  seasonal=False)
        #arima_result = forecast.forecast(fitarima)
        #arimares = np.array(list(arima_result.rx2('mean')))
        
        
        
        #fit = forecast.holt(ry, h=1, damped=False, initial="simple", exponential=False, seasonal='additive')
        fit = forecast.ses(ry, h=1)
        #print "K: %d AIC:" % k
        #print fit.rx2('aicc')
        #forecast_result = forecast.forecast(fit, xreg=zf, h=2)
        forecast_result = forecast.forecast(fit)
        res = np.array(list(forecast_result.rx2('mean')))
        
        #print res
        res = [res[0]] if res[0] > 0 else np.array([0.0])
        train_y = y[start:end-1]
        #train_y = pd.ewma(train_y, span=len(train_y))        
        #ewma_predictions = train_y[-1]
        #predictions = np.mean([ewma_predictions, knn_predictions[0]])
        #hwpredictions, alpha, beta, gamma = hw.multiplicative(train_y.tolist(), 168, 1)
        #predictions = es.exp_smoothing(train_y, 0.2, 0.1, 0.1, 168, damp=1,
        #          trend='multiplicative', forecast=1, season='multiplicative')
        #hwpredictions = np.asarray(hwpredictions)
        #hwpredictions = [hwpredictions[0]] if hwpredictions[0] > 0 else np.array([0.0])
        #train_y = pd.ewma(train_y, span=len(train_y))        
        #ewma_predictions = train_y[-1]
        #seasonal="additive",
       
        #print "K: %d AIC:" % k
        #print fit.rx2('aicc')
        #forecast_result = forecast.forecast(fit, xreg=zf, h=2)
        #try:
        #    hw = forecast.holt(ry, h=1,  damped=False, initial="simple", exponential=True)
        #    hw_result = forecast.forecast(hw)
        #    hwpredictions = np.array(list(hw_result.rx2('mean')))
        #    hwpredictions = [hwpredictions[0]] if hwpredictions[0] > 0 else np.array([0.0])
        #except RRuntimeError:
        #    hwpredictions = np.array([0.0])

        
        #predictions = np.mean([hwpredictions[0], arimares[0], train[-1], knn_y[-1], res[0], knn_predictions[1][0], knn_predictions[3][0], knn_predictions[6][0]])
        predictions = 10 ** np.mean([res[0], knn_predictions[3][0], knn_predictions[6][0]])
        #print predictions
        #predictions = np.mean([hwpredictions, res[0], knn_predictions1[0]])
        #predictions, alpha, beta, gamma = hw.multiplicative(train_y.tolist(), 168, 1)
        #predictions = es.exp_smoothing(train_y, 0.2, 0.1, 0.1, 168, damp=1,
        #          trend='multiplicative', forecast=1, season='multiplicative')
        #predictions = np.asarray(predictions)
        #holtwinters(train_y, 0.2, 0.1, 0.1, 24)
        error = calculate_squareLogError(predictions, test_y[0])
        results['prediction'][i] = np.abs(predictions)
        results['squared_logarithmic_error'][i] = error
        #print i
        start += 1
        end += 1

rmsle_score = calculate_rmsle(results['squared_logarithmic_error'])

fig, axs = plt.subplots(1, 2, figsize=(15, 5))
targ_chart = results['target'].plot(title='Bike Share Actual Usage', ax=axs[0])
targ_chart.set_xlabel("Date")
targ_chart.set_ylabel("Bike Rental Count")

res_chart = results['prediction'].plot(title='Combined Log10 predictions. RMSLE('+str(rmsle_score)+')', ax=axs[1], color='green')
res_chart.set_xlabel("Date")
res_chart.set_ylabel("Bike Rental Count")
plt.savefig('baglog', dpi=150)