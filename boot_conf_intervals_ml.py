# boot_conf_intervals_ml.py just concatenates the gists referenced below

# Example usage: 
# from boot_conf_intervals_ml import ci # enough to calc conf intervals
#
# But for more detailed usage like in the article, you might want:
# from boot_conf_intervals_ml import ci, data_args_to_df, make_boot_df \
#    , specificity_score, raw_metric_samples, metric_boot_histogram \
#    #don't need:#     , metrics_from_df, metrics_from_data_args
         
# https://gist.github.com/DavidRosen/356fd5bc7fe74d444c9e971680fadd5e
import numpy as np
import pandas as pd
import re
def data_args_to_df(*data_args):
    return pd.DataFrame( { arg.name if hasattr(arg,'name') else str(i) :
                           np.array(arg)   for i,arg in enumerate(data_args)
                         }# end of dict comprehen.; np.array() removes index
                       )
def make_boot_df(dforig):
    """Returns one boot sample dataframe, same shape as dforig
    """
    return dforig.sample(frac=1,replace=True).reset_index(drop=True)

# https://gist.github.com/DavidRosen/20dc3143d1354eba911a995fb75f81d7
from sklearn import metrics
def specificity_score(true, pred, **kwargs):
    return metrics.recall_score(1-pd.Series(true), 1-pd.Series(pred), **kwargs)
specificity_score.__name__ = "Specificity (Recall of −ve)"

def metrics_from_df(metrics, df, **metric_kwargs):
    data_args=[col_as_arg for (colname,col_as_arg) in df.items()]
    return metrics_from_data_args(metrics, *data_args, **metric_kwargs)
def metrics_from_data_args(metrics, *data_args, **metric_kwargs):
    if callable(metrics): metrics=[metrics] # single metric func to list of
    return   [m(*data_args, **_kws_this_metric(m,**metric_kwargs)) for m in 
      metrics]#end of list comprehension of all metrics applied to *data_args

def _get_default_keywords(metric): # how to make it work on specificity_score??
    return metric.__wrapped__.__kwdefaults__ if \
      ( hasattr(metric,"__wrapped__") and
        hasattr(metric.__wrapped__,"__kwdefaults__") and
        isinstance(metric.__wrapped__.__kwdefaults__, dict) 
      ) else dict() # empty dict if no __kwdefaults__ dict
def _kws_this_metric(metric,**metric_kwargs): # kwargs this metric accepts
    return { k: metric_kwargs[k] for k in # set intersection of keys:
             metric_kwargs.keys() & _get_default_keywords(metric).keys() 
           }# end of dict comprehension to convert keys set back to dict
def _metric_name(metric):
    name=re.sub(' score$','',metric.__name__.replace('_',' '))
    return name.title() if name.lower()==name else name

# https://gist.github.com/DavidRosen/dd79e5e46ae53c2bde63c8a07460a044
from tqdm import trange
DFLT_NBOOTS=500
def raw_metric_samples(metrics, *data_args, nboots=DFLT_NBOOTS, sort=False,
       **metric_kwargs):
    """Return dataframe containing metric(s) for nboots boot sample datasets.
    metrics is a metric func or iterable of funcs e.g. [m1, m2, m3]
    """
    if callable(metrics): metrics=[metrics] # single metric func to list of
    metrics=list(metrics) # in case it is a generator
    dforig=data_args_to_df(*data_args)
    res=pd.DataFrame( { b: metrics_from_df(metrics, dfboot, **metric_kwargs)
                           for b,dfboot in # generator expr, not huge df list!
                           ((b,make_boot_df(dforig)) for b in trange(nboots))
                           if dfboot.iloc[:,0].nunique()>1 # >1: log loss, roc
                      }, index=[_metric_name(m) for m in metrics]
                    ) # above { b: ... for b,dfboot in... } is dict comprehen.
    res.index.name="Metric (class 1 +ve)"
    return res.apply(lambda row: np.sort(row), axis=1) if sort else res

# https://gist.github.com/DavidRosen/74c35f12ead6a984649f7d6efb9895d2
import matplotlib, matplotlib.ticker as mtick
DFLT_QUANTILES=[0.025,0.975]
def metric_boot_histogram( metric, *data_args, quantiles=DFLT_QUANTILES, 
                           nboots=DFLT_NBOOTS, **metric_kwargs
                         ):
    point = metric(*data_args, **metric_kwargs)
    data = raw_metric_samples(metric, *data_args, **metric_kwargs).transpose()
    (lower, upper) = data.quantile(quantiles).iloc[:,0]
    import seaborn; seaborn.set_style('whitegrid')  #optional
    matplotlib.rcParams["figure.dpi"] = 300
    ax = data.hist(bins=50, figsize=(5, 2), alpha=0.4)[0][0]
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    for x in lower, point, upper:
        ax.plot([x, x],[0, 30], lw=2.5)

# https://gist.github.com/DavidRosen/feddfcc657110b56cc6e4a49e20fa4db
def ci( metrics, *data_args, quantiles=DFLT_QUANTILES, 
        nboots=DFLT_NBOOTS, **metric_kwargs
      ):
    """Return Pandas data frame of bootstrap confidence intervals. 
    PARAMETERS:
    metrics : a metric func or iterable of funcs e.g. [m1, m2, m3]
    data_args : 1+ (often 2, e.g. ytrue,ypred) iterables for metric 
    quantiles : [upper,lower] if for 1 CI (dflt is for 95% 2-tail)
    nboots : number of bootstrap samples drawn from data (dflt 500)
    metric_kwargs : each metric gets any KW's it accepts optionally
    """
    if callable(metrics): metrics=[metrics] # single metric func to list of
    metrics=list(metrics) # in case it is a generator [expr]
    result=raw_metric_samples(metrics, *data_args, nboots=nboots, sort=False,
              **metric_kwargs)
    resboots=result.shape[1]
    if resboots<nboots:  # results may be biased if too many were dropped
        print( f'Note: {nboots-resboots} boot sample datasets dropped \n'
               f'(out of {nboots}) because all vals were same in 1st data arg.'
             ) # only 1st data arg because it is probably true target arg
    result=result.apply(lambda row: row.quantile(quantiles),axis=1)
    result.columns=[f'{q*100}%ile' for q in quantiles]
    result.columns.name=f"{resboots} Boot Samples"
    result.insert( 0, "Point Estim", metrics_from_data_args(metrics, 
                                *data_args, **metric_kwargs)
                 ) # inserted (in place) all point estims as first col
    return result