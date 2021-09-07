# boot_conf_intervals_ml.py just concatenates the three gists referenced below

# example usage: 
# from boot_conf_intervals_ml import specificity_score, make_boot_df, raw_metric_samples, ci
         
# gist make_df_and_boot.py 
# https://gist.github.com/DavidRosen/3aeedbd2ccb73e636e7eb22657373b2b
import numpy as np
import pandas as pd
import re
def make_boot_df(dforig):
    """Returns one boot sample dataframe, same shape as dforig
    """
    return dforig.sample(frac=1,replace=True).reset_index(drop=True)

# gist raw_metric_samples.py https://gist.github.com/DavidRosen/4c80d1e295c39c089a62630fef878e26
from tqdm import tqdm
def raw_metric_samples(metrics, *data_args, nboots=10, sort=False,
       **metric_kwargs):
    """Return dataframe containing metric(s) for nboots boot sample datasets.
    metrics is a metric func or iterable of funcs e.g. [m1, m2, m3]
    """
    if callable(metrics): metrics=[metrics] # single metric func to list
    metrics=list(metrics) # in case it is a generator
    dforig=pd.DataFrame\
    ( { arg.name if hasattr(arg,'name') else str(i) : np.array(arg)
          for i,arg in enumerate(data_args) 
      }# end of dict comprehen.; np.array() removes index
    ) # I like ( ) to be above one another
    res=pd.DataFrame\
    ( # dictionary comprehension:
      { b:[ m( *[col_as_arg for (colname,col_as_arg) in dfboot.items()],
               **_kws_this_metric(m,**metric_kwargs)
             ) for m in metrics # list comprehension ends w/following "]":
          ] for b,dfboot in # generator expr. avoids huge mem. of *list* of df's:
            ((b,make_boot_df(dforig)) for b in tqdm(range(nboots)))
            if dfboot.iloc[:,0].nunique()>1 # >1 for log loss (no labels), roc
      }, index=[_metric_name(m) for m in metrics]
    ) # sorry but I like ( ) to be above #one #another:-)
    res.index.name="Metric (class 1 +ve)"
    return res.apply(lambda row: np.sort(row), axis=1) if sort else res
from sklearn import metrics
def specificity_score(true,pred, **kwargs):
    return metrics.recall_score(1-pd.Series(true), 1-pd.Series(pred), **kwargs)
specificity_score.__name__ = "Specificity (Recall of -ve)"
def _kws_this_metric(m,**metric_kwargs):
    # dict of just those metric_kwargs that this metric accepts
    return \
    { k: metric_kwargs[k] for k in metric_kwargs.keys()
           & m.__wrapped__.__kwdefaults__.keys() # intersect
    } if ( hasattr(m,"__wrapped__") and
               hasattr(m.__wrapped__,"__kwdefaults__") and
               isinstance(m.__wrapped__.__kwdefaults__, dict) 
         ) else dict() # no keywords if attribs aren't there
def _metric_name(m):
    name=re.sub(' score$','',m.__name__.replace('_',' '))
    return name.title() if name.lower()==name else name

# gist ci.py 
# https://gist.github.com/DavidRosen/c85a2d075f64e0c9fd02e5bfbc968eb0
DFLT_NBOOTS=500
def ci(metrics, *data_args, quantiles=[0.025,0.975], 
           nboots=DFLT_NBOOTS, **metric_kwargs):
    """Return Pandas data frame of bootstrap confidence intervals. 
    PARAMETERS:
    metrics : a metric func or iterable of funcs e.g. [m1, m2, m3]
    data_args : 1+ (often 2, e.g. ytrue,ypred) iterables for metric 
    quantiles : [upper,lower] if for 1 CI (dflt is for 95% 2-tail)
    nboots : number of bootstrap samples drawn from data (dflt 500)
    metric_kwargs : each metric gets any KW's it accepts optionally
    """ # non-std expr fmt: ( ) above one another unless same line
    if callable(metrics): metrics=[metrics] # single metric func to list
    metrics=list(metrics) # in case it is a generator
    result=raw_metric_samples\
      (metrics, *data_args, nboots=nboots, sort=False, **metric_kwargs)
    resboots=result.shape[1]
    if resboots<nboots: print\
            ( f'Note: {nboots-resboots} bootstrap '
              f'samples dropped (out of {nboots}) because all '
              f'values were identical within first data arg'
            ) # results may be biased if too many were dropped
    result=result.apply(lambda row: row.quantile(quantiles),axis=1)
    result.columns=[f'{q*100}%ile' for q in quantiles]
    result.columns.name=f"{resboots} Boot Samples"
    result.insert\
    ( 0, "Point Estim", [ m( *[data_arg for data_arg in data_args], 
                             **_kws_this_metric(m,**metric_kwargs)
                           ) for m in metrics
                        ]# end of list compreh. of all metrics for col
    ) # inserted (in place) point estim col as first col (all metrics)
    return result
