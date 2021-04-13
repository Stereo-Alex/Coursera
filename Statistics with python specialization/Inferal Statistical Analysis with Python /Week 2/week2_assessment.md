
## Confidence intervals in python
In this assessment, you will look at data from a study on toddler sleep habits. 

The confidence intervals you create and the questions you answer in this Jupyter notebook will be used to answer questions in the following graded assignment.


```python
import numpy as np
import pandas as pd
from scipy.stats import t
pd.set_option('display.max_columns', 30) # set so can see all columns of the DataFrame
```

Your goal is to analyse data which is the result of a study that examined
differences in a number of sleep variables between napping and non-napping toddlers. Some of these
sleep variables included: Bedtime (lights-off time in decimalized time), Night Sleep Onset Time (in
decimalized time), Wake Time (sleep end time in decimalized time), Night Sleep Duration (interval
between sleep onset and sleep end in minutes), and Total 24-Hour Sleep Duration (in minutes). Note:
[Decimalized time](https://en.wikipedia.org/wiki/Decimal_time) is the representation of the time of day using units which are decimally related.   


The 20 study participants were healthy, normally developing toddlers with no sleep or behavioral
problems. These children were categorized as napping or non-napping based upon parental report of
children’s habitual sleep patterns. Researchers then verified napping status with data from actigraphy (a
non-invasive method of monitoring human rest/activity cycles by wearing of a sensor on the wrist) and
sleep diaries during the 5 days before the study assessments were made.


You are specifically interested in the results for Bedtime. 

Reference: Akacem LD, Simpkin CT, Carskadon MA, Wright KP Jr, Jenni OG, Achermann P, et al. (2015) The Timing of the Circadian Clock and Sleep Differ between Napping and Non-Napping Toddlers. PLoS ONE 10(4): e0125181. https://doi.org/10.1371/journal.pone.0125181


```python
# Import the data
df = pd.read_csv("nap_no_nap.csv") 
```


```python
# First, look at the DataFrame to get a sense of the data
df
```

**Question**: What value is used in the column 'napping' to indicate a toddler takes a nap? (see reference article)  
**Question**: What is the overall sample size $n$? What is the sample size for toddlers who nap, $n_1$, and toddlers who don't nap, $n_2$?

### Average bedtime confidence interval for napping and non napping toddlers
Create two 95% confidence intervals for the average bedtime, one for toddler who nap and one for toddlers who don't.

First, isolate the column 'night bedtime' for those who nap into a new variable, and those who didn't nap into another new variable. 


```python
bedtime_nap = 
```


```python
bedtime_no_nap = 
```

Now find the sample mean bedtime for nap and no_nap.


```python
nap_mean_bedtime = 
```


```python
no_nap_mean_bedtime = 
```

Now find the sample standard deviation for $X_{nap}$ and $X_{no\ nap}$.


```python
# The np.std function can be used to find the standard deviation. The
# ddof parameter must be set to 1 to get the sample standard deviation.
# If it is not, you will be using the population standard deviation which
# is not the correct estimator
nap_s_bedtime = 
```


```python
no_nap_s_bedtime = 
```

Now find the standard error for $\bar{X}_{nap}$ and $\bar{X}_{no\ nap}$.


```python
nap_se_mean_bedtime = 
```


```python
no_nap_se_mean_bedtime = 
```

**Question**: Given our sample sizes of $n_1$ and $n_2$ for napping and non napping toddlers respectively, how many degrees of freedom ($df$) are there for the associated $t$ distributions?

To build a 95% confidence interval, what is the value of t\*?  You can find this value using the percent point function (PPF): 
```
from scipy.stats import t

t.ppf(probability, df)
```
This will return the quantile value such that to the left of this value, the tail probabiliy is equal to the input probabiliy (for the specified degrees of freedom). 

Example: to find the $t^*$ for a 90% confidence interval, we want $t^*$ such that 90% of the density of the $t$ distribution lies between $-t^*$ and $t^*$.

Or in other words if $X \sim t(df)$:

P($-t^*$ < X < $t^*$) = .90

Which, because the $t$ distribution is symmetric, is equivalent to finding $t^*$ such that:  

P(X < $t^*$) = .95

(0.95 = 1 - (1 - confidence) / 2 = 1 - 0.1 / 2 = 1 - 0.05)

So the $t^*$ for a 90% confidence interval, and lets say df=10, will be:

t_star = t.ppf(.95, df=10)



```python
# Find the t_stars for the 95% confidence intervals
nap_t_star = 
```


```python
no_nap_t_star = 
```

**Quesion**: What is $t^*$ for nap and no nap?

Now to create our confidence intervals. For the average bedtime for nap and no nap, find the upper and lower bounds for the respective 95% confidence intervals.

**Question**: What are the 95% confidence intervals for the average bedtime for toddlers who nap and for toddlers who don't nap? 

CI = $\bar{X} \pm \ t^* \cdot s.e.(\bar{X})$
