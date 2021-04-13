
# Hypothesis Testing

In this notebook we demonstrate formal hypothesis testing using the NHANES data.

It is important to note that the NHANES data are a "complex survey".  The data are not an independent and representative sample from the target population.  Proper analysis of complex survey data should make use of additional information about how the data were collected.  Since complex survey analysis is a somewhat specialized topic, we ignore this aspect of the data here, and analyze the NHANES data as if it were an independent and identically distributed sample from a population.


```python
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # workaround, there may be a better way
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats.distributions as dist
```

Below we read the data, and convert some of the integer codes to text values.


```python
url = "nhanes_2015_2016.csv"
da = pd.read_csv(url)

da["SMQ020x"] = da.SMQ020.replace({1: "Yes", 2: "No", 7: np.nan, 9: np.nan})
```


```python
da["SMQ020x"].head()
```




    0    Yes
    1    Yes
    2    Yes
    3     No
    4     No
    Name: SMQ020x, dtype: object




```python
da["RIAGENDRx"] = da.RIAGENDR.replace({1: "Male", 2: "Female"})

da["RIAGENDRx"].head()
```




    0      Male
    1      Male
    2      Male
    3    Female
    4    Female
    Name: RIAGENDRx, dtype: object



### Hypothesis Tests for One Proportion

The most basic hypothesis test may be the one-sample test for a proportion.  This test is used if we have specified a particular value as the null value for the proportion, and we wish to assess if the data are compatible with the true parameter value being equal to this specified value.  One-sample tests are not used very often in practice, because it is not very common that we have a specific fixed value to use for comparison. For illustration, imagine that the rate of lifetime smoking in another country was known to be 40%, and we wished to assess whether the rate of lifetime smoking in the US were different from 40%.  In the following notebook cell, we carry out the (two-sided) one-sample test that the population proportion of smokers is 0.4, and obtain a p-value of 0.43.  This indicates that the NHANES data are compatible with the proportion of (ever) smokers in the US being 40%. 


```python
x = da.SMQ020x.dropna() == "Yes"
```


```python
p = x.mean()
```


```python
p
```




    0.4050655021834061




```python
se = np.sqrt(.4 * (1 - .4)/ len(x))
se
```




    0.00647467353462031




```python
test_stat = (p - 0.4) / se
test_stat
```




    0.7823563854332805




```python
pvalue = 2 * dist.norm.cdf(-np.abs(test_stat))
print(test_stat, pvalue)
```

    0.7823563854332805 0.4340051581348052


The following cell carries out the same test as performed above using the Statsmodels library.  The results in the first (default) case below are slightly different from the results obtained above because Statsmodels by default uses the sample proportion instead of the null proportion when computing the standard error.  This distinction is rarely consequential, but we can specify that the null proportion should be used to calculate the standard error, and the results agree exactly with what we calculated above.  The first two lines below carry out tests using the normal approximation to the sampling distribution of the test statistic, and the third line below carries uses the exact binomial sampling distribution.  We can see here that the p-values are nearly identical in all three cases. This is expected when the sample size is large, and the proportion is not close to either 0 or 1.


```python
sm.stats.proportions_ztest(x.sum(), len(x), 0.4)
```




    (0.7807518954896244, 0.43494843171868214)




```python
sm.stats.binom_test(x.sum(), len(x), 0.4)
```




    0.4340360854459431



### Hypothesis Tests for Two Proportions

Comparative tests tend to be used much more frequently than tests comparing one population to a fixed value.  A two-sample test of proportions is used to assess whether the proportion of individuals with some trait differs between two sub-populations.  For example, we can compare the smoking rates between females and males. Since smoking rates vary strongly with age, we do this in the subpopulation of people between 20 and 25 years of age.  In the cell below, we carry out this test without using any libraries, implementing all the test procedures covered elsewhere in the course using Python code.  We find that the smoking rate for men is around 10 percentage points greater than the smoking rate for females, and this difference is statistically significant (the p-value is around 0.01).


```python
dx = da[["SMQ020x", "RIDAGEYR", "RIAGENDRx"]].dropna()

dx.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SMQ020x</th>
      <th>RIDAGEYR</th>
      <th>RIAGENDRx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Yes</td>
      <td>62</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Yes</td>
      <td>53</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Yes</td>
      <td>78</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>3</th>
      <td>No</td>
      <td>56</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>4</th>
      <td>No</td>
      <td>42</td>
      <td>Female</td>
    </tr>
  </tbody>
</table>
</div>




```python
p = dx.groupby("RIAGENDRx")["SMQ020x"].agg([lambda z: np.mean(z == "Yes"), "size"])
p.columns = ["Smoke", "N"]
print(p)
```

                  Smoke     N
    RIAGENDRx                
    Female     0.304845  2972
    Male       0.513258  2753


Essentially the same test as above can be conducted by converting the "Yes"/"No" responses to numbers (Yes=1, No=0) and conducting a two-sample t-test, as below:


```python
p_comb = (dx.SMQ020x == "Yes").mean()
va = p_comb * (1 - p_comb)

se = np.sqrt(va * (1 / p.N.Female + 1 / p.N.Male))
```


```python
(p_comb, va, se)
```




    (0.4050655021834061, 0.2409874411243111, 0.01298546309757376)




```python
test_stat = (p.Smoke.Female - p.Smoke.Male) / se
p_value = 2 * dist.norm.cdf(-np.abs(test_stat))
(test_stat, p_value)
```




    (-16.049719603652488, 5.742288777302776e-58)




```python
dx_females = dx.loc[dx.RIAGENDRx == "Female", "SMQ020x"].replace({"Yes": 1, "No": 0})
dx_females
```




    3       0
    4       0
    5       0
    7       0
    12      1
    13      0
    15      0
    16      0
    17      0
    18      1
    19      0
    21      0
    22      1
    23      0
    25      0
    27      1
    29      0
    30      1
    33      0
    34      0
    35      1
    36      0
    38      0
    39      0
    43      0
    46      0
    47      0
    50      0
    52      0
    54      0
           ..
    5678    1
    5679    0
    5681    0
    5682    1
    5683    0
    5684    0
    5685    0
    5686    0
    5689    0
    5692    0
    5696    1
    5697    0
    5699    0
    5703    1
    5704    0
    5707    0
    5708    0
    5710    0
    5712    0
    5715    0
    5716    1
    5719    1
    5721    0
    5722    0
    5723    1
    5724    0
    5727    0
    5730    1
    5732    1
    5734    0
    Name: SMQ020x, Length: 2972, dtype: int64




```python
dx_males = dx.loc[dx.RIAGENDRx == "Male", "SMQ020x"].replace({"Yes": 1, "No": 0})
dx_males
```




    0       1
    1       1
    2       1
    6       1
    8       0
    9       0
    10      1
    11      1
    14      0
    20      0
    24      0
    26      1
    28      0
    31      0
    32      1
    37      0
    40      1
    41      0
    42      0
    44      1
    45      1
    48      0
    49      1
    51      0
    53      1
    56      1
    57      0
    59      0
    60      1
    64      1
           ..
    5672    0
    5673    1
    5677    1
    5680    0
    5687    1
    5688    0
    5690    1
    5691    0
    5693    0
    5694    0
    5695    0
    5698    1
    5700    1
    5701    0
    5702    0
    5705    1
    5706    1
    5709    1
    5711    1
    5713    0
    5714    0
    5717    1
    5718    0
    5720    0
    5725    0
    5726    1
    5728    0
    5729    0
    5731    0
    5733    1
    Name: SMQ020x, Length: 2753, dtype: int64




```python
sm.stats.ttest_ind(dx_females, dx_males)
```




    (-16.42058555898443, 3.032088786691117e-59, 5723.0)



### Hypothesis Tests Comparing Means

Tests of means are similar in many ways to tests of proportions.  Just as with proportions, for comparing means there are one and two-sample tests, z-tests and t-tests, and one-sided and two-sided tests.  As with tests of proportions, one-sample tests of means are not very common, but we illustrate a one sample test in the cell below.  We compare systolic blood pressure to the fixed value 120 (which is the lower threshold for "pre-hypertension"), and find that the mean is significantly different from 120 (the point estimate of the mean is 126).


```python
dx = da[["BPXSY1", "RIDAGEYR", "RIAGENDRx"]].dropna()
dx
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BPXSY1</th>
      <th>RIDAGEYR</th>
      <th>RIAGENDRx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>128.0</td>
      <td>62</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>146.0</td>
      <td>53</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>2</th>
      <td>138.0</td>
      <td>78</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>3</th>
      <td>132.0</td>
      <td>56</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100.0</td>
      <td>42</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>5</th>
      <td>116.0</td>
      <td>72</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>6</th>
      <td>110.0</td>
      <td>22</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>7</th>
      <td>120.0</td>
      <td>32</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>9</th>
      <td>178.0</td>
      <td>56</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>10</th>
      <td>144.0</td>
      <td>46</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>11</th>
      <td>116.0</td>
      <td>45</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>12</th>
      <td>104.0</td>
      <td>30</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>13</th>
      <td>124.0</td>
      <td>67</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>14</th>
      <td>132.0</td>
      <td>67</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>15</th>
      <td>134.0</td>
      <td>57</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>16</th>
      <td>102.0</td>
      <td>19</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>17</th>
      <td>110.0</td>
      <td>24</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>18</th>
      <td>138.0</td>
      <td>27</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>19</th>
      <td>136.0</td>
      <td>54</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>20</th>
      <td>110.0</td>
      <td>49</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>21</th>
      <td>148.0</td>
      <td>80</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>22</th>
      <td>140.0</td>
      <td>69</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>23</th>
      <td>116.0</td>
      <td>58</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>24</th>
      <td>136.0</td>
      <td>56</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>25</th>
      <td>108.0</td>
      <td>27</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>26</th>
      <td>122.0</td>
      <td>22</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>27</th>
      <td>142.0</td>
      <td>60</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>28</th>
      <td>132.0</td>
      <td>51</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>29</th>
      <td>122.0</td>
      <td>68</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>30</th>
      <td>146.0</td>
      <td>69</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5702</th>
      <td>116.0</td>
      <td>38</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5703</th>
      <td>178.0</td>
      <td>64</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>5704</th>
      <td>134.0</td>
      <td>75</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>5705</th>
      <td>174.0</td>
      <td>80</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5706</th>
      <td>124.0</td>
      <td>72</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5707</th>
      <td>130.0</td>
      <td>25</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>5708</th>
      <td>102.0</td>
      <td>29</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>5709</th>
      <td>132.0</td>
      <td>38</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5711</th>
      <td>144.0</td>
      <td>62</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5712</th>
      <td>114.0</td>
      <td>27</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>5713</th>
      <td>116.0</td>
      <td>43</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5714</th>
      <td>162.0</td>
      <td>39</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5715</th>
      <td>124.0</td>
      <td>34</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>5717</th>
      <td>112.0</td>
      <td>32</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5718</th>
      <td>128.0</td>
      <td>45</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5720</th>
      <td>110.0</td>
      <td>38</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5721</th>
      <td>118.0</td>
      <td>35</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>5722</th>
      <td>114.0</td>
      <td>34</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>5723</th>
      <td>142.0</td>
      <td>72</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>5724</th>
      <td>132.0</td>
      <td>41</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>5725</th>
      <td>110.0</td>
      <td>34</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5726</th>
      <td>132.0</td>
      <td>53</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5727</th>
      <td>164.0</td>
      <td>69</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>5728</th>
      <td>112.0</td>
      <td>32</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5729</th>
      <td>112.0</td>
      <td>25</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5730</th>
      <td>112.0</td>
      <td>76</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>5731</th>
      <td>118.0</td>
      <td>26</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5732</th>
      <td>154.0</td>
      <td>80</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>5733</th>
      <td>104.0</td>
      <td>35</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5734</th>
      <td>118.0</td>
      <td>24</td>
      <td>Female</td>
    </tr>
  </tbody>
</table>
<p>5401 rows × 3 columns</p>
</div>




```python
dx = dx.loc[(dx.RIDAGEYR >= 40) & (dx.RIDAGEYR <= 50) & (dx.RIAGENDRx == "Male"), :]
dx
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BPXSY1</th>
      <th>RIDAGEYR</th>
      <th>RIAGENDRx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>144.0</td>
      <td>46</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>11</th>
      <td>116.0</td>
      <td>45</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>20</th>
      <td>110.0</td>
      <td>49</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>42</th>
      <td>128.0</td>
      <td>42</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>51</th>
      <td>118.0</td>
      <td>50</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>66</th>
      <td>124.0</td>
      <td>41</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>70</th>
      <td>104.0</td>
      <td>40</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>72</th>
      <td>140.0</td>
      <td>48</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>94</th>
      <td>112.0</td>
      <td>49</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>101</th>
      <td>104.0</td>
      <td>43</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>116</th>
      <td>124.0</td>
      <td>45</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>119</th>
      <td>132.0</td>
      <td>43</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>133</th>
      <td>134.0</td>
      <td>49</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>135</th>
      <td>120.0</td>
      <td>40</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>144</th>
      <td>130.0</td>
      <td>40</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>152</th>
      <td>154.0</td>
      <td>43</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>173</th>
      <td>112.0</td>
      <td>44</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>176</th>
      <td>102.0</td>
      <td>46</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>197</th>
      <td>136.0</td>
      <td>40</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>204</th>
      <td>120.0</td>
      <td>45</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>224</th>
      <td>104.0</td>
      <td>46</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>246</th>
      <td>192.0</td>
      <td>45</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>249</th>
      <td>152.0</td>
      <td>46</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>251</th>
      <td>156.0</td>
      <td>43</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>252</th>
      <td>152.0</td>
      <td>46</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>269</th>
      <td>106.0</td>
      <td>45</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>299</th>
      <td>148.0</td>
      <td>50</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>323</th>
      <td>116.0</td>
      <td>41</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>339</th>
      <td>114.0</td>
      <td>40</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>358</th>
      <td>98.0</td>
      <td>42</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5309</th>
      <td>144.0</td>
      <td>44</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5317</th>
      <td>124.0</td>
      <td>46</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5330</th>
      <td>118.0</td>
      <td>40</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5358</th>
      <td>114.0</td>
      <td>49</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5369</th>
      <td>114.0</td>
      <td>41</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5370</th>
      <td>136.0</td>
      <td>46</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5376</th>
      <td>142.0</td>
      <td>49</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5378</th>
      <td>110.0</td>
      <td>43</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5379</th>
      <td>138.0</td>
      <td>42</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5388</th>
      <td>128.0</td>
      <td>50</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5421</th>
      <td>116.0</td>
      <td>46</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5448</th>
      <td>162.0</td>
      <td>48</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5486</th>
      <td>116.0</td>
      <td>40</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5501</th>
      <td>132.0</td>
      <td>47</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5555</th>
      <td>124.0</td>
      <td>44</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5593</th>
      <td>126.0</td>
      <td>48</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5596</th>
      <td>146.0</td>
      <td>50</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5601</th>
      <td>114.0</td>
      <td>50</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5610</th>
      <td>106.0</td>
      <td>47</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5612</th>
      <td>124.0</td>
      <td>46</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5625</th>
      <td>114.0</td>
      <td>47</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5628</th>
      <td>104.0</td>
      <td>41</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5644</th>
      <td>134.0</td>
      <td>48</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5662</th>
      <td>146.0</td>
      <td>47</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5666</th>
      <td>106.0</td>
      <td>50</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5680</th>
      <td>134.0</td>
      <td>50</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5690</th>
      <td>138.0</td>
      <td>48</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5693</th>
      <td>96.0</td>
      <td>41</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5713</th>
      <td>116.0</td>
      <td>43</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5718</th>
      <td>128.0</td>
      <td>45</td>
      <td>Male</td>
    </tr>
  </tbody>
</table>
<p>421 rows × 3 columns</p>
</div>




```python
print(dx.BPXSY1.mean())
```

    125.86698337292161



```python
sm.stats.ztest(dx.BPXSY1, value=120)
```




    (7.469764137102597, 8.033869113167905e-14)



In the cell below, we carry out a formal test of the null hypothesis that the mean blood pressure for women between the ages of 50 and 60 is equal to the mean blood pressure of men between the ages of 50 and 60.  The results indicate that while the mean systolic blood pressure for men is slightly greater than that for women (129 mm/Hg versus 128 mm/Hg), this difference is not statistically significant. 

There are a number of different variants on the two-sample t-test. Two often-encountered variants are the t-test carried out using the t-distribution, and the t-test carried out using the normal approximation to the reference distribution of the test statistic, often called a z-test.  Below we display results from both these testing approaches.  When the sample size is large, the difference between the t-test and z-test is very small.  


```python
dx = da[["BPXSY1", "RIDAGEYR", "RIAGENDRx"]].dropna()
dx = dx.loc[(dx.RIDAGEYR >= 50) & (dx.RIDAGEYR <= 60), :]
dx.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BPXSY1</th>
      <th>RIDAGEYR</th>
      <th>RIAGENDRx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>146.0</td>
      <td>53</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>3</th>
      <td>132.0</td>
      <td>56</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>9</th>
      <td>178.0</td>
      <td>56</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>15</th>
      <td>134.0</td>
      <td>57</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>19</th>
      <td>136.0</td>
      <td>54</td>
      <td>Female</td>
    </tr>
  </tbody>
</table>
</div>




```python
bpx_female = dx.loc[dx.RIAGENDRx=="Female", "BPXSY1"]
bpx_male = dx.loc[dx.RIAGENDRx=="Male", "BPXSY1"]
print(bpx_female.mean(), bpx_male.mean())
```

    127.92561983471074 129.23829787234044



```python
print(sm.stats.ztest(bpx_female, bpx_male))
```

    (-1.105435895556249, 0.2689707570859362)



```python
print(sm.stats.ttest_ind(bpx_female, bpx_male))
```

    (-1.105435895556249, 0.26925004137768577, 952.0)


Another important aspect of two-sample mean testing is "heteroscedasticity", meaning that the variances within the two groups being compared may be different. While the goal of the test is to compare the means, the variances play an important role in calibrating the statistics (deciding how big the mean difference needs to be to be declared statisitically significant). In the NHANES data, we see that there are moderate differences between the amount of variation in BMI for females and for males, looking within 10-year age bands. In every age band, females having greater variation than males. 


```python
dx = da[["BMXBMI", "RIDAGEYR", "RIAGENDRx"]].dropna()
da["agegrp"] = pd.cut(da.RIDAGEYR, [18, 30, 40, 50, 60, 70, 80])
da.groupby(["agegrp", "RIAGENDRx"])["BMXBMI"].agg(np.std).unstack()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>RIAGENDRx</th>
      <th>Female</th>
      <th>Male</th>
    </tr>
    <tr>
      <th>agegrp</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(18, 30]</th>
      <td>7.745893</td>
      <td>6.649440</td>
    </tr>
    <tr>
      <th>(30, 40]</th>
      <td>8.315608</td>
      <td>6.622412</td>
    </tr>
    <tr>
      <th>(40, 50]</th>
      <td>8.076195</td>
      <td>6.407076</td>
    </tr>
    <tr>
      <th>(50, 60]</th>
      <td>7.575848</td>
      <td>5.914373</td>
    </tr>
    <tr>
      <th>(60, 70]</th>
      <td>7.604514</td>
      <td>5.933307</td>
    </tr>
    <tr>
      <th>(70, 80]</th>
      <td>6.284968</td>
      <td>4.974855</td>
    </tr>
  </tbody>
</table>
</div>



The standard error of the mean difference (e.g. mean female blood pressure minus mean mal blood pressure) can be estimated in at least two different ways. In the statsmodels library, these approaches are referred to as the "pooled" and the "unequal" approach to estimating the variance. If the variances are equal (i.e. there is no heteroscedasticity), then there should be little difference between the two approaches. Even in the presence of moderate heteroscedasticity, as we have here, we can see that the results for the two differences are quite similar. Below we have a loop that considers each 10-year age band and assesses the evidence for a difference in mean BMI for women and for men. The results printed in each row of output are the test-statistic and p-value. 


```python
for k, v in da.groupby("agegrp"):
    bmi_female = v.loc[v.RIAGENDRx=="Female", "BMXBMI"].dropna()
    bmi_female = sm.stats.DescrStatsW(bmi_female)
    bmi_male = v.loc[v.RIAGENDRx=="Male", "BMXBMI"].dropna()
    bmi_male = sm.stats.DescrStatsW(bmi_male)
    print(k)
    print("pooled: ", sm.stats.CompareMeans(bmi_female, bmi_male).ztest_ind(usevar='pooled'))
    print("unequal: ", sm.stats.CompareMeans(bmi_female, bmi_male).ztest_ind(usevar='unequal'))
    print()
```

    (18, 30]
    pooled:  (1.7026932933643388, 0.08862548061449649)
    unequal:  (1.7174610823927268, 0.08589495934713022)
    
    (30, 40]
    pooled:  (1.4378280405644916, 0.1504828511464818)
    unequal:  (1.4437869620833494, 0.14879891057892475)
    
    (40, 50]
    pooled:  (2.8933761158070186, 0.003811246059501354)
    unequal:  (2.9678691663536725, 0.0029987194174035366)
    
    (50, 60]
    pooled:  (3.362108779981367, 0.0007734964571391746)
    unequal:  (3.375494390173923, 0.0007368319423226574)
    
    (60, 70]
    pooled:  (3.6172401442432753, 0.000297761021031936)
    unequal:  (3.62848309454456, 0.0002850914147149227)
    
    (70, 80]
    pooled:  (2.926729252512258, 0.0034254694144858636)
    unequal:  (2.937779886769224, 0.003305716331519299)
    

