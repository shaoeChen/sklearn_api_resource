# Scikit-Learn sklearn.preprocessing.StandardScaler
###### tags: `scikit-learn` `sklearn` `python` `machine learning` `preprocessing`
>[name=Marty.chen ] [time=Fri, Jan 3, 2020 5:32 PM]
>以下範例資料皆來自官方文件 
>[HackMD hyperlink](https://hackmd.io/@shaoeChen/HyEVuY21L)

:::danger
官方文件：
* [API](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler)
* [範例](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py)
* [Z-Score](http://terms.naer.edu.tw/detail/1315812/?index=1)
:::

## 說明
實作機器學習的時候，特徵的前置預處理是必經過程，因為每一個特徵有它自己的值域與單位，有大有小，沒有在相同的空間範圍內，這會造成擁有較大值域的特徵對模型的影響過大。舉例來說，一棟屋子，三房兩廳二衛浴，屋齡30年，每坪20,000元，郵政編碼200。每坪20,000元，這個值域相對其它特徵來說都過大了，這對模型會有不良的影響。

因此每一個特徵我們都需要做前置預處理，將它們縮放至相同的大小。也因此，在實作過程中我們會取整個訓練資料集的特徵來計算各別特徵的值域空間，然後在測試的時候將測試資料經過相同的空間縮放至相同的值域大小。

簡單說，你的100元不是我的100元，可能你談的是100日元，而我是100新台幣，但是當我們經過相同空間縮放為美金的時候，所談就是一樣的價錢了。

### 公式
StandardScaler是sklearn中實作Z-score的一種方法，其公式思維很簡單：$z = (x - u) / s$，其中：
* $x$：input
* $u$：均值
* $s$：方差

其中均值可透過參數設置為0，而方差可透過參數設置為1。透過公式得到的結果可以簡單的判讀：
* $z$為正： 代表input大於均值
* $z$為負： 代表input小於均值

>取自[華人百科](https://www.itsfun.com.tw/%E6%A8%99%E6%BA%96%E5%88%86%E6%95%B8/wiki-9331053-8934233)範例
>>例如:某中學高(1)班期末考試，已知語文期末考試的全班平均分為73分，標準差為7分，甲得了78分;數學期末考試的全班平均分為80分，標準差為6.5分，甲得了83分。甲哪一門考試成績比較好?
因為兩科期末考試的標準差不同，因此不能用原始分數直接比較。需要將原始分數轉換成標準分數，然後進行比較。
Z(語文)=(78-73)/7=0.71 Z(數學)=(83-80)/6.5=0.46 甲的語文成績在其整體分布中位于平均分之上0.71個標準差的地位，他的數學成績在其整體分布中位于平均分之上0.46個標準差的地位。由此可見，甲的語文期末考試成績優于數學期末考試成績。
## 應用
```python
 from sklearn.preprocessing import StandardScaler
```
### class
```python
sklearn.preprocessing.StandardScaler(copy=True, 
                                     with_mean=True, 
                                     with_std=True
```
#### parameters
* copy:
    * type: boolean
    * default: True
* with_mean:
    * type: boolean
    * default: True
    * note: 若False則為0
* with_std:
    * type: boolean
    * default: True
    * note: 若False則為1 
#### attributes
* scale_: ndarray or None, shape (n_features,)
    * note: 每一個特徵的相對縮放比例，以`np.sqrt(var_)`計算。當`with_std=False`則為`None`
* mean_: ndarray or None, shape (n_features,)
    * note: 訓練集中每一個特徵的均值。當`with_mean=False`則為`None`
* var_: ndarray or None, shape (n_features,)
    * note: 訓練集中每一個特徵的方差。用來計算`scale_`。當`with_std=False`則為`None`
* n_samples_seen_: int or array, shape (n_features,)
    * note: 估計器為每個特徵處理的樣本數。如果沒有缺失值，其值為整數，如有缺失則為陣列。如果重新執行`fit`則數值會重置，但若執行`partial_fit`會累加計算
#### methods
* fit
    * note: 計算均值、方差做為續縮放使用
* fit_transform
    * note: 擬合並轉換
* get_params
    * note: 取得估計器的參數
* inverse_transform
    * not: 將縮放過的資料轉回原始資料大小
* partial_fit
    * note: 在線計算均值、方差，以便後續縮放使用
* set_params
    * note: 設置估計器的參數
* transform
    * note: 執行標準化(要先fit)
## 範例
請參考[github]()