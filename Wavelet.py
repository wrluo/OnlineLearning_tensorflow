import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pywt   # python 小波变换的包

# 取数据
# beginDate = '20130101'
# endDate = '20151231'
# data = DataAPI.MktIdxdGet(ticker='000300',beginDate=beginDate,endDate=endDate,field=['tradeDate','closeIndex'],pandas="1")
# index_list = np.array(data['closeIndex'])

# 函数打包
def wt(index_list,wavefunc,lv,m,n):
#def wt(index_list,data,wavefunc,lv,m,n):   # 打包为函数，方便调节参数。  lv为分解层数；data为最后保存的dataframe便于作图；index_list为待处理序列；wavefunc为选取的小波函数；m,n则选择了进行阈值处理的小波系数层数
    # 分解
    coeff = pywt.wavedec(index_list,wavefunc,mode='sym',level=lv)   # 按 level 层分解，使用pywt包进行计算， cAn是尺度系数 cDn为小波系数

    sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0 # sgn函数

    # 去噪过程
    for i in range(m,n+1):   # 选取小波系数层数为 m~n层，尺度系数不需要处理
        cD = coeff[i]
        for j in range(len(cD)):
            Tr = np.sqrt(2*np.log(len(cD)))  # 计算阈值
            if cD[j] >= Tr:
                coeff[i][j] = sgn(cD[j]) - Tr  # 向零收缩
            else:
                coeff[i][j] = 0   # 低于阈值置零

    # 重构
    denoised_index = pywt.waverec(coeff,wavefunc)

    return denoised_index

    # 在原dataframe中添加处理后的列便于画图
#     data['denoised_index']=pd.Series('x',index=data.index)
#     for i in range(len(data)):
#         data['denoised_index'][i] = denoised_index[i] 
# ​
#     # 画图
#     data = data.set_index(data['tradeDate'])
#     data.plot(figsize=(20,20),subplots=(2,1))
#     data.plot(figsize=(20,10))

# 调用函数wt
# wt(index_list,data,'db4',4,1,4)  # 小波函数为db4, 分解层数为4， 选出小波层数为1-4层