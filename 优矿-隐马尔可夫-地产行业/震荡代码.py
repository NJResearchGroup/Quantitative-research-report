from hmmlearn.hmm import GaussianHMM
from sklearn.naive_bayes import GaussianNB
import datetime
from CAL.PyCAL import *
from datetime import datetime
import functools
from pandas import DataFrame


start='20151215'
end='20160215'
benchmark='HS300'
capital_base = 100000                      # 起始资金
freq='d'
refresh_rate=1
universe = ['000002.XSHE','000005.XSHE','000006.XSHE','000009.XSHE','000011.XSHE','000014.XSHE','000024.XSHE','000029.XSHE','000031.XSHE','000036.XSHE','000038.XSHE','000040.XSHE','000042.XSHE','000043.XSHE','000046.XSHE','000069.XSHE','000090.XSHE','601155.XSHG','600048.XSHG','600383.XSHG',]           # 证券池，支持股票和基金

def initialize(account):
    pass


# 获取数据
beginDate = '20110401'
endDate = '20140401'
data = DataAPI.MktIdxdGet(ticker='000001',beginDate=beginDate,endDate=endDate,field=['tradeDate','closeIndex','lowestIndex','highestIndex','turnoverVol'],pandas="1")#1指数日行情数据
data1 = DataAPI.FstTotalGet(exchangeCD=u"XSHE",beginDate=beginDate,endDate=endDate,field=['tradeVal'],pandas="1")#1获取深圳交易所的融资融券数据
data2 = DataAPI.FstTotalGet(exchangeCD=u"XSHG",beginDate=beginDate,endDate=endDate,field=['tradeVal'],pandas="1")
tradeVal = data1 + data2 #1融资融券数据总和
tradeDate = pd.to_datetime(data['tradeDate'][5:])#日期列表
volume = data['turnoverVol'][5:]#2 成交量数据
closeIndex = data['closeIndex'] # 3 收盘价数据
deltaIndex = np.log(np.array(data['highestIndex'])) - np.log(np.array(data['lowestIndex'])) #3 当日对数高低价差
deltaIndex = deltaIndex[5:]
logReturn1 = np.array(np.diff(np.log(closeIndex))) #4 对数收益率
logReturn1 = logReturn1[4:]
logReturn5 = np.log(np.array(closeIndex[5:])) - np.log(np.array(closeIndex[:-5]))# 5日 对数收益差
logReturnFst = np.array(np.diff(np.log(tradeVal['tradeVal'])))[4:]
closeIndex = closeIndex[5:]
X = np.column_stack([logReturn1,logReturn5,deltaIndex,volume,logReturnFst]) # 将几个array合成一个2Darray
# Make an HMM instance and execute fit
model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000).fit([X])
# Predict the optimal sequence of internal hidden state
hidden_states = model.predict(X)
print hidden_states
res = pd.DataFrame({'tradeDate':tradeDate,'logReturn1':logReturn1,'logReturn5':logReturn5,'volume':volume,'hidden_states':hidden_states}).set_index('tradeDate')
for i in range(model.n_components):
    idx = (hidden_states==i)
    idx = np.append(0,idx[:-1])#获得状态结果后第二天进行买入操作
#fast factor backtest
    df = res.logReturn1
    res['sig_ret%s'%i] = df.multiply(idx,axis=0)
    res['sig_cumret%s'%i] =np.exp(res['sig_ret%s'%i].cumsum())
ss=res[['sig_cumret0','sig_cumret1','sig_cumret2']]
ss=ss.tail(1).iloc[0]
#log.info(ss)	ss.sort_values(ascending=False,inplace=False)
tt=ss.sort_values(ascending=False,inplace=False)

testbegin = '20170401'
testend = '20171001'

tdata = DataAPI.MktIdxdGet(ticker='000001',beginDate=testbegin,endDate=testend,field=['tradeDate','closeIndex','lowestIndex','highestIndex','turnoverVol'],pandas="1")#1指数日行情数据
tdata1 = DataAPI.FstTotalGet(exchangeCD=u"XSHE",beginDate=testbegin,endDate=testend,field=['tradeVal'],pandas="1")#1获取深圳交易所的融资融券数据
tdata2 = DataAPI.FstTotalGet(exchangeCD=u"XSHG",beginDate=testbegin,endDate=testend,field=['tradeVal'],pandas="1")
ttradeVal = tdata1 + tdata2 #1融资融券数据总和
ttradeDate = pd.to_datetime(tdata['tradeDate'][5:])#日期列表
tvolume = tdata['turnoverVol'][5:]#2 成交量数据
tcloseIndex = tdata['closeIndex'] # 3 收盘价数据
tdeltaIndex = np.log(np.array(tdata['highestIndex'])) - np.log(np.array(tdata['lowestIndex'])) #3 当日对数高低价差
tdeltaIndex = tdeltaIndex[5:]
tlogReturn1 = np.array(np.diff(np.log(tcloseIndex))) #4 对数收益率
tlogReturn1 = tlogReturn1[4:]
tlogReturn5 = np.log(np.array(tcloseIndex[5:])) - np.log(np.array(tcloseIndex[:-5]))# 5日 对数收益差
tlogReturnFst = np.array(np.diff(np.log(ttradeVal['tradeVal'])))[4:]
tcloseIndex = tcloseIndex[5:]
Y = np.column_stack([tlogReturn1,tlogReturn5,tdeltaIndex,tvolume,tlogReturnFst]) # 将几个array合成一个2Darray
model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000).fit([Y])
# Predict the optimal sequence of internal hidden state
thidden_states = model.predict(Y)

# NB训练集
y_train = hidden_states
x_train = X
y_test = thidden_states
x_test = Y
# NB测试集
clf = GaussianNB()
clf.fit(x_train, y_train)
preY = clf.predict(x_test)
#print preY
accuracy = clf.score(x_test, y_test)
#print accuracy

#构建日期函数
date=DataAPI.TradeCalGet(exchangeCD=u"XSHG,XSHE",beginDate=u"20151215",endDate=u"20160215",field=u"calendarDate,isOpen",pandas="1")
date=date[date['isOpen']==1]
date=date['calendarDate']
datelist=date.tolist()#结果['2007-08-01', '2007-08-02', '2007-08-03',
datelist=[v.replace('-', '') for v in datelist]#把日期中间的'-'去掉
#datelist
#先取因子
factor_name = ['OperatingRevenueGrowRate','ROE','NetProfitGrowRate','LCAP','PE','VOL20']#有在因子函数中有三个我没找到，其中股东户数变化数据是要购买的，如果有的话可以调用函数 DataAPI.JY.EquHdNumJYGet

cal = Calendar('China.SSE')
period = Period('-1B')


def handle_data(account): # 每个交易日的买入卖出指令
    cal=Calendar('China.SSE')
    calDate=Date.fromDateTime(account.current_date)
    #print calDate
    endDatePridi=cal.advanceDate(calDate,'-1d',BizDayConvention.Preceding)
    beginDatePridi=cal.advanceDate(endDatePridi,'-20d',BizDayConvention.Preceding)
    testbegin = beginDatePridi
    testend = endDatePridi
    tdata = DataAPI.MktIdxdGet(ticker='000001',beginDate=testbegin,endDate=testend,field=['tradeDate','closeIndex','lowestIndex','highestIndex','turnoverVol'],pandas="1")#1指数日行情数据
    tdata1 = DataAPI.FstTotalGet(exchangeCD=u"XSHE",beginDate=testbegin,endDate=testend,field=['tradeVal'],pandas="1")#1获取深圳交易所的融资融券数据
    tdata2 = DataAPI.FstTotalGet(exchangeCD=u"XSHG",beginDate=testbegin,endDate=testend,field=['tradeVal'],pandas="1")
    ttradeVal = tdata1 + tdata2 #1融资融券数据总和
    ttradeDate = pd.to_datetime(tdata['tradeDate'][5:])#日期列表
    tvolume = tdata['turnoverVol'][5:]#2 成交量数据
    tcloseIndex = tdata['closeIndex'] # 3 收盘价数据
    tdeltaIndex = np.log(np.array(tdata['highestIndex'])) - np.log(np.array(tdata['lowestIndex'])) #3 当日对数高低价差
    tdeltaIndex = tdeltaIndex[5:]
    tlogReturn1 = np.array(np.diff(np.log(tcloseIndex))) #4 对数收益率
    tlogReturn1 = tlogReturn1[4:]
    tlogReturn5 = np.log(np.array(tcloseIndex[5:])) - np.log(np.array(tcloseIndex[:-5]))# 5日 对数收益差
    tlogReturnFst = np.array(np.diff(np.log(ttradeVal['tradeVal'])))[4:]
    tcloseIndex = tcloseIndex[5:]
    Y = np.column_stack([tlogReturn1,tlogReturn5,tdeltaIndex,tvolume,tlogReturnFst]) # 将几个array合成一个2Darray
    preY = clf.predict(Y)

    currentdate=account.current_date
    currentdate=Date.fromDateTime(account.current_date)
    cal = Calendar('China.SSE')
    lastdate=cal.advanceDate(currentdate,period)
    lastdate=lastdate.toDateTime()

    #factordata_body=[]

    factordata =DataAPI.MktStockFactorsOneDayGet(tradeDate=lastdate.strftime('%Y%m%d'),secID=account.universe,ticker=u"",field=['secID','tradeDate'] + factor_name,pandas="1")#取因子
    factordata=factordata.dropna()
        #factordata_body.append(factordata)
        #new_fctrdata=pd.concat(factordata_body)#将上边遍历出来的每个DataFrame拼接起来
        #对营业利润同比增长率进行标准化_正向因子
    OperatRevenueGR=factordata.iloc[:,:3]
    OperatRevenueGR['ticker'] = OperatRevenueGR['secID'].apply(lambda x: x[0:6])
    OperatRevenueGR.set_index('ticker',inplace=True)
    OpratRvnuGR=OperatRevenueGR['OperatingRevenueGrowRate'].to_dict()
    OpratRvnuGR=standardize(OpratRvnuGR)#standardize()函数要求传入的类型是dict
        #对ROE标准化_正向因子
    ROE_data=factordata.iloc[:,[0,1,3]]
    ROE_data['ticker'] = ROE_data['secID'].apply(lambda x: x[0:6])
    ROE_data.set_index('ticker',inplace=True)
    ROE=ROE_data['ROE'].to_dict()
    ROE=standardize(ROE)
        #对净利润增长率标准化_正向因子
    NProfitGR=factordata.iloc[:,[0,1,4]]
    NProfitGR['ticker'] = NProfitGR['secID'].apply(lambda x: x[0:6])
    NProfitGR.set_index('ticker',inplace=True)
    NetProfitGrowRate=NProfitGR['NetProfitGrowRate'].to_dict()
    NetProfitGrowRate=standardize(NetProfitGrowRate)
        #对总市值（这里用的是对数总市值）标准化_负向因子
    mktvalue=factordata.iloc[:,[0,1,5]]
    mktvalue['ticker'] = mktvalue['secID'].apply(lambda x: x[0:6])
    mktvalue.set_index('ticker',inplace=True)
    LCAP=mktvalue['LCAP'].to_dict()
    LCAP=standardize(LCAP)
        #对PE标准化_负向因子
    PE_raw=factordata.iloc[:,[0,1,6]]
    PE_raw['ticker'] = PE_raw['secID'].apply(lambda x: x[0:6])
    PE_raw.set_index('ticker',inplace=True)
    PE=PE_raw['PE'].to_dict()
    PE=standardize(PE)
        #对换手率VOL20标准化_负向因子
    VOL20_raw=factordata.iloc[:,[0,1,7]]
    VOL20_raw['ticker'] = VOL20_raw['secID'].apply(lambda x: x[0:6])
    VOL20_raw.set_index('ticker',inplace=True)
    VOL20=VOL20_raw['VOL20'].to_dict()
    VOL20=standardize(VOL20)

        #计算综合得分
    pop={'OpratRvnuGR':OpratRvnuGR,
             'ROE':ROE,
            'NetProfitGrowRate':NetProfitGrowRate,
            'LCAP':LCAP,
            'PE':PE,
            'VOL20':VOL20}
    factor_total=DataFrame(pop,columns=['OpratRvnuGR','ROE','NetProfitGrowRate','LCAP','PE','VOL20'])#把上边的每个因子从字典类型换成DataFrame
    factor_total['positive_sum']=factor_total['OpratRvnuGR']+factor_total['ROE']+factor_total['NetProfitGrowRate']
    factor_total['negative_sum']=factor_total['LCAP']+factor_total['PE']+factor_total['VOL20']
    factor_total['score']=factor_total['positive_sum']-factor_total['negative_sum']
    factor_score=factor_total['score']
    factor_score=pd.DataFrame(factor_score)#把Series 转换成DataFrame
    factor_score=factor_score.sort_index(by='score',ascending=False)#按得分降序排序
    dealdata=factor_score.head(4)       #取前4只
    tickerdata=dealdata.index.tolist()#取ticker
    secIDdata=DataAPI.SecTypeRelGet(typeID=u"",secID=u"",ticker=tickerdata,field=u"secID,ticker",pandas="1")#ticker转成对应的secID，因为交易买的指标是secID
    secIDdata=secIDdata.secID.unique()
    #buylist=secIDdata.tolist()
    total_score=sum(dealdata['score'])
    print(dealdata)

    #择时

    if dealdata[0:sum(dealdata[0:4]['score']>0)]['score'].all>=0:
        s=sum(dealdata[0:sum(dealdata[0:4]['score']>0)]['score'])          #计算前四只得分大于0的股票得分总和

        num1=dealdata[0:1]['score']/s*0.8                                  #第一支股票的持仓比例，*0.8是对仓位的控制
        tickerdata=num1.index.tolist()#取ticker
        secIDdata=DataAPI.SecTypeRelGet(typeID=u"",secID=u"",ticker=tickerdata,field=u"secID,ticker",pandas="1")#ticker转成对应的secID，因为交易买的指标是secID
        secIDdata=secIDdata.secID.unique()
        universe1=list(secIDdata)
        for stk in universe1:
            for num in num1:
                if(account.referencePortfolioValue*0.2 < account.cash):         #持仓小于80%的时候，才会有买卖
                    if  (num >= 0):
                        if  (('sig_cumret%s'%preY[-1]==tt.index[0]) & len(account.avail_security_position)==0):      #持仓为零的时候，并且满足0th hidden state
                            order_pct_to(stk,num)
                if  (('sig_cumret%s'%preY[-1]==tt.index[2]) & len(account.avail_security_position)>0 ):
                    order_to(stk,0)
                if(stk in account.security_cost):
                    if ((account.referencePrice[stk]) < account.valid_seccost[stk]*0.8):     #当跌的超过了20%，全部卖掉
                        order_to(stk,0)
                    if ((account.referencePrice[stk]) > account.valid_seccost[stk]*1.2):    #当涨的超过了20%，全部卖掉
                        order_to(stk,0)


        num2=dealdata[1:2]['score']/s*0.8
        tickerdata=num2.index.tolist()#取ticker
        secIDdata=DataAPI.SecTypeRelGet(typeID=u"",secID=u"",ticker=tickerdata,field=u"secID,ticker",pandas="1")#ticker转成对应的secID，因为交易买的指标是secID
        secIDdata=secIDdata.secID.unique()
        universe2=list(secIDdata)
        for stk in universe2:
            for num in num2:
                if(account.referencePortfolioValue*0.2 < account.cash):         #持仓小于80%的时候，才会有买卖
                    if (num >= 0):
                        if  (('sig_cumret%s'%preY[-1]==tt.index[0]) & len(account.avail_security_position)==0):      #持仓为零的时候，并且满足0th hidden state
                            order_pct_to(stk,num)
                if  (('sig_cumret%s'%preY[-1]==tt.index[2]) & len(account.avail_security_position)>0 ):
                    order_to(stk,0)
                if(stk in account.security_cost):
                    if ((account.referencePrice[stk]) < account.valid_seccost[stk]*0.8):     #当跌的超过了20%，全部卖掉
                        order_to(stk,0)
                    if ((account.referencePrice[stk]) > account.valid_seccost[stk]*1.2):    #当涨的超过了20%，全部卖掉
                        order_to(stk,0)


        num3=dealdata[2:3]['score']/s*0.8
        tickerdata=num3.index.tolist()#取ticker
        secIDdata=DataAPI.SecTypeRelGet(typeID=u"",secID=u"",ticker=tickerdata,field=u"secID,ticker",pandas="1")#ticker转成对应的secID，因为交易买的指标是secID
        secIDdata=secIDdata.secID.unique()
        universe3=list(secIDdata)
        for stk in universe3:
            for num in num3:
                if(account.referencePortfolioValue*0.2 < account.cash):         #持仓小于80%的时候，才会有买卖
                    if(num >=0):
                        if  (('sig_cumret%s'%preY[-1]==tt.index[0]) & len(account.avail_security_position)==0):      #持仓为零的时候，并且满足0th hidden state
                            order_pct_to(stk,num)
                if  (('sig_cumret%s'%preY[-1]==tt.index[2]) & len(account.avail_security_position)>0 ):
                    order_to(stk,0)
                if(stk in account.security_cost):
                    if ((account.referencePrice[stk]) < account.valid_seccost[stk]*0.8):     #当跌的超过了20%，全部卖掉
                        order_to(stk,0)
                    if ((account.referencePrice[stk]) > account.valid_seccost[stk]*1.2):    #当涨的超过了20%，全部卖掉
                        order_to(stk,0)


        num4=dealdata[3:4]['score']/s*0.8
        tickerdata=num4.index.tolist()#取ticker
        secIDdata=DataAPI.SecTypeRelGet(typeID=u"",secID=u"",ticker=tickerdata,field=u"secID,ticker",pandas="1")#ticker转成对应的secID，因为交易买的指标是secID
        secIDdata=secIDdata.secID.unique()
        universe4=list(secIDdata)
        for stk in universe4:
            for num in num4:
                if(account.referencePortfolioValue*0.2 < account.cash):         #持仓小于80%的时候，才会有买卖
                    if  (num >= 0):
                        if  (('sig_cumret%s'%preY[-1]==tt.index[0]) & len(account.avail_security_position)==0):      #持仓为零的时候，并且满足0th hidden state
                            order_pct_to(stk,num)
                if  (('sig_cumret%s'%preY[-1]==tt.index[2]) & len(account.avail_security_position)>0 ):
                    order_to(stk,0)
                if(stk in account.security_cost):
                    if ((account.referencePrice[stk]) < account.valid_seccost[stk]*0.8):     #当跌的超过了20%，全部卖掉
                        order_to(stk,0)
                    if ((account.referencePrice[stk]) > account.valid_seccost[stk]*1.2):    #当涨的超过了20%，全部卖掉
                        order_to(stk,0)


