#导入相关的模块
from CAL.PyCAL import *
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import datetime # python基本模块
import numpy as np
from matplotlib import cm, pyplot as plt
import matplotlib.dates as dates
import pandas as pd
import seaborn as sns
sns.set_style('white')

beginDate='2014-01-01'
endDate='2016-01-01'

start='2015-04-30'
end='2015-08-01'
benchmark='HS300'
capital_base=100000
freq='d'
refresh_rate=1
universe = ['600741.XSHG','601799.XSHG','600104.XSHG','601311.XSHG','601238.XSHG','6601965.XSHG','601633.XSHG','601966.XSHG','603766.XSHG','600297.XSHG','600742.XSHG','600066.XSHG','000338.XSHE','002594.XSHE','000581.XSHE','000980.XSHE','000887.XSHE','002085.XSHE','000951.XSHE','000625.XSHE',]

global MonthEndDate
global HMM_Model  #定义一个全局的隐含马尔科夫对象
global HMMFlag  #定义一个全局变量还判断隐含马儿科夫对象是否建立
global HMMState

def GetDataForHMM(beginDate,endDate):
	data = DataAPI.MktIdxdGet(ticker='000001',beginDate=beginDate,endDate=endDate,field=['tradeDate','closeIndex','lowestIndex','highestIndex','turnoverVol'],pandas="1")#1指数日行情数据
	tradeDate = pd.to_datetime(data['tradeDate'][5:])#日期列表
	volume = data['turnoverVol'][5:]#2 成交量数据
	closeIndex = data['closeIndex'] # 3 收盘价数据
	deltaIndex = np.log(np.array(data['highestIndex'])) - np.log(np.array(data['lowestIndex'])) #3 当日对数高低价差
	deltaIndex = deltaIndex[5:]
	logReturn1 = np.array(np.diff(np.log(closeIndex))) #4 对数收益率
	logReturn1 = logReturn1[4:]
	logReturn5 = np.log(np.array(closeIndex[5:])) - np.log(np.array(closeIndex[:-5]))# 5日 对数收益差
	closeIndex = closeIndex[5:]
	X = np.column_stack([logReturn1,logReturn5,deltaIndex,volume]) # 将几个array合成一个2Darray
	return X
#利用上证指数更新马尔科夫模型
def UpdateHMM(beginDate,endDate):

	X =GetDataForHMM(beginDate,endDate)
	# Make an HMM instance and execute fit
	model = GaussianHMM(n_components=5, covariance_type="diag", n_iter=800).fit([X])
	return model
def HMMPredict(model,beginDate,endDate):
	X = GetDataForHMM(beginDate,endDate)
	hidden_states = model.predict(X)
	return hidden_states
def IsMonthEndDate(datetime1,MonthEndDate):
	tmpDateStr=datetime1.strftime('%Y-%m-%d')
	dd=MonthEndDate[MonthEndDate['calendarDate']==tmpDateStr]
	return dd.shape[0]
def UpdateHMMState(HMM_Model,beginDate,endDate):
	data = DataAPI.MktIdxdGet(ticker='000001',beginDate=beginDate,endDate=endDate,field=['tradeDate','closeIndex','lowestIndex','highestIndex','turnoverVol'],pandas="1")#1指数日行情数据

	tradeDate = pd.to_datetime(data['tradeDate'][5:])#日期列表
	volume = data['turnoverVol'][5:]#2 成交量数据
	closeIndex = data['closeIndex'] # 3 收盘价数据
	deltaIndex = np.log(np.array(data['highestIndex'])) - np.log(np.array(data['lowestIndex'])) #3 当日对数高低价差
	deltaIndex = deltaIndex[5:]
	logReturn1 = np.array(np.diff(np.log(closeIndex))) #4 对数收益率
	logReturn1 = logReturn1[4:]
	logReturn5 = np.log(np.array(closeIndex[5:])) - np.log(np.array(closeIndex[:-5]))# 5日 对数收益差

	closeIndex = closeIndex[5:]
	X = np.column_stack([logReturn1,logReturn5,deltaIndex,volume]) # 将几个array合成一个2Darray
	# Predict the optimal sequence of internal hidden state
	hidden_states = HMM_Model.predict(X)
	#整合数据
	res = pd.DataFrame({'tradeDate':tradeDate,'logReturn1':logReturn1,'logReturn5':logReturn5,'volume':volume,'state':hidden_states}).set_index('tradeDate')
	for i in range(HMM_Model.n_components):
		idx = (hidden_states==i)
		idx = np.append(0,idx[:-1])#获得状态结果后第二天进行买入操作
		#fast factor backtest
		df = res.logReturn1
		res['sig_ret%s'%i] = df.multiply(idx,axis=0)
		res['sig_cumret%s'%i] =np.exp(res['sig_ret%s'%i].cumsum())
	ss=res[['sig_cumret0','sig_cumret1','sig_cumret2','sig_cumret3','sig_cumret4']]
	ss=ss.tail(1).iloc[0]
	#log.info(ss)	ss.sort_values(ascending=False,inplace=False)
	tt=ss.sort_values(ascending=False,inplace=False)
	#log.info(tt)
	return tt


def initialize(account):                   # 初始化虚拟账户状态
	global MonthEndDate
	global HMMFlag
	HMMFlag=0
	MonthEndDate=DataAPI.TradeCalGet(exchangeCD=u"XSHG",beginDate=start,endDate=u"",field=u"calendarDate,isMonthEnd",pandas="1")
	MonthEndDate=MonthEndDate[MonthEndDate['isMonthEnd']==1]
    #log.info(MonthEndDate)



import pandas as pd
import numpy as np
from CAL.PyCAL import *
from datetime import datetime
import functools
from pandas import DataFrame



#构建日期函数
date=DataAPI.TradeCalGet(exchangeCD=u"XSHG,XSHE",beginDate=u"20150430",endDate=u"20150801",field=u"calendarDate,isOpen",pandas="1")
date=date[date['isOpen']==1]
date=date['calendarDate']
datelist=date.tolist()#结果['2007-08-01', '2007-08-02', '2007-08-03',
datelist=[v.replace('-', '') for v in datelist]#把日期中间的'-'去掉
#datelist
#先取因子
factor_name = ['OperatingRevenueGrowRate','ROE','NetProfitGrowRate','LCAP','PE','VOL20']#有在因子函数中有三个我没找到，其中股东户数变化数据是要购买的，如果有的话可以调用函数 DataAPI.JY.EquHdNumJYGet

cal = Calendar('China.SSE')
period = Period('-1B')


def handle_data(account):
    currentdate=account.current_date
    currentdate=Date.fromDateTime(account.current_date)
    cal = Calendar('China.SSE')
    lastdate=cal.advanceDate(currentdate,period)
    lastdate=lastdate.toDateTime()

    #factordata_body=[]

    factordata =DataAPI.MktStockFactorsOneDayGet(tradeDate=lastdate.strftime('%Y-%m-%d'),secID=account.universe,ticker=u"",field=['secID','tradeDate'] + factor_name,pandas="1")#取因子
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
    dealdata=factor_score.head(4)#取前20只
    tickerdata=dealdata.index.tolist()#取ticker
    secIDdata=DataAPI.SecTypeRelGet(typeID=u"",secID=u"",ticker=tickerdata,field=u"secID,ticker",pandas="1")#ticker转成对应的secID，因为交易买的指标是secID
    secIDdata=secIDdata.secID.unique()#不知道为什么上一步我转出来的secID每个都有6个重复值，我就用unique取每个的唯一值
    buylist=secIDdata.tolist()





    if IsMonthEndDate(account.current_date,MonthEndDate)==1: #如果月末（这个也可以通过CAL库的Date日期类实现），更新马尔可夫模型
		cal=Calendar('China.SSE')
		calDate=Date.fromDateTime(account.current_date)
		beginDatehmm=cal.advanceDate(calDate,'-40M',BizDayConvention.Preceding)
		HMM_Model=UpdateHMM(beginDatehmm.strftime('%Y%m%d'),calDate.strftime('%Y%m%d'))
		beginDateState= cal.advanceDate(calDate,'-10M',BizDayConvention.Preceding)
		HMMState=UpdateHMMState(HMM_Model,beginDateState,calDate)
		HMMFlag=1
	#根据马尔可夫模型进行状态预测
    global HMM_Model
    if HMMFlag==1:
        cal1=Calendar('China.SSE')
        calDate1=Date.fromDateTime(account.current_date)
        endDatePridi=cal1.advanceDate(calDate1,'-1d',BizDayConvention.Preceding)
        beginDatePridi=cal1.advanceDate(endDatePridi,'-20d',BizDayConvention.Preceding)
        hidden_states=HMMPredict(HMM_Model,beginDatePridi,endDatePridi)
	#根据状态进行仓位操作
	print(dealdata)
    global HMMFlag
    global HMMState
    if  HMMFlag==1:
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
                            if  ('sig_ret%s'%hidden_states[-1]==HMMState.index[0] ) & len(account.avail_security_position)==0:      #持仓为零的时候，并且满足0th hidden state
                                order_pct_to(stk,num)
                    if  ('sig_ret%s'%hidden_states[-1]==HMMState.index[3] or 'sig_ret%s'%hidden_states[-1]==HMMState.index[4]) & len(account.avail_security_position)>0:
                        order_to(stk,0)
                    if(stk in account.security_cost):
                        if ((account.referencePrice[stk]) < account.valid_seccost[stk]*0.9):     #当跌的超过了10%，全部卖掉
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
                            if  ('sig_ret%s'%hidden_states[-1]==HMMState.index[0] ) & len(account.avail_security_position)==0:      #持仓为零的时候，并且满足0th hidden state
                                order_pct_to(stk,num)
                    if  ('sig_ret%s'%hidden_states[-1]==HMMState.index[3] or 'sig_ret%s'%hidden_states[-1]==HMMState.index[4]) & len(account.avail_security_position)>0:
                        order_to(stk,0)
                    if(stk in account.security_cost):
                        if ((account.referencePrice[stk]) < account.valid_seccost[stk]*0.9):     #当跌的超过了10%，全部卖掉
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
                            if  ('sig_ret%s'%hidden_states[-1]==HMMState.index[0] ) & len(account.avail_security_position)==0:      #持仓为零的时候，并且满足0th hidden state
                                order_pct_to(stk,num)
                    if  ('sig_ret%s'%hidden_states[-1]==HMMState.index[3] or 'sig_ret%s'%hidden_states[-1]==HMMState.index[4]) & len(account.avail_security_position)>0:
                        order_to(stk,0)
                    if(stk in account.security_cost):
                        if ((account.referencePrice[stk]) < account.valid_seccost[stk]*0.9):     #当跌的超过了10%，全部卖掉
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
                            if  ('sig_ret%s'%hidden_states[-1]==HMMState.index[0] ) & len(account.avail_security_position)==0:      #持仓为零的时候，并且满足0th hidden state
                                order_pct_to(stk,num)
                    if  ('sig_ret%s'%hidden_states[-1]==HMMState.index[3] or 'sig_ret%s'%hidden_states[-1]==HMMState.index[4]) & len(account.avail_security_position)>0:
                        order_to(stk,0)
                    if(stk in account.security_cost):
                        if ((account.referencePrice[stk]) < account.valid_seccost[stk]*0.9):     #当跌的超过了10%，全部卖掉
                            order_to(stk,0)
                        if ((account.referencePrice[stk]) > account.valid_seccost[stk]*1.2):    #当涨的超过了20%，全部卖掉
                            order_to(stk,0)



