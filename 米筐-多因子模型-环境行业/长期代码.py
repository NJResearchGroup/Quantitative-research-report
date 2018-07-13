from datetime import datetime
from CAL.PyCAL import *
import talib
import pandas as pd
import numpy as np
from CAL.PyCAL import *
import functools
from pandas import DataFrame

start = '2014-1-1'
end = '2017-1-1'


benchmark = 'HS300'                # 策略参考标准
capital_base = 100000             # 起始资金
freq = 'd'                      # 策略类型，'d'表示日间策略使用日线回测，'m'表示日内策略使用分钟线回测
refresh_rate = 1                  # 调仓频率，表示执行handle_data的时间间隔，若freq = 'd'时间间隔的单位为交易日，若freq = 'm'时间间隔为分钟
universe = ['600008.XSHG','600168.XSHG','600874.XSHG','601158.XSHG','601199.XSHG','600724.XSHG','600656.XSHG','600475.XSHG','600797.XSHG','000544.XSHE','000598.XSHE','000685.XSHE','000712.XSHE','000753.XSHE','001896.XSHE','002015.XSHE','002479.XSHE','002499.XSHE','000581.XSHE','000826.XSHE','000915.XSHE',]


#构建日期函数
date=DataAPI.TradeCalGet(exchangeCD=u"XSHG,XSHE",beginDate=u"20140101",endDate=u"20170101",field=u"calendarDate,isOpen",pandas="1")
date=date[date['isOpen']==1]
date=date['calendarDate']
datelist=date.tolist()#结果['2007-08-01', '2007-08-02', '2007-08-03',
datelist=[v.replace('-', '') for v in datelist]#把日期中间的'-'去掉
#datelist
#先取因子
factor_name = ['OperatingRevenueGrowRate','ROE','NetProfitGrowRate','LCAP','PE','VOL20']#有在因子函数中有三个我没找到，其中股东户数变化数据是要购买的，如果有的话可以调用函数 DataAPI.JY.EquHdNumJYGet

cal = Calendar('China.SSE')
period = Period('-1B')

def initialize(account):
    pass

def handle_data(account):
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
    dealdata=factor_score.head(4)#取前20只
    tickerdata=dealdata.index.tolist()#取ticker
    secIDdata=DataAPI.SecTypeRelGet(typeID=u"",secID=u"",ticker=tickerdata,field=u"secID,ticker",pandas="1")#ticker转成对应的secID，因为交易买的指标是secID
    secIDdata=secIDdata.secID.unique()#不知道为什么上一步我转出来的secID每个都有6个重复值，我就用unique取每个的唯一值
    buylist=secIDdata.tolist()




    day1 = cal.advanceDate(account.current_date, '-30B', BizDayConvention.Preceding)
    df = DataAPI.MktIdxdGet(tradeDate=u"", indexID=u"", ticker="000300", beginDate=day1, endDate=account.current_date, field=["preCloseIndex"], pandas="1")
    close = np.array(df["preCloseIndex"])

    rsi = talib.RSI(close, 5)[-1]
    s=sum(dealdata[0:sum(dealdata[0:4]['score']>0)]['score'])          #计算前四只得分大于0的股票得分总和
    print rsi


    # RSI择时
    if rsi > 70:

        num1=dealdata[0:1]['score']/s*0.8                                  #第一支股票的持仓比例，*0.8是对仓位的控制
        tickerdata=num1.index.tolist()#取ticker
        secIDdata=DataAPI.SecTypeRelGet(typeID=u"",secID=u"",ticker=tickerdata,field=u"secID,ticker",pandas="1")#ticker转成对应的secID，因为交易买的指标是secID
        secIDdata=secIDdata.secID.unique()
        universe1=list(secIDdata)
        for stk in universe1:
            for num in num1:
                if  (num >= 0):
                    order_pct_to(stk,num)


        num2=dealdata[1:2]['score']/s*0.8
        tickerdata=num2.index.tolist()#取ticker
        secIDdata=DataAPI.SecTypeRelGet(typeID=u"",secID=u"",ticker=tickerdata,field=u"secID,ticker",pandas="1")#ticker转成对应的secID，因为交易买的指标是secID
        secIDdata=secIDdata.secID.unique()
        universe2=list(secIDdata)
        for stk in universe2:
            for num in num2:
                if(account.referencePortfolioValue*0.2 < account.cash):         #持仓小于80%的时候，才会有买卖
                    if (num >= 0):
                            order_pct_to(stk,num)


        num3=dealdata[2:3]['score']/s*0.8
        tickerdata=num3.index.tolist()#取ticker
        secIDdata=DataAPI.SecTypeRelGet(typeID=u"",secID=u"",ticker=tickerdata,field=u"secID,ticker",pandas="1")#ticker转成对应的secID，因为交易买的指标是secID
        secIDdata=secIDdata.secID.unique()
        universe3=list(secIDdata)
        for stk in universe3:
            for num in num3:
                if(num >=0):
                    order_pct_to(stk,num)


        num4=dealdata[3:4]['score']/s*0.8
        tickerdata=num4.index.tolist()#取ticker
        secIDdata=DataAPI.SecTypeRelGet(typeID=u"",secID=u"",ticker=tickerdata,field=u"secID,ticker",pandas="1")#ticker转成对应的secID，因为交易买的指标是secID
        secIDdata=secIDdata.secID.unique()
        universe4=list(secIDdata)
        for stk in universe4:
            for num in num4:
                if  (num >= 0):
                    order_pct_to(stk,num)

    elif rsi < 30:
        # 卖出不在买入列表内的
        for stk in account.security_position:
            if stk not in buylist:
                order_to(stk, 0)
            if(stk in account.security_cost):
                if ((account.referencePrice[stk]) < account.valid_seccost[stk]*0.8):     #当跌的超过了20%，全部卖掉
                    order_to(stk,0)
                if ((account.referencePrice[stk]) > account.valid_seccost[stk]*1.2):    #当涨的超过了20%，全部卖掉
                    order_to(stk,0)



        return
    