from CAL.PyCAL import *
import pandas as pd
import numpy as np
import datetime
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler
import talib


def st_remove(source_universe, st_date=None):
    """
    给定股票列表,去除其中在某日被标为ST的股票
    Args:
        source_universe (list of str): 需要进行筛选的股票列表
        st_date (datetime): 进行筛选的日期,默认为调用当天
    Returns:
        list: 去掉ST股票之后的股票列表

    Examples:
        >> universe = set_universe('A')
        >> universe_without_st = st_remove(universe)
    """
    st_date = st_date if st_date is not None else datetime.datetime.now().strftime('%Y%m%d')
    df_ST = DataAPI.SecSTGet(secID=source_universe, beginDate=st_date, endDate=st_date, field=['secID'])
    return [s for s in source_universe if s not in list(df_ST['secID'])]

def new_remove(ticker,tradeDate= None,day = 120):
    tradeDate = tradeDate if tradeDate is not None else datetime.datetime.now()
    period = '-' + str(day) + 'B'
    pastDate = cal.advanceDate(tradeDate,period)
    pastDate = pastDate.strftime("%Y-%m-%d")

    tickerDist={}
    tickerShort=[]
    for index in range(len(ticker)):
        OneTickerShort=ticker[index][0:6]
        tickerShort.append(OneTickerShort)
        tickerDist[OneTickerShort]=ticker[index]

    ipo_date = DataAPI.SecIDGet(partyID=u"",assetClass=u"",ticker=tickerShort,cnSpell=u"",field=u"ticker,listDate",pandas="1")
    remove_list = ipo_date[ipo_date['listDate'] > pastDate]['ticker'].tolist()
    remove_list=[values for keys,values in tickerDist.items() if keys in remove_list ]
    return [stk for stk in ticker if stk not in remove_list]

def AdaBoost(stocklist,date):
    preday=cal.advanceDate(date,Period('-3M'))
    yesterday=cal.advanceDate(date,'-1B')
    factors=['EMA10', 'EMA60', 'ROA', 'PE', 'LCAP', 'DHILO', 'DebtEquityRatio', 'OperatingProfitGrowRate', 'TotalAssetGrowRate', 'NPToTOR']

    # 建立训练集
    fac=DataAPI.MktStockFactorsOneDayGet(tradeDate=preday,secID=stocklist,field=['secID']+factors,pandas="1")
    price1=DataAPI.MktEqudAdjGet(secID=stocklist,tradeDate=preday,field=u"secID,closePrice",pandas="1")
    price2=DataAPI.MktEqudAdjGet(secID=stocklist,tradeDate=yesterday,field=u"secID,closePrice",pandas="1")
    price2['closePrice2']=price2['closePrice']
    del price2['closePrice']
    price=pd.merge(price1,price2)
    tmp1=[]
    tmp=(price['closePrice2']-price['closePrice'])/price['closePrice']*100
    for i in tmp:
        tmp1.append(int(i))
    price['zf']=tmp1
    del price['closePrice']
    del price['closePrice2']
    traindf=pd.merge(fac,price,how='inner')
    traindf.set_index(traindf.secID)
    del traindf['secID']
    traindf=traindf.dropna()
    traindf=traindf.sort(columns='zf')
    classification=list(traindf['zf'].apply(lambda x:1 if x>np.mean(list(traindf['zf'])) else 0))
    train=[]
    for x in range(0,len(traindf.iloc[:])):
        train.append(list(traindf.iloc[x][0:-1]))

    # 建立当期数据集
    test1=DataAPI.MktStockFactorsOneDayGet(tradeDate=yesterday,secID=stocklist,field=['secID']+factors,pandas="1")
    test1=test1.dropna()
    test=[]
    for x in range(0,len(test1.index)):
        test.append(list(test1.iloc[x][1:]))

    # 归一化
    nm=MinMaxScaler()
    train=nm.fit_transform(train)
    test=nm.transform(test)

    # 建立Adaboost-svm模型
    rf=AdaBoostClassifier(svm.SVC(kernel='rbf', C=15.0, gamma=10, probability=True), n_estimators=50, learning_rate=0.7, algorithm='SAMME.R', random_state=None)
    rf.fit(train, classification)
    predicted_results = [x[1] for index, x in enumerate(rf.predict_proba(test))]
    test1['predict']=predicted_results
    test1=test1.sort(columns='predict',ascending=False)
    stock=test1['secID'][:20].head(3)
    return stock


start = '2015-5-1'              # 回测起始时间
end = '2015-8-1'                # 回测结束时间
benchmark = 'HS300'               # 策略参考标准
universe = ['600908.XSHG','002807.XSHE','601009.XSHG','601128.XSHG','002839.XSHE','601988.XSHG','601169.XSHG','600015.XSHG','601169.XSHG','600926.XSHG','600848.XSHG','600036.XSHG','600383.XSHG','600919.XSHG','002142.XSHE','600016.XSHG','000001.XSHE','601398.XSHG','601818.XSHG','601288.XSHG',]   # 证券池，支持股票和基金
capital_base = 100000            # 起始资金
freq = 'd'                        # 策略类型，'d'表示日间策略使用日线回测，'m'表示日内策略使用分钟线回测
refresh_rate = 1                # 调仓频率，表示执行handle_data的时间间隔，若freq = 'd'时间间隔的单位为交易日，若freq = 'm'时间间隔为分钟

cal = Calendar('China.SSE')
commission = Commission(buycost=0.0003, sellcost=0.0013, unit='perValue')
slippage = Slippage(value=0.01, unit='perShare')

def initialize(account):                   # 初始化虚拟账户状态
    pass

def handle_data(account):                  # 每个交易日的买入卖出指令
    cal = Calendar('China.SSE')
    day1 = cal.advanceDate(account.current_date, '-30B', BizDayConvention.Preceding)
    df = DataAPI.MktIdxdGet(tradeDate=u"", indexID=u"", ticker="000300", beginDate=day1, endDate=account.current_date, field=["preCloseIndex"], pandas="1")
    close = np.array(df["preCloseIndex"])
    rsi = talib.RSI(close, 5)[-1]

    _universe1 = account.get_universe(exclude_halt=False)
    _universe2 = new_remove(_universe1)
    _universe = st_remove(_universe2)
    buylist=AdaBoost(_universe,account.current_date)
    print rsi

    if rsi > 70:
        for stock in buylist:
            order_pct_to(stock,0.8/len(buylist))

    if rsi < 30:
        for stock in account.avail_security_position.keys():
            if stock in _universe and stock not in buylist:
                order_to(stock,0)

        for stk in _universe:
            if(stk in account.security_cost):
                if ((account.referencePrice[stk]) < account.valid_seccost[stk]*0.8):     #当跌的超过了10%，全部卖掉
                    order_to(stk,0)
                if ((account.referencePrice[stk]) > account.valid_seccost[stk]*1.2):    #当涨的超过了10%，全部卖掉
                    order_to(stk,0)

