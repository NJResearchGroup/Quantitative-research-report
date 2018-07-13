
#本代码运行平台：优矿量化交易平台

import numpy as np
import pandas as pd
import scipy.linalg as sl
sl.expm2=sl.expm
import pybrain
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

start = '2014-01-01'                       # 回测起始时间
end = '2016-12-29'                         # 回测结束时间
benchmark = 'HS300'                        # 策略参考标准
universe =['600166.XSHG','000589.XSHE','000572.XSHE','601258.XSHG','600698.XSHG','600609.XSHG','000753.XSHE','000927.XSHE','002031.XSHE','601058.XSHG','002593.XSHE','600104.XSHG','601238.XSHG','601633.XSHG','000338.XSHE','000800.XSHE','000957.XSHE','002594.XSHE','000951.XSHE','601238.XSHG']             # 证券池，支持股票和基金
capital_base = 1000000                      # 起始资金
freq = 'd'                             # 策略类型，'d'表示日间策略使用日线回测，'m'表示日内策略使用分钟线回测
refresh_rate = 5   # 调仓频率，表示执行handle_data的时间间隔，若freq = 'd'时间间隔的单位为交易日，若freq = 'm'时间间隔为分钟



#################################################     构建并训练神经网络

start_train_date = '20151009'
end_train_date = '20151228'

net = buildNetwork(8,13,1)
dataset = SupervisedDataSet(8,1)


for sec in universe:
    data=DataAPI.MktEqudGet(tradeDate=u"",secID=sec,ticker=u"",beginDate=start_train_date,endDate=end_train_date,field=u"secID,tradeDate,preClosePrice,openPrice,highestPrice,lowestPrice,closePrice,turnoverRate,PE,PB,dealAmount",pandas="1").dropna()  
    
    length = len(data)
    for i in range(length-1):
        sample_input = data.irow(i)
        sample_target = data.irow(i+1).closePrice
        dataset.addSample([sample_input.openPrice,sample_input.highestPrice,sample_input.lowestPrice,sample_input.closePrice,sample_input.turnoverRate,sample_input.PE,sample_input.PB,sample_input.dealAmount],[sample_target])
    print 'Training samples have been created! ' + sec


print 'Start training!'
trainer = BackpropTrainer(net,dataset)
trainer.trainEpochs(10)
print 'Now the neural net has been fully trained!!!'

###############################################       神经网络训练完毕



def initialize(account):                   # 初始化虚拟账户状态
    pass

def handle_data(account):                  # 每个交易日的买入卖出指令
    
    market_val = DataAPI.MktEqudGet(tradeDate=account.current_date,field=u"secID,negMarketValue",pandas="1")    #获取所有股票的市值
    yesterday = DataAPI.TradeCalGet(exchangeCD=u"XSHG,XSHE",beginDate=account.current_date,endDate=account.current_date,field=u"prevTradeDate",pandas="1").prevTradeDate[0]
    factor = DataAPI.MktEqudGet(tradeDate=yesterday,field=u"secID,tradeDate,preClosePrice,openPrice,highestPrice,lowestPrice,closePrice,turnoverRate,PE,PB,dealAmount",pandas="1").dropna()
    # turnoverVol,turnoverValue,dealAmount
    
    sec_val_mkt = {'symbol':[], 'factor_value':[], 'market_value':[]}
    
    for stock in account.universe:
        if stock in list(factor.secID):
            # print 'stock in factor'
            sec_factor = factor[factor.secID==stock]
            predict_price = net.activate([sec_factor.openPrice,sec_factor.highestPrice,sec_factor.lowestPrice,sec_factor.closePrice,sec_factor.turnoverRate,sec_factor.PE,sec_factor.PB,sec_factor.dealAmount])
            yesterday_price = float(sec_factor.closePrice)
            if predict_price>0:
                buy=[]
                percentage=float(predict_price/yesterday_price)
                buy.append(percentage)
    
    
                                 
                # print 'price>0'
                sec_val_mkt['symbol'].append(stock)
                sec_val_mkt['factor_value'].append(float(predict_price/yesterday_price))
                
                sec_val_mkt['market_value'].append(float(market_val.negMarketValue[market_val.secID==stock]))
    
    sec_val_mkt = pd.DataFrame(sec_val_mkt).sort(columns='factor_value').reset_index()
    sec_val_mkt = sec_val_mkt[int(len(sec_val_mkt)*0.75):]           #排序并选择前25%
    print account.current_date
    print 'Here is the sec_val_mkt dataframe:'
    print sec_val_mkt
    
    buylist = list(sec_val_mkt.symbol)           #买入股票列表
    sum_market_val = sum(sec_val_mkt.market_value)
    position = np.array(sec_val_mkt.market_value)/sum_market_val*account.cash
    
    for stock in account.valid_secpos:
        if stock not in buylist:
            order_to(stock, 0)
    for stock in buylist:
        if stock not in account.valid_secpos:
            order(stock, position[buylist.index(stock)])

    return
