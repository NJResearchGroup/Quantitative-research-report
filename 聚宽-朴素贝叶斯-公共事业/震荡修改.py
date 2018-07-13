import talib
import pandas as pd
from jqdata import *
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import datetime
import matplotlib.pyplot as plt

def initialize(context):
    g.stocks = ['600900.XSHG','300203.XSHE','600674.XSHG','600008.XSHG','000883.XSHE','600642.XSHG','600027.XSHG','600011.XSHG','000027.XSHE','600886.XSHG','000967.XSHE','000035.XSHE','600452.XSHG','600995.XSHG','600681.XSHG','600917.XSHG','600461.XSHG','300335.XSHE','601139.XSHG','300072.XSHE','300422.XSHE','002039.XSHE','002672.XSHE','600323.XSHG','000826.XSHE','000820.XSHE','300332.XSHE']
    #g.security = '000002.XSHE'
    # 设定沪深300作为基准
    set_benchmark('000300.XSHG')
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)


def Strategy(test_stock,start_date,end_date):
    # 开始预测的日期，也需要是个交易日
    test_start_date = datetime.date(2015, 2, 26)

    # 得到所有的交易日列表
    trading_days = get_all_trade_days()
    start_date_index = list(trading_days).index(start_date)
    end_date_index = list(trading_days).index(end_date)
    test_start_index = list(trading_days).index(test_start_date)

    # 保存每一天预测的结果，如果某天预测对了，保存1，如果某天预测错了，保存-1
    result_list = []

    # 每一天都对应于一个index_end
    for index_end in range(test_start_index, end_date_index):
        # x_all中保存所有的特征信息
        # y_all中保存所有的标签信息（要预测的对象）
        x_all = []
        y_all = []
        # 这个时间段产生所有的训练数据
        for index in range(start_date_index, index_end):
            #计算特征的代码
            start_day = trading_days[index - 35]
            end_day = trading_days[index]
            stock_data = get_price(test_stock, start_date=start_day, end_date=end_day, frequency='daily', fields=['close','high','low','volume'])
            close_prices = stock_data['close'].values
            high_prices = stock_data['high'].values
            low_prices = stock_data['low'].values
            volumes = stock_data['volume'].values
            #通过数据计算指标

            macd, macdsignal, macdhist = talib.MACD(close_prices)
            macd_data = macd[-2]
            rsi_data = talib.RSI(close_prices,timeperiod=10)[-2]
            obv_data = talib.OBV(close_prices, volumes)[-2]
        
            # 保存训练数据中的一组训练数据
            features = []
            features.append(macd_data)
            features.append(rsi_data)
            features.append(obv_data)
            # 特征离散化的时候用到的临时变量，离散化之后删除
            features.append(close_prices[-1])
    
            # 计算分类标签的代码
            start_day = trading_days[index]
            end_day = trading_days[index + 1]
            stock_data = get_price(test_stock, start_date=start_day, end_date=end_day, frequency='daily', fields=['close','high','low','volume'])
            close_prices = stock_data['close'].values
        
            label = False
            if close_prices[-1] > close_prices[-2]:
                label = True
        
            x_all.append(features)
            y_all.append(label)
        
        
        
    # 去除第一行数据
    x_all = x_all[1:]
    y_all = y_all[1:]
    # 训练数据是除去最后一个数据之后的全部数据
    x_train = x_all[:-1]
    y_train = y_all[:-1]
    # 测试数据就是最后一个数据
    x_test = x_all[-1]
    y_test = y_all[-1]
    
    
    #开始利用机器学习算法计算
    clf = GaussianNB()
    # 训练过程
    clf.fit(x_train, y_train)
    # 预测过程
    prediction = clf.predict(x_test)
    return prediction

def process(security):    
    close_data = attribute_history(security, 5, '1d', ['close'])
    # 取得过去五天的平均价格
    MA5 = close_data['close'].mean()
    # 取得上一时间点价格
    current_price = close_data['close'][-1]
    gain=current_price/MA5
    return gain
    

def handle_data(context, data):
    security_order=[]
    init_cash = context.portfolio.cash
    for security in g.stocks:
        start_date = datetime.date(2015,10,9)
        end_date = datetime.date(2015,12,28)
        prediction=Strategy(security,start_date,end_date)
        if prediction==True:
            security_order.append(security)
            
    for security in security_order:
        gain=process(security)
        buying=[]
        buy=[]
        sell1=[]
        
        if context.portfolio.positions[security].closeable_amount == 0:
            buying.append(security)
        for security in buying:
            if gain>1.05:
                order_target_value(security,init_cash*0.75/len(buying))
                log.info("Buying %s" % (security))
            elif gain<1.05 and gain>1:
                order_target_value(security,init_cash*0.05/len(buying))
                log.info("Buying %s" % (security))
                

    for security in g.stocks:
        if gain<0.99 and context.portfolio.positions[security].closeable_amount > 0:        
            order_target(security, 0)
            log.info("Selling %s" % (security))
    
       
