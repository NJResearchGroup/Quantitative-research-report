import numpy as np
import pandas as pd
import datetime
import time
import talib

#第二部分：运用关联规则apriori算法得出股票暴涨的频繁模式

##股票特征处理
def chs_ticker(dataday,n = 4):
    k = list(pd.qcut(dataday,n,labels = [str(x) for x in range(1,n+1)]))
    l = k[0]
    for i in k:
        l = l+i
    l = l[1:]
    return l

##分位数提取
def percent(p,n):
    perti = []
    for i in np.linspace(0,100,n+1)[1:-1]:
        pt = np.percentile(p,i)
        perti.append(pt)
    return perti
##序列生成函数
def n_list(glist,x = 2):
    relist = []
    for k in range(0,len(glist)-x+1):
        relist.append(glist[k:k+x])
    return relist
##大涨模式
def dz_parn(dazhang,n = 4):
    dz_pattern = []
    for i in dazhang:
        if i > 5:
            i_add = [x for x in range(i-n+1,i+1)]
            t = list(dataday.ix[i_add]['label'])
            f = t[0]
            for i in t[1:]:
                f = f+i
            dz_pattern.append(f)
    return dz_pattern
##ck序列与置信度生成函数
def ck(frq,x,n=4):
    klist = []
    frq_k = []
    sup_k = []
    cond_s = []
    re_list = n_list(l,x-1)
    ren_list = n_list(l,x)
    total_num = len(ren_list)
    for i in frq:
        b = np.float(re_list.count(i))
        for j in [str(x) for x in range(1,n+1)]:
            k = i+j
            p = ren_list.count(k)
            if p/np.float(total_num)>0.01:
                if (p/b)>0.3:
                    frq_k.append(k)
                    sup_k.append(p/np.float(total_num))
                    cond_s.append(p/b)
    return frq_k,sup_k,cond_s

##同时满足大涨模式与频繁模式
def chs_dz(frq_,dz_pattern):
    choose_list = []
    for choose in frq_:
        if choose in dz_pattern:
            choose_list.append(choose)
    return choose_list

tick = list(DataAPI.MktEqudGet(tradeDate=u"20180511",secID=u"",ticker=current_ind.ticker,beginDate=u'',endDate= u'',isOpen="1",field=u'',pandas="1")['ticker'])
now = time.time()
for ticker in tick[0:235]+tick[237:]:
    dataday = chs_factor(ticker)
    dataday['label'] = pd.qcut(dataday['chgPct'],4,labels = ['1','2','3','4'])
    l = chs_ticker(dataday['chgPct'])
    dazhang = dataday[dataday['chgPct']>0.05].index
    frq = list(set(n_list(l,2)))
    frq = ck(frq,3)[0]
    if len(ck(frq,4)[0]) != 0:
        frq_,sup,cond = ck(frq,4)
        ku = pd.DataFrame([frq_,sup,cond],index = ['frq','sup','con']).T
        dz_pattern = dz_parn(dazhang)
        choose_list = chs_dz(frq_,dz_pattern)
        ku.index = list(ku.frq)
        ku = ku.ix[choose_list]
        ku.index = range(len(ku))
    print ticker,ku
time.time()-now

#第三部分 回测优化

start = '2015-12-01'                       # 回测起始时间
end = '2016-03-01'                         # 回测结束时间
benchmark = 'HS300'                        # 策略参考标准
#universe = DynamicUniverse(IndSW.CaiJueL1)
universe = ['002230.XSHE','300168.XSHE','000034.XSHE','000021.XSHE','000066.XSHE','000555.XSHE','000748.XSHE','000909.XSHE','000938.XSHE','000948.XSHE','600100.XSHG','000997.XSHE','600271.XSHG','601360.XSHG','600446.XSHG','600764.XSHG','600570.XSHG','600850.XSHG','600845.XSHG','600601.XSHG']
# universe = universeInit()                # 证券池，支持股票和基金

freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测，'m'表示日内策略使用分钟线回测
refresh_rate = 1                          # 调仓频率
# 配置账户信息，支持多资产多账户
accounts = {
    'fantasy_account': AccountConfig(account_type='security', capital_base=100000)
}

def initialize(context):
    pass


########################################################################################
#    Aprior algorithm
########################################################################################
def elementsDet(datasets):
    if type(datasets) == list:
        elements = {}
        for index in datasets:
            for index1 in index:
                if elements.has_key(index1) == False:
                    elements[index1] = 1
                else:
                    elements[index1] += 1
        return elements
    if type(datasets) == dict:
        elements = {}
        for index in datasets:
            if type(index) == tuple:
                index = list(index)
                for index1 in index:
                    if elements.has_key(index1) == False:
                        elements[index1] = 0
            else:
                elements[index] = 0
        return elements
    pass

def checkAssociation(subset,objset):
    for index in subset:
        if index not in objset:
            return False
    return True
    pass

def support(subset,datasets):
    count = 0
    for transaction in datasets:
        if checkAssociation(subset,transaction) == True:
            count += 1
    return 1.0*count/len(datasets)
    pass

def apriori(datasets,minsup):
    candidateIterator = []
    electIterator = []
    length = len(datasets)
    ##init part 
    #the candidate
    elements = elementsDet(datasets)
    candidate = {}
    for index in elements:
            candidate[index] = 1.0*elements[index]/length
    candidateIterator.append(candidate)
    #the elect
    elect = {}
    for index in candidate:
        if candidate[index] > minsup:
            elect[index] = candidate[index]
    electIterator.append(elect)

    ##the update part
    itera = 1
    while(len(electIterator[-1]) != 0):

        candidateOld = candidateIterator[-1]
        electOld = electIterator[-1]
        elementsOld = elementsDet(electOld)
        # print elementsOld
        candidate = {}
        
        ##the candidate
        for index in electOld:
            for index1 in elementsOld:
                if type(index) != list and type(index) != tuple:
                    if index1 != index:
                        tmp = []
                        tmp.append(index)
                        tmp.append(index1)
                        tmp.sort()
                        if candidate.has_key(tuple(tmp)) == False:
                            candidate[tuple(tmp)] = 0

                if type(index) == tuple:
                    tmp = list(index)
                    if tmp.count(index1) == False:
                        tmp1 = tmp
                        tmp1.append(index1)
                        tmp1.sort()
                        if candidate.has_key(tuple(tmp1)) == False:
                            candidate[tuple(tmp1)] = 0
        candidateIterator.append(candidate)

        ##the elect 
        elect = {}
        for index in candidate:
            candidate[index] = support(index,datasets)

        for index in candidate:
            if candidate[index] > minsup:
                elect[index] = candidate[index]
        electIterator.append(elect)

        # print 'iteartion ' + str(itera) + ' is finished!'
        itera += 1

    ##the elected frequency sets dictionary: the value is the key's support
    electedDict = {}
    for index in electIterator:
        for index1 in index:
            electedDict[index1] = index[index1]

    ##the elected frequency sets lists
    electedList = []
    for index in electIterator:
        tmp = []
        for index1 in index:
            if type(index1) == tuple:
                tmp1 = []
                for ele in index1:
                    tmp1.append(ele)
                tmp.append(tmp1)
            else:
                tmp.append([str(index1)])
        tmp.sort()
        for index1 in tmp:
            electedList.append(index1)

    return electedDict,electedList


####deal with the trading signals
def handle_data(context):                  # 每个交易日的买入卖出指令
    ####Presettings
    histLength = 10
    stockDataThres = 0

    ####Dictionary of the return Rate 
    closePrice = context.get_attribute_history('closePrice',histLength)
    retRate = {}
    for index in context.get_universe(exclude_halt=True):
        retRate[index] = ((np.array(closePrice[1:][index]) - np.array(closePrice[:-1][index]))/np.array(closePrice[:-1][index])).tolist()
    #print np.array(closePrice[1:][index])
    #print "N"
    #print closePrice[:-1][index]
    #print index
    account = context.get_account('fantasy_account')
    ###ret list of the benchmark
 
    ####List of transactions
    transactions = []
    for index in range(histLength-1):
        tmpt = []
        for stock in context.get_universe(exclude_halt=True):
            if retRate[stock][index] > stockDataThres:
            # if retRate[stock][index] > bmRet[index]:
                tmpt.append(stock)
        transactions.append(tmpt)
    #print transactions    
    ####List of hot stocks
    hotStock = []
    hotStockDict,hotStockList = apriori(transactions,0.5)
    for index in hotStockList:
        for stock in index:
            if stock not in hotStock:
                hotStock.append(stock)
       
    ####List of the portfolio
    
    #validSecHist = context.get_attribute_history('closePrice',5)
    #junxian = sum(validSecHist[index][0:4])
    #gain=validSecHist[index][-1]/junxian
    #close_data = attribute_history(security, 5, '1d', ['close'])
    # 取得过去五天的平均价格
    #MA5 = talib.SMA(validSecHist,timeperiod=5)
    # 取得上一时间点价格
    #current_price = context.get_attribute_history('closePrice',1)
    #gain=current_price/MA5
    hist=context.get_attribute_history('closePrice',30)
    for index in universe:
        ma5=hist[index][-6:-2].mean()
        close_price=hist[index][-1]
        gain=close_price/ma5
        if index in hotStock:
            if gain>=1.05:
                amount=amount = 0.3*account.cash/len(hotStock)/context.current_price(index)
                account.order(index,amount)
            elif gain<1.05 and gain>1:
                amount=amount = 0.05*account.cash/len(hotStock)/context.current_price(index)
                account.order(index,amount)
        else:
            account.order(index,0)              
    return
