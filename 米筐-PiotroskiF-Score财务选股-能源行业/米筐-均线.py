# 代码运行环境为米筐运行结果不稳定，但大致相同。
import numpy as np
import talib
import pandas as pd
import datetime as dt
from math import log

def init(context):
    # 执行调度器的参数设置
    scheduler.run_weekly(pick_stocks,tradingday=1,time_rule=market_open(minute=30))
    scheduler.run_daily(handle_bar, time_rule = market_open(80))
    # 上证指数,用于判断大盘暴跌来进行清仓
    context.index = '000001.XSHG'
    # 能源指数，用于测算超跌股超跌强度来按不同权重买入
    context.energy='000070.XSHG'
    # 第一套当日清仓方案单日下跌幅度超过-4.3% 且跌破20日线要卖出，这个数值很重要
    context.stop_index_dropone = -0.043
    # 第一套次日清仓方案，在前一日清仓后，因为跌停没有卖出的，今日再次清仓。这个要求次日跌幅也超过-4.3%，如果没有超过，一样不会清仓。尝试使用过-2.8% ，效果不好。最终还是这个数值。
    context.stop_index_droptwo = -0.043
   # 第二套清仓方案 大盘单日跌幅超过-6.8% 要卖出
    context.stop_index_drop = -0.068
   # 超跌买入参数
   # 超跌买入阈值
    context.rel_return = -0.6
    # 大盘回调幅度的观测值
    context.win_length = 160
    # 最大仓位
    context.max_weight = 0.3
    # MACD的几个参数
    context.SHORTPERIOD = 6
    context.LONGPERIOD = 41
    context.SMOOTHPERIOD = 6
    context.OBSERVATION = 400
    # MA的几个参数
    context.SHORT_MA = 5
    context.MID_MA = 10
    context.LONG_MA = 20
    # 定义股票池
    context.stockpool = sector('energy')
    context.stockpool = filter_stlist(context.stockpool)
    context.stockpool = filter_paused_stock(context.stockpool)
    context.stockpool = filter_industry_stock(context, context.stockpool)
    context.firstlist = []
def pick_stocks(context, bar_dict):
    # 盘前选股
    # 1 筛选pb ratio低的股票（取前80%，剔除pb ratio为负的股票）
    fundamental_df = get_fundamentals(
        query(fundamentals.eod_derivative_indicator.pb_ratio).
            filter(fundamentals.eod_derivative_indicator.pb_ratio>0)
            .order_by(fundamentals.eod_derivative_indicator.pb_ratio.asc())
    )
    universe = filter_industry_stock(context, filter_paused_stock(filter_stlist(list(fundamental_df.columns))))
    universe = universe[:int(len(universe) * 0.8)]
    # 2.盈利能力分析
    #2.1 ROA标准化在universe中
    ROA_df = get_fundamentals(
        query(
            fundamentals.financial_indicator.return_on_asset
        ).filter(
            fundamentals.financial_indicator.stockcode.in_(universe)
        )
    )
    ROA_df = ROA_df.T
    ROA_df_standard = ROA_df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    ROA_df_standard = ROA_df_standard.fillna(value = 0)
    # 2.2 资产收益率变化（△ROA）：标准化
    ROA_df1 = get_fundamentals(
        query(
            fundamentals.financial_indicator.return_on_asset
        ).filter(
            fundamentals.financial_indicator.stockcode.in_(universe)
        )
    )
    ROA_df1 = ROA_df1.T
    ROA_df2 = get_fundamentals(
        query(
            fundamentals.financial_indicator.return_on_asset
        ).filter(
            fundamentals.financial_indicator.stockcode.in_(universe)
        ), entry_date = context.now.date() - dt.timedelta(366)
    )
    ROA_df2 = ROA_df2.T
    C_ROA_df = ROA_df1 - ROA_df2
    C_ROA_df.rename(columns={'return_on_asset': 'C_ROA'}, inplace=True)
    C_ROA_df_standard = C_ROA_df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    C_ROA_df_standard = C_ROA_df_standard.fillna(value = 0)
    # 2.3 经营性现金流量（CFO）：数量级较大，除以市值后标准化
    CFO_df = get_fundamentals(
        query(
            fundamentals.cash_flow_statement.cash_flow_from_operating_activities, fundamentals.eod_derivative_indicator.market_cap
        ).filter(
            fundamentals.cash_flow_statement.stockcode.in_(universe)
        )
    )
    CFO_df = CFO_df.T
    CFO_df['lv']=CFO_df['cash_flow_from_operating_activities']/CFO_df['market_cap']
    del CFO_df['cash_flow_from_operating_activities'],CFO_df['market_cap']
    CFO_df_standard= CFO_df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    CFO_df_standard = CFO_df_standard.fillna(value=0)
    # 2.4 公司自然增长获利（ACCRUAL）： 数量级大，除以市值后正则化
    ACC_df = get_fundamentals(
        query(
            fundamentals.financial_indicator.ebit,
            fundamentals.cash_flow_statement.cash_flow_from_operating_activities,fundamentals.eod_derivative_indicator.market_cap
        ).filter(
            fundamentals.cash_flow_statement.stockcode.in_(universe)
        )
    )
    ACC_df = ACC_df.T
    ACC_df['ACC'] = (ACC_df['cash_flow_from_operating_activities'] - ACC_df['ebit'])/ACC_df['market_cap']
    del ACC_df['cash_flow_from_operating_activities'], ACC_df['ebit'],ACC_df['market_cap']
    ACC_df_standard = ACC_df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    ACC_df_standard = ACC_df_standard.fillna(value=0)
    # 3 资本结构打分
    # 3.1 财务杠杆变化（△LEVER）：标准化，反转因子，不解释。
    LEVER_df1 = get_fundamentals(
        query(
            fundamentals.balance_sheet.non_current_liabilities,
            fundamentals.balance_sheet.non_current_assets
        ).filter(
            fundamentals.balance_sheet.stockcode.in_(universe)
        )
    )
    LEVER_df1 = LEVER_df1.T
    LEVER_df2 = get_fundamentals(
        query(
            fundamentals.balance_sheet.non_current_liabilities,
            fundamentals.balance_sheet.non_current_assets
        ).filter(
            fundamentals.balance_sheet.stockcode.in_(universe)
        ), entry_date = context.now.date() - dt.timedelta(365)
    )
    LEVER_df2 = LEVER_df2.T
    LEVER_df3 = get_fundamentals(
        query(
            fundamentals.balance_sheet.non_current_liabilities,
            fundamentals.balance_sheet.non_current_assets
        ).filter(
            fundamentals.balance_sheet.stockcode.in_(universe)
        ), entry_date = context.now.date() - dt.timedelta(730)
    )
    LEVER_df3 = LEVER_df3.T
    LEVER_df1['non_current_assets_before'] = LEVER_df2['non_current_assets']
    LEVER_df1['LEVER'] = 2.0 * LEVER_df1['non_current_liabilities'] / (
                LEVER_df1['non_current_assets'] + LEVER_df1['non_current_assets_before'])
    LEVER_df1 = LEVER_df1.fillna(value=0)
    LEVER_df2['non_current_assets_before'] = LEVER_df3['non_current_assets']
    LEVER_df2['LEVER'] = 2.0 * LEVER_df2['non_current_liabilities'] / (
                LEVER_df2['non_current_assets'] + LEVER_df2['non_current_assets_before'])
    LEVER_df2 = LEVER_df2.fillna(value = 0)
    LEVER_df = pd.DataFrame()
    LEVER_df = LEVER_df1['LEVER'] - LEVER_df2['LEVER']
    LEVER_df = pd.DataFrame(LEVER_df)
    LEVER_df_standard = LEVER_df.apply(lambda x: -((x - np.min(x)) / (np.max(x) - np.min(x))))
    LEVER_df_standard = LEVER_df_standard.fillna(value=0)
    # 3.2流动比率变化（△LIQUID）:标准化
    LIQUID_df1 = get_fundamentals(
        query(
            fundamentals.financial_indicator.current_ratio
        ).filter(
            fundamentals.financial_indicator.stockcode.in_(universe)
        )
    ).T
    LIQUID_df2 = get_fundamentals(
        query(
            fundamentals.financial_indicator.current_ratio
        ).filter(
            fundamentals.financial_indicator.stockcode.in_(universe)
        ), entry_date=context.now.date() - dt.timedelta(365)
    ).T
    LIQUID_df = LIQUID_df1 - LIQUID_df2
    LIQUID_df_standard = LIQUID_df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    LIQUID_df_standard = LIQUID_df_standard.fillna(value = 0)
    # 4运营效率打分
    # 4.1 毛利润率变化（△MARGIN）：标准化
    MARGIN_df1 = get_fundamentals(
        query(
            fundamentals.financial_indicator.gross_profit_margin
        ).filter(
            fundamentals.financial_indicator.stockcode.in_(universe)
        )
    ).T
    MARGIN_df2 = get_fundamentals(
        query(
            fundamentals.financial_indicator.gross_profit_margin
        ).filter(
            fundamentals.financial_indicator.stockcode.in_(universe)
        ), entry_date=context.now.date() - dt.timedelta(365)
    ).T
    MARGIN_df = MARGIN_df1 - MARGIN_df2
    MARGIN_df_standard = MARGIN_df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    MARGIN_df_standard = MARGIN_df_standard.fillna(value = 0)
    # 4.2 资产周转率变化（△TURN）：资产周转率：标准化
    TURN_df1 = get_fundamentals(
        query(
            fundamentals.financial_indicator.total_asset_turnover
        ).filter(
            fundamentals.financial_indicator.stockcode.in_(universe)
        )
    )
    TURN_df1 = TURN_df1.T
    TURN_df2 = get_fundamentals(
        query(
            fundamentals.financial_indicator.total_asset_turnover
        ).filter(
            fundamentals.financial_indicator.stockcode.in_(universe)
        ), entry_date = context.now.date() - dt.timedelta(366)
    )
    TURN_df2 = TURN_df2.T
    TURN_df = TURN_df1 - TURN_df2
    TURN_df_standard = TURN_df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    TURN_df_standard = TURN_df_standard.fillna(value = 0)
     # 5 最后，把所有的表格接到一起，将上述8项财务指标的得分加总，计算排序，选出得分最高30只股票
    total_df = pd.concat([ROA_df_standard, C_ROA_df_standard, CFO_df_standard, ACC_df_standard, LEVER_df_standard, LIQUID_df_standard, MARGIN_df_standard, TURN_df_standard], axis=1)
    total_df['score'] = total_df['return_on_asset'] + total_df['C_ROA'] + total_df['lv'] + total_df['ACC'] + total_df['LEVER'] +total_df['current_ratio'] + total_df['gross_profit_margin'] + total_df['total_asset_turnover']
    total_df = total_df.sort(['score'], ascending = [False])
    total_df_30=total_df.head(30)
    financial_filter_list = list(total_df_30.index)
    context.firstlist = financial_filter_list
# 排除ST股
def filter_stlist(stock_list):
    return [ticker for ticker in stock_list if not is_st_stock(ticker)]
# 过滤停牌股票
def filter_paused_stock(stock_list):
    return [stock for stock in stock_list if not is_suspended(stock)]
# 过滤不在行业范围内的股票
def filter_industry_stock(context, stock_list):
    return [stock for stock in stock_list if stock in context.stockpool]
# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_bar(context, bar_dict):
    #先卖出不在列表中的股票
    for stock in context.portfolio.positions.keys():
        if stock not in context.firstlist:
            order_target_percent(stock, 0)

    index_hist = history_bars(context.index, 3, "1d", "close") # 前3个交易日的指数 index是上证指数
    index_return_1d = log(index_hist[2]/index_hist[1]) # log(昨日指数/前日指数)4号清仓信号
    index_return_2d = log(index_hist[1]/index_hist[0]) # 昨天

    con1 = index_return_1d < context.stop_index_dropone

    con2 = index_return_2d < context.stop_index_droptwo

    indexma5 = bar_dict[context.index].mavg(5,frequency = 'day')
    indexma20 = bar_dict[context.index].mavg(20,frequency = 'day')

    macondiction = indexma5 < indexma20
# 清仓条件一触发
    if con2 and macondiction:
        for stock in context.portfolio.positions.keys():
            order_target_percent(stock, 0)
        return

#   也许还有跌停的，清仓条件二触发
    if con1 and macondiction:
        for stock in context.portfolio.positions.keys():
            order_target_percent(stock, 0)
        return

    if index_return_1d < context.stop_index_drop:
        for stock in context.portfolio.positions.keys():
            order_target_percent(stock, 0)
        return
    # 所有根据均线选中的指标将放在这里
    second_list=[]

    for stock in context.firstlist:
        # 参数准备
        open_array = history_bars(stock, context.OBSERVATION, '1d', 'open')
        high_array = history_bars(stock, context.OBSERVATION, '1d', 'high')
        low_array = history_bars(stock, context.OBSERVATION, '1d', 'low')
        close_array = history_bars(stock, context.OBSERVATION, '1d', 'close')
        volume_array = history_bars(stock, context.OBSERVATION, '1d', 'volume')
        macd, signal, hist = talib.MACD(close_array, context.SHORTPERIOD, context.LONGPERIOD, context.SMOOTHPERIOD)
        # 均线配合量能的信号，昨日MA5小于MA20,今日放量突破PVMA5>PVMA20，以及PVMA5>PVMA10>PVMA20多头排列配合量能。
        short_avg = talib.SMA(close_array, context.SHORT_MA)
        mid_avg = talib.SMA(close_array, context.MID_MA)
        long_avg = talib.SMA(close_array, context.LONG_MA)
        up_array=0
        down_array=0
        # 这里获取20日PVMA
        i=1
        for i in range(1,20):
            up_array=up_array+volume_array[-i]*(open_array[-i]+close_array[-i])/2
            down_array=down_array+volume_array[-i]
        pvma20_value = up_array/down_array
        # 这里获取5日PVMA
        up_value=0
        down_value=0
        s=1
        for s in range(1,5):
            up_value=up_array+volume_array[-s]*(open_array[-s]+close_array[-s])/2
            down_value=down_array+volume_array[-s]
        pvma5_value = up_value/down_value
        # 这里获取10日PVMA
        up_word=0
        down_word=0
        t=1
        for t in range(1,10):
            up_word=up_word+volume_array[-t]*(open_array[-t]+close_array[-t])/2
            down_word=down_word+volume_array[-t]
        pvma10_value=up_word/down_word

        # 设立底背离卖出，先卖出，有资金再买入
        if hist[-1]<hist[-2]<hist[-3] and close_array[-1]>close_array[-2]:
            order_target_percent(stock,0)

        # 背离
        if hist[-1]>hist[-2]>hist[-3] and close_array[-1]<close_array[-2]:
            second_list.append(stock)
        # 金叉
        if macd[-1]>signal[-1] and macd[-2]<signal[-2]:
            second_list.append(stock)
        # 变盘
        if hist[-1]>hist[-2]>hist[-3] and hist[-1]>0 and hist[-2]<0:
            second_list.append(stock)
        # 放量上涨信号
        if (short_avg[-2] < long_avg[-2] and pvma5_value > pvma20_value) or (pvma5_value > pvma10_value > pvma20_value):
            second_list.append(stock)

    second_list=list(set(second_list))
    final_list=second_list

    # 这是加上上证指数的list，用于后续择时操作。
    second_list.append(context.energy)
    update_universe(second_list)
    # 在这里是真正的择时部分，由于second_list是最终列表，所以就不单独设置函数，择时部分主要是看股票相对于大盘的超跌幅度进行买入，超跌越多，买入越多。
    # 160天以来，所有被选出股票相对上证的涨跌幅度，所以这个是dataframe
    stock_hist = history(context.win_length, "1d", "close")
    # 过去N天每个股票的回调幅度 -0.03表示跌了3%
    stock_return = (stock_hist.ix[context.win_length-1]-stock_hist.ix[0])/stock_hist.ix[0]
    # 过去N天上证的回调幅度
    index_return = stock_return[context.energy]
    # 相对强度，正数表示股票表现较强，负数表示表现弱
    rel_return = stock_return - index_return
    # 遍历所有候选池子，要求
    # 1. 股票可以交易
    # 2. 股票没有涨停跌停
    # 3. 此股票的回调差 < 上证回调差绝对值的-0.6倍，说明股票超跌
    final_list = [stock for stock in final_list
    if bar_dict[stock].is_trading
    and bar_dict[stock].open< 1.095*stock_hist[stock].iloc[-1]
    and bar_dict[stock].open> 0.905*stock_hist[stock].iloc[-1]
    and rel_return[stock]<abs(index_return)*context.rel_return]
    # 备选股数为0就不交易
    if len(final_list) == 0:
            return
    # 超跌越多，权重越大，最大30%， 最小0%
    weight = {} # 保存每个股票的超跌幅度作为权重
    sum_weight = 0 # 所有超跌幅度求和，为后续归一化准备
    for stock in final_list:
        weight[stock] = abs((rel_return[stock]-abs(index_return)*context.rel_return)/(index_return) ) * context.max_weight
        # 超跌的相对情况
        if weight[stock] > context.max_weight:
            weight[stock] = context.max_weight
        sum_weight += weight[stock]

    for stock in final_list:
        weight[stock] /= sum_weight # 归一化
        if weight[stock] > context.max_weight: # 单个股票仓位控制
            weight[stock] = context.max_weight
        order_target_percent(stock, weight[stock])

# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading(context):
    pass
