import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def evaluator(df, min_date, max_date, train_model_fn, stock_name, nshares=None):
    """
    评估预测模型
    """
    # 起始日期是结束日期一年前
    start_date = max_date - pd.DateOffset(years=1)
    end_date = max_date
    # 获取训练数据DataFrame，数据是在start_date前三年
    train = df[(df['Date'] < start_date.date()) & 
                       (df['Date'] > (start_date - pd.DateOffset(years=3)).date())]
    # 获取测试数据DataFrame，最后一年就是测试数据
    test = df[(df['Date'] >= start_date.date()) & (df['Date'] <= end_date.date())]
    # 训练模型，365表示预测未来一年的
    model, future = train_model_fn(train, 365, True, True)
    # 合并测试数据和预测的数据
    # 参数on：表示条件，在两个DataFrame中都必须要有ds列
    # 参数how：表示内连接，就像SQL中的inner join一样
    test = pd.merge(test, future, on='ds', how='inner')
    # 通过内连接的方式，合并训练数据和预测的数据
    train = pd.merge(train, future, on='ds', how='inner')
    # 计算DataFrame中的元素的差异
    test['pred_diff'] = test['yhat'].diff()
    test['real_diff'] = test['y'].diff()
    # 计算真实的数值和预测的数值是否正确
    test['correct'] = (np.sign(test['pred_diff']) == np.sign(test['real_diff'])) * 1
    # 计算差异值增加时的准确度和减少时的准确度
    increase_accuracy = 100 * np.mean(test[test['pred_diff'] > 0]['correct'])
    decrease_accuracy = 100 * np.mean(test[test['pred_diff'] < 0]['correct'])
    # 计算测试数据的平均绝对误差
    test_errors = abs(test['y'] - test['yhat'])
    test_mean_error = np.mean(test_errors)
    # 计算训练数据的平均绝对误差
    train_errors = abs(train['y'] - train['yhat'])
    train_mean_error = np.mean(train_errors)
    # 标记真实测试数值是在预测的数值的上限和下限范围内
    test['in_range'] = False
    for i in test.index:
        if (test.loc[test.index[i], 'y'] < test.loc[test.index[i], 'yhat_upper']) \
         & (test.loc[test.index[i], 'y'] > test.loc[test.index[i], 'yhat_lower']):
            test.loc[test.index[i], 'in_range'] = True
    # 计算范围内的平均值
    in_range_accuracy = 100 * np.mean(test['in_range'])
    # 获取只有预测到上涨的股票数据
    test_pred_increase = test[test['pred_diff'] > 0]
    test_pred_increase.reset_index(inplace=True)
    # 遍历所有的预测数值并计算收益
    prediction_profit = []
    for i, correct in enumerate(test_pred_increase['correct']):
        # 计算预测收益
        prediction_profit.append(nshares * test_pred_increase.loc[i, 'real_diff'])
    test_pred_increase['pred_profit'] = prediction_profit
    # 通过左连接的方式，合并测试数据和预测收益数据
    test = pd.merge(test, test_pred_increase[['ds', 'pred_profit']], on='ds', how='left')
    test.loc[0, 'pred_profit'] = 0
    # 计算预测累计的和
    test['pred_profit'] = test['pred_profit'].cumsum().ffill()
    # 计算利润
    test['hold_profit'] = nshares * (test['y'] - float(test.loc[0, 'y']))
    # 打印输出信息
    print("{}的股票从{}到{}，购买{}股。".format(stock_name, 
                                             start_date.date(), 
                                             end_date.date(), 
                                             nshares))
    print("当模型预测到上涨时，股价在此时间点上涨{}%。".format(increase_accuracy))
    print("当模型预测到下跌时，股价在此时间点下跌{}%。".format(decrease_accuracy))
    print("使用Prophet模型预测整个收益是：${}。".format(np.sum(prediction_profit)))
    print("买入并持有的利润：${}。".format(float(test.loc[len(test) - 1, 'hold_profit'])))

    # Final profit and final smart used for locating text
    final_profit = test.loc[len(test) - 1, 'pred_profit']
    final_smart = test.loc[len(test) - 1, 'hold_profit']
    # 文本定位
    last_date = test.loc[len(test) - 1, 'ds']
    text_location = (last_date - pd.DateOffset(months=1)).date()
    # 绘图使用黑色背景
    plt.style.use('default')
    # 绘制持有利润
    plt.plot(test['ds'], 
             test['hold_profit'], 
             'o-', 
             color='b',
             linewidth=1.8, 
             label='Buy and Hold Strategy') 
    # 绘制预测收益
    plt.plot(test['ds'], 
             test['pred_profit'], 
             '+-', 
             color=('g' if final_profit > 0 else 'r'),
             linewidth=1.8, 
             label='Prediction Strategy')
    # 绘制在图上的文字值
    plt.text(x=text_location, 
             y=final_profit + (final_profit / 40),
             s="${:.1f}".format(final_profit),
            color='m' if final_profit > 0 else 'r',
            size=18)
    plt.text(x=text_location, 
             y=final_smart + (final_smart / 40),
             s="${:.1f}".format(final_smart),
            color='m' if final_smart > 0 else 'r',
            size=18)
    # 设置绘图的一些属性
    plt.ylabel('Profit  (US $)') 
    plt.xlabel('Date')
    plt.title('Predicted versus Buy and Hold Profits')
    plt.legend(loc=2, prop={'size': 10})
    plt.grid(alpha=0.2)
    plt.show()
