from wordcloud import WordCloud
import pandas as pd
import numpy as np
import jieba
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import july
from july.utils import date_range

def get_wordcloud(data_path, save_path, is_sender):

    df = pd.read_csv(data_path, encoding='utf-8', parse_dates=['StrTime'])  # 解析时间列为日期时间类型
    df = df[df['IsSender'] == is_sender]  # 筛选发送者
    df = df[df['Type'] == 1]  # 筛选消息类型

    # 日期筛选
    # df = df[(df['StrTime'] >= '2023-01-01') & (df['StrTime'] < '2024-01-01')]

    # 提取消息内容
    texts = df['StrContent'].tolist()

    with open("CNstopwords.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        stopwords = [line.strip().replace("\ufeff", "") for line in lines]

    # 分词，去除停用词和表情（表情都是这样的格式：[xx]）
    norm_texts = []
    pattern = re.compile(r"(\[.+?\])")

    for text in texts:
        text = str(text)
        text = pattern.sub('', text).replace("\n", "")  # 删除表情、换行符
        words = [word for word in jieba.lcut(text) if word not in stopwords and len(word) > 1]
        norm_texts.extend(words)

    count_dict = dict(Counter(norm_texts))

    # 单独处理高频词“哈哈哈哈”
    if '哈哈哈哈' in count_dict:
        count_dict['哈哈哈'] += count_dict.pop('哈哈哈哈')

    # 将字典转换为列表，并按照词频降序排列
    sorted_items = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
    top_ten = sorted_items[:10]

    for word, count in top_ten:
        print(f"{word}: {count}")

    wc = WordCloud(font_path="simhei.ttf", background_color='white', include_numbers=False,
                   width=1000, height=1000, scale=2, random_state=0)  # 如果不指定中文字体路径，词云会乱码
    wc = wc.fit_words(count_dict)

    # 根据发送者设置标题
    sender_title = "Mr. Wang" if is_sender else "Miss Ye"
    plt.title(f'Word Cloud of Number of Messages Sent by {sender_title}')
    plt.imshow(wc)
    plt.show()
    wc.to_file(save_path)

def plot_monthly_histogram(data_path, save_path, is_sender):
    # 读取数据
    df = pd.read_csv(data_path, encoding='utf-8')

    # 重置索引或使用 .loc 进行筛选
    sender_msgs = df.loc[(df['IsSender'] == is_sender) & (df['Type'] == 1)].copy()

    # 将时间字符串转换为 pandas 的 datetime 对象
    sender_msgs['StrTime'] = pd.to_datetime(sender_msgs['StrTime'])

    # 提取年/月信息
    sender_msgs['Month'] = sender_msgs['StrTime'].dt.to_period('M').astype(str)
    sender_msgs['Year'] = sender_msgs['StrTime'].dt.year.astype(str)

    # 限制数据在指定日期范围内
    # sender_msgs = sender_msgs[(sender_msgs['StrTime'] >= '2023-01-01') & (sender_msgs['StrTime'] < '2024-01-01')]
    sender_msgs = sender_msgs[(sender_msgs['StrTime'] < '2024-01-01')]

    # 统计每年/月的消息数量
    monthly_counts = sender_msgs.groupby('Month').size()

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    sns.histplot(data=sender_msgs, x='Month', bins=len(monthly_counts), hue='Year', multiple='stack')

    # 根据发送者设置标题
    sender_title = "Mr. Wang" if is_sender else "Miss Ye"
    plt.title(f'Monthly Histogram of Number of Messages Sent by {sender_title}')
    plt.xticks([])
    plt.xlabel('Month')
    plt.ylabel('Number of Messages')
    plt.xticks(rotation=45, ha='right')

    # 在直方图上显示具体数字
    # for i, count in enumerate(monthly_counts):
    #     plt.text(i, count, str(count), ha='center', va='bottom')

    sns.histplot(data=sender_msgs, x='Month', bins=len(monthly_counts), kde=True, alpha=0.12)

    plt.tight_layout()

    # 保存图片
    plt.savefig(save_path)
    plt.show()

def plot_heatmap(data_path, save_path, is_sender):
    # 读取数据
    df = pd.read_csv(data_path, encoding='utf-8')

    # 重置索引或使用 .loc 进行筛选
    sender_msgs = df.loc[(df['IsSender'] == is_sender) & (df['Type'] == 1)].copy()

    # 将时间字符串转换为 pandas 的 datetime 对象
    sender_msgs['StrTime'] = pd.to_datetime(sender_msgs['StrTime'])

    # 提取日期信息
    sender_msgs['Date'] = sender_msgs['StrTime'].dt.date

    # 限制数据在指定日期范围内
    sender_msgs = sender_msgs[(sender_msgs['StrTime'] >= '2023-01-01') & (sender_msgs['StrTime'] < '2024-01-01')]

    # 统计每天的消息数量
    daily_counts = sender_msgs.groupby('Date').size()

    # 生成日期和对应的消息数量数据框
    date_range = pd.date_range(start='2023-01-01', end='2023-12-31')
    date_counts = pd.DataFrame({'Date': date_range, 'Count': 0})
    date_counts['Count'] = date_counts['Date'].map(daily_counts).fillna(0)

    # 构建日历热力图数据框
    calendar_data = date_counts.pivot_table(index=date_counts['Date'].dt.month, 
                                            columns=date_counts['Date'].dt.day, 
                                            values='Count', 
                                            fill_value=0)

    # 绘制热力图
    plt.figure(figsize=(12, 8))
    sns.heatmap(calendar_data, cmap='YlGnBu', linewidths=0.5, linecolor='gray', cbar=True)

    # 根据发送者设置标题
    sender_title = "Mr. Wang" if is_sender else "Miss Ye"
    plt.title(f'Calendar Heatmap of Number of Messages Sent by {sender_title}')
    
    plt.xlabel('Day')
    plt.ylabel('Month')
    plt.tight_layout()

    # 保存图片
    plt.savefig(save_path)
    plt.show()

def count_total_characters(data_path, is_sender):

    df = pd.read_csv(data_path, encoding='utf-8')

    # 筛选出指定发送者的消息
    sender_msgs = df[(df['IsSender'] == is_sender) & (df['Type'] == 1)]

    # 日期筛选
    # sender_msgs = sender_msgs[(sender_msgs['StrTime'] >= '2023-01-01') & (sender_msgs['StrTime'] < '2024-01-01')]

    # 计算总字数
    total_characters = sender_msgs['StrContent'].apply(lambda x: len(str(x))).sum()

    return total_characters

def find_most_least_active_day(data_path, is_sender):

    df = pd.read_csv(data_path, encoding='utf-8')

    # 筛选出指定发送者的消息
    sender_msgs = df[(df['IsSender'] == is_sender) & (df['Type'] == 1)]

    # 将时间字符串转换为日期
    sender_msgs['Date'] = pd.to_datetime(sender_msgs['StrTime']).dt.date
    
    # 日期筛选
    sender_msgs = sender_msgs[(sender_msgs['StrTime'] >= '2023-01-01') & (sender_msgs['StrTime'] < '2024-01-01')]

    # 计算每天的总字数
    daily_character_counts = sender_msgs.groupby('Date')['StrContent'].apply(lambda x: x.str.len().sum()).reset_index()

    # 找出字数最多的一天和最少的一天
    most_active_day = daily_character_counts.loc[daily_character_counts['StrContent'].idxmax()]
    least_active_day = daily_character_counts.loc[daily_character_counts['StrContent'].idxmin()]

    return most_active_day, least_active_day



data_path = "data.csv"

# total_characters_0 = count_total_characters(data_path, is_sender=0)
# total_characters_1 = count_total_characters(data_path, is_sender=1)

# print("TA发送的总字数：", total_characters_0)
# print("你发送的总字数：", total_characters_1)

# most_active_day_0, least_active_day_0 = find_most_least_active_day(data_path, is_sender=0)
# most_active_day_1, least_active_day_1 = find_most_least_active_day(data_path, is_sender=1)

# print("TA字数最多的一天：", most_active_day_0['Date'], "，字数：", most_active_day_0['StrContent'])
# print("TA字数最少的一天：", least_active_day_0['Date'], "，字数：", least_active_day_0['StrContent'])

# print("你字数最多的一天：", most_active_day_1['Date'], "，字数：", most_active_day_1['StrContent'])
# print("你字数最少的一天：", least_active_day_1['Date'], "，字数：", least_active_day_1['StrContent'])

# 词云图
# get_wordcloud(data_path,save_path="总词云-TA.png",is_sender=0)
# get_wordcloud(data_path,save_path="总词云-你.png",is_sender=1)

# 2023-2024年每月聊天数
# plot_monthly_histogram(data_path, save_path="聊天数-TA.png", is_sender=0)
# plot_monthly_histogram(data_path, save_path="聊天数-你.png", is_sender=1)

# 2023-2024年聊天热力图
# plot_heatmap(data_path, save_path="热力图-TA.png", is_sender=0)
# plot_heatmap(data_path, save_path="热力图-你.png", is_sender=1)