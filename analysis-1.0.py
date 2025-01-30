import pandas as pd
import numpy as np
import jieba
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from wordcloud import WordCloud
from snownlp import SnowNLP
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Template

# 全局样式设置
plt.style.use('ggplot')
sns.set_palette(sns.color_palette(["#ff69b4", "#db7093"]))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置 Plotly 的模板为适合移动端的模板
pio.templates.default = "plotly_dark"

#%% 核心分析类
class WeChatLoveAnalyzer:
    def __init__(self, data_path):
        """
        初始化分析器
        :param data_path: 聊天记录CSV文件路径
        """
        self.df = self._preprocess_data(data_path)
        self.stopwords = self._load_stopwords("CNstopwords.txt")
        
    def _preprocess_data(self, data_path):
        """数据预处理管道"""
        df = pd.read_csv(data_path, parse_dates=['StrTime'])
        df = df[df['Type'] == 1]  # 仅分析文本消息
        df = df.dropna(subset=['StrContent'])  # 移除空内容

        # 小红书链接虽然是链接，但微信会识别为文本。所以需要去除包含小红书链接的消息
        df = df[~df['StrContent'].str.contains(r'https://www\.xiaohongshu\.com', na=False)]
        
        # 时间特征提取
        df['date'] = df['StrTime'].dt.date
        df['hour'] = df['StrTime'].dt.hour
        df['weekday'] = df['StrTime'].dt.weekday
        df['month'] = df['StrTime'].dt.month
        return df
    
    def _load_stopwords(self, path):
        """加载停用词表"""
        with open(path, 'r', encoding='utf-8') as f:
            return set([line.strip() for line in f])
    
    #%% 核心分析方法
    def generate_wordcloud(self, is_sender, save_path):
        
        texts = self.df[self.df['IsSender'] == is_sender]['StrContent']
        pattern = re.compile(r"\[.+?\]")  # 过滤表情符号
        
        words = []
        for text in texts:
            clean_text = pattern.sub('', str(text))
            words += [word for word in jieba.lcut(clean_text) 
                     if word not in self.stopwords and len(word) > 1]
        
        word_count = Counter(words)
        wc = WordCloud(font_path="simhei.ttf", 
                      mask=None,
                      background_color='white',
                      colormap='RdPu',
                      width=1600, height=1200)
        wc.generate_from_frequencies(word_count)
        
        plt.figure(figsize=(20,15))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    def plot_calendar_heatmap(self, is_sender, save_path):
        """交互式日历热力图"""
        df = self.df[self.df['IsSender'] == is_sender]
        daily_counts = df.groupby('date').size().reset_index(name='count')
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])
        
        fig = px.density_heatmap(
            daily_counts,
            x=daily_counts['date'].dt.day,
            y=daily_counts['date'].dt.month_name(),
            z='count',
            histfunc="sum",
            color_continuous_scale='reds'
        )
        fig.update_layout(
            yaxis_title="月份",
            xaxis_title="日期",
            height=800
        )
        fig.write_image(save_path)
    
    def analyze_response_time(self, save_path):
        """响应时间分析"""
        temp_df = self.df.sort_values('StrTime')
        temp_df['time_diff'] = temp_df.groupby('IsSender')['StrTime'].diff()
        response_df = temp_df[temp_df['IsSender'].diff() != 0]
        
        plt.figure(figsize=(12,6))
        sns.boxplot(x='IsSender', y='time_diff', data=response_df, 
                   showfliers=False, width=0.4)
        plt.ylabel('响应时间间隔', fontsize=12)
        plt.xticks([0,1], ['TA的回复速度', '你的回复速度'])
        plt.title('消息响应时间分布分析', fontsize=14)
        plt.savefig(save_path)
        plt.close()
    
    #%% 高级分析方法
    def create_love_report(self):
        """生成综合报告"""
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'xy'}, {'type': 'polar'}],
                   [{'type': 'xy'}, {'type': 'xy'}]],
            subplot_titles=("每日聊天趋势", "消息类型分布", 
                           "情感波动曲线", "时段分析")
        )
        
        # 每日趋势
        daily_counts = self.df.groupby('date').size()
        fig.add_trace(go.Scatter(x=daily_counts.index, y=daily_counts.values,
                                mode='lines', name='每日消息量'),
                     row=1, col=1)
        
        # 消息类型雷达图
        type_counts = self.df.groupby(['IsSender', 'Type']).size().unstack()
        fig.add_trace(go.Scatterpolar(
            r=type_counts.loc[0].values,
            theta=['文字','图片','语音','视频','其他'],
            fill='toself',
            name='TA'
        ), row=1, col=2)
        fig.add_trace(go.Scatterpolar(
            r=type_counts.loc[1].values,
            theta=['文字','图片','语音','视频','其他'],
            fill='toself',
            name='你'
        ), row=1, col=2)
        
        # 情感分析
        self.df['sentiment'] = self.df['StrContent'].apply(
            lambda x: SnowNLP(str(x)).sentiments)
        try:
            weekly_sentiment = self.df.resample("W", on="StrTime")["sentiment"].mean()
        except Exception as e:
            print(f"情感分析数据生成失败: {str(e)}")
            return
        fig.add_trace(go.Scatter(x=weekly_sentiment.index, y=weekly_sentiment.values,
                                mode='lines+markers', name='情感值'),
                     row=2, col=1)
        # fig.add_hline(y=0.5, line_dash="dot", row=2, col=1, annotation_text="中性阈值")
        
        # 时段分析
        hour_counts = self.df.groupby('hour').size()
        fig.add_trace(go.Bar(x=hour_counts.index, y=hour_counts.values,
                            name='时段分布'),
                     row=2, col=2)
        
        fig.update_layout(height=1200, width=1600, 
                         title_text="爱情数据综合分析报告", 
                         showlegend=False)
        fig.write_html("love_report.html")
    
    #%% 辅助方法
    def get_metrics(self):
        """获取关键指标"""
        metrics = {}
        # 消息量对比
        metrics['msg_count'] = self.df.groupby('IsSender').size().to_dict()
        # 最活跃时段
        metrics['peak_hour'] = self.df.groupby('hour').size().idxmax()
        # 最长连续聊天
        return metrics

# #%% 生成 HTML 报告
# def generate_html_report(analyzer, output_path="love_report.html"):
#     """生成 HTML 报告"""
#     from jinja2 import Environment, FileSystemLoader
#     import os

#     # 获取当前目录
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     env = Environment(loader=FileSystemLoader(current_dir))

#     # 加载模板
#     template = env.get_template("report_template.html")

#     # 生成图表
#     ta_wordcloud = analyzer.generate_wordcloud(0)
#     my_wordcloud = analyzer.generate_wordcloud(1)
#     ta_heatmap = analyzer.plot_calendar_heatmap(0)
#     my_heatmap = analyzer.plot_calendar_heatmap(1)
#     response_time = analyzer.analyze_response_time()
#     love_report = analyzer.create_love_report()

#     # 渲染 HTML
#     html_output = template.render(
#         ta_wordcloud=ta_wordcloud,
#         my_wordcloud=my_wordcloud,
#         ta_heatmap=ta_heatmap.to_html(include_plotlyjs="cdn", full_html=False),
#         my_heatmap=my_heatmap.to_html(include_plotlyjs="cdn", full_html=False),
#         response_time=response_time.to_html(include_plotlyjs="cdn", full_html=False),
#         love_report=love_report.to_html(include_plotlyjs="cdn", full_html=False),
#         metrics=analyzer.get_metrics()
#     )

#     # 保存 HTML 文件
#     with open(output_path, "w", encoding="utf-8") as f:
#         f.write(html_output)

#%% 使用示例
if __name__ == "__main__":
    analyzer = WeChatLoveAnalyzer("data.csv")
    
    # 生成基础图表
    print("开始构建词云图...")
    analyzer.generate_wordcloud(0, "ta_wordcloud.png")
    analyzer.generate_wordcloud(1, "my_wordcloud.png")
    print("词云图全部完成")
    print("开始构建热力图...")
    analyzer.plot_calendar_heatmap(0, "ta_calendar.png")
    print("热力图完成")
    analyzer.analyze_response_time("response_time.png")
    
    # 生成综合报告
    print("生成综合报告中...")
    analyzer.create_love_report()
    
    # 打印关键指标
    print("关键指标分析:")
    metrics = analyzer.get_metrics()
    print(f"消息总量对比: TA发送{metrics['msg_count'][0]}条，你发送{metrics['msg_count'][1]}条")
    print(f"最常聊天时段: 晚上{metrics['peak_hour']}点")

    # generate_html_report(analyzer, "love_report.html")
    # print("HTML 报告已生成：love_report.html")

#%% 依赖安装说明
"""
需提前安装以下库：
pip install pandas jieba matplotlib seaborn wordcloud snownlp plotly july kaleido
pip install --upgrade "kaleido==0.1.*"
"""