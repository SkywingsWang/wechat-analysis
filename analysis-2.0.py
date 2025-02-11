import pandas as pd
import jieba
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from snownlp import SnowNLP
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# 全局样式设置
plt.style.use('ggplot')
sns.set_palette(sns.color_palette(["#ff69b4", "#db7093"]))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置 Plotly 的模板
pio.templates.default = "plotly_white"

class WeChatAnalyzer:
    def __init__(self, df, stopwords_path="CNstopwords.txt"):
        self.df = self._preprocess_data(df)
        self.stopwords = self._load_stopwords(stopwords_path)
        
    def _preprocess_data(self, df):
        df = df[df['Type'] == 1]
        df = df.dropna(subset=['StrContent'])
        df = df[~df['StrContent'].str.contains(r'https://www\.xiaohongshu\.com', na=False)]
        
        df['StrTime'] = pd.to_datetime(df['StrTime'])
        df['date'] = df['StrTime'].dt.date
        df['hour'] = df['StrTime'].dt.hour
        df['weekday'] = df['StrTime'].dt.weekday
        df['month'] = df['StrTime'].dt.month
        return df
    
    def _load_stopwords(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return set([line.strip() for line in f])
    
    def generate_wordcloud(self, is_sender):
        texts = self.df[self.df['IsSender'] == is_sender]['StrContent']
        pattern = re.compile(r"\[.+?\]")
        
        words = []
        for text in texts:
            clean_text = pattern.sub('', str(text))
            words += [word for word in jieba.lcut(clean_text) 
                     if word not in self.stopwords and len(word) > 1]
        
        word_count = Counter(words)
        wc = WordCloud(font_path="simhei.ttf", 
                      background_color='white',
                      colormap='RdPu',
                      width=1600, height=1200)
        wc.generate_from_frequencies(word_count)
        return wc, word_count.most_common(5)
    
    def plot_calendar_heatmap(self, is_sender):
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
            height=600
        )
        max_day = daily_counts.loc[daily_counts['count'].idxmax()]
        
        # Determine the label based on is_sender
        label = "TA" if is_sender == 0 else "你"
        analysis = f"📅 {label}在{max_day['date'].strftime('%Y-%m')}月的{max_day['date'].day}日达到单日峰值（{max_day['count']}条）"
        return fig, analysis
    
    def analyze_response_time(self):
        temp_df = self.df.sort_values('StrTime')
        temp_df['time_diff'] = temp_df.groupby('IsSender')['StrTime'].diff().dt.total_seconds() / 60
        response_df = temp_df[(temp_df['IsSender'].diff() != 0) & (temp_df['time_diff'] > 0)]
        
        fig = px.box(response_df, x='IsSender', y='time_diff', 
                     color='IsSender',
                     color_discrete_map={0: "#ff69b4", 1: "#db7093"},
                     title='消息响应时间分布分析')
        fig.update_layout(
            yaxis_title='响应时间（分钟）',
            xaxis_title='发送者',
            showlegend=False,
            xaxis={'ticktext': ['TA的回复速度', '你的回复速度'], 'tickvals': [0, 1]}
        )
        median_times = response_df.groupby('IsSender')['time_diff'].median()
        analysis = f"⏱ TA的回复中位时间为{median_times[0]:.1f}分钟，你的回复中位时间为{median_times[1]:.1f}分钟"
        return fig, analysis
    
    def plot_daily_trend(self):
        daily_counts = self.df.groupby('date').size()
        fig = go.Figure(data=[go.Scatter(x=daily_counts.index, y=daily_counts.values,
                                        mode='lines', line=dict(width=2, color='#ff69b4'))])
        fig.update_layout(xaxis_title="日期", yaxis_title="消息量", height=400)
        max_day = daily_counts.idxmax().strftime('%Y-%m-%d')
        analysis = f"📈 你们的聊天高峰出现在{max_day}，当日共发送{daily_counts.max()}条消息"
        return fig, analysis

    def plot_sentiment_analysis(self):
        self.df['sentiment'] = self.df['StrContent'].apply(
            lambda x: SnowNLP(str(x)).sentiments if len(str(x)) > 2 else 0.5)
        weekly_sentiment = self.df.resample('W', on='StrTime')['sentiment'].mean()
        fig = go.Figure(data=[go.Scatter(x=weekly_sentiment.index, y=weekly_sentiment.values,
                                        mode='lines+markers', line=dict(color='#db7093'))])
        fig.update_layout(xaxis_title="日期", yaxis_title="情感值", height=400)
        avg_sentiment = weekly_sentiment.mean()
        analysis = f"😊 你们聊天中的平均情感值为{avg_sentiment:.2f}（1为积极，0为消极），近期趋势：{'上升' if weekly_sentiment.iloc[-1] > weekly_sentiment.iloc[-2] else '下降'}"
        return fig, analysis

    def plot_hourly_distribution(self):
        hour_counts = self.df.groupby('hour').size()
        fig = go.Figure(data=[go.Bar(x=hour_counts.index, y=hour_counts.values,
                                    marker_color='#ff69b4')])
        fig.update_layout(xaxis_title="小时", yaxis_title="消息量", height=400)
        peak_hour = hour_counts.idxmax()
        analysis = f"🌙 你们聊天中最活跃的时段为{peak_hour}点，共发送{hour_counts.max()}条消息"
        return fig, analysis

    def get_metrics(self):
        metrics = {}
        metrics['msg_count'] = self.df.groupby('IsSender').size().to_dict()
        metrics['peak_hour'] = self.df.groupby('hour').size().idxmax()
        
        # 最常聊天天数（消息量>50的天数）
        daily_counts = self.df.groupby('date').size()
        metrics['active_days'] = len(daily_counts[daily_counts > 50])
        return metrics


# Streamlit App
st.title("💬 WeChat 聊天记录分析")

# File upload
uploaded_file = st.file_uploader("请上传微信聊天记录CSV文件", type="csv", help="需要首先从[memotrace](https://memotrace.cn/doc/posts/deploy/install.html)获取csv文件")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    analyzer = WeChatAnalyzer(df)
    
    # 关键指标
    st.subheader("📊 核心数据概览")
    metrics = analyzer.get_metrics()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="TA的消息量", value=metrics['msg_count'].get(0, 0))
    with col2:
        st.metric(label="你的消息量", value=metrics['msg_count'].get(1, 0))
    with col3:
        st.metric(label="火热聊天天数", value=metrics['active_days'], 
                help="单日消息量超过50条的天数")

    # 词云图
    st.subheader("🔤 关键词分析")
    col1, col2 = st.columns(2)
    with col1:
        wc_ta, top_words_ta = analyzer.generate_wordcloud(0)
        plt.figure(figsize=(8, 5))
        plt.imshow(wc_ta, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        st.caption(f"TA的高频词：{'、'.join([w[0] for w in top_words_ta])}")
    
    with col2:
        wc_my, top_words_my = analyzer.generate_wordcloud(1)
        plt.figure(figsize=(8, 5))
        plt.imshow(wc_my, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        st.caption(f"你的高频词：{'、'.join([w[0] for w in top_words_my])}")

    # 每日趋势
    st.subheader("📅 聊天趋势分析")
    daily_trend, trend_analysis = analyzer.plot_daily_trend()
    st.plotly_chart(daily_trend)
    st.markdown(f"`{trend_analysis}`")

    # 日历热力图
    st.subheader("🗓️ 活跃度分布")
    col3, col4 = st.columns(2)
    with col3:
        heatmap_ta, analysis_ta = analyzer.plot_calendar_heatmap(0)
        st.plotly_chart(heatmap_ta)
        st.markdown(f"`{analysis_ta}`")
    
    with col4:
        heatmap_my, analysis_my = analyzer.plot_calendar_heatmap(1)
        st.plotly_chart(heatmap_my)
        st.markdown(f"`{analysis_my}`")

    # 响应时间分析
    st.subheader("⏳ 响应时间分析")
    response_time_fig, response_analysis = analyzer.analyze_response_time()
    st.plotly_chart(response_time_fig)
    st.markdown(f"`{response_analysis}`")

    # 情感分析
    st.subheader("😊 情感分析")
    sentiment_fig, sentiment_analysis = analyzer.plot_sentiment_analysis()
    st.plotly_chart(sentiment_fig)
    st.markdown(f"`{sentiment_analysis}`")

    # 时段分析
    st.subheader("⏰ 聊天时段分布")
    hourly_fig, hourly_analysis = analyzer.plot_hourly_distribution()
    st.plotly_chart(hourly_fig)
    st.markdown(f"`{hourly_analysis}`")