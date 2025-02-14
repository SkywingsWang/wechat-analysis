import pandas as pd
import numpy as np
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
import plotly.subplots as sp
import streamlit as st
from datetime import timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import warnings
warnings.filterwarnings('ignore')

# 全局样式设置
plt.style.use('ggplot')
sns.set_palette(sns.color_palette(["#ff69b4", "#db7093"]))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置 Plotly 的模板
pio.templates.default = "plotly_white"

# 全局颜色设置
COLORS = {
    'your_color': '#ff69b4',  # 你的颜色
    'ta_color': '#db7093',    # TA的颜色
    'wordcloud_color': 'RdPu',  # 词云颜色
    'heatmap_color': 'reds',    # 热力图颜色
}

class WeChatAnalyzer:
    def __init__(self, df, stopwords_path="CNstopwords.txt", your_name="你", ta_name="TA"):
        self.df = self._preprocess_data(df)
        self.stopwords = self._load_stopwords(stopwords_path)
        self.your_name = your_name
        self.ta_name = ta_name
        
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
                      colormap=COLORS['wordcloud_color'],
                      width=1600, height=1200)
        wc.generate_from_frequencies(word_count)
        return wc, word_count.most_common(5)
    
    def plot_monthly_keywords(self):
        # 准备数据
        results = []
        for is_sender in [1, 0]:
            df = self.df[self.df['IsSender'] == is_sender]
            monthly_data = {}
            
            # 按月份处理数据
            for month, group in df.groupby(pd.Grouper(key='StrTime', freq='ME')):
                month_key = month.strftime("%Y-%m")
                texts = group['StrContent']
                pattern = re.compile(r"\[.+?\]")
                word_counter = Counter()
                
                for text in texts:
                    clean_text = pattern.sub('', str(text))
                    words = [word for word in jieba.lcut(clean_text) 
                            if word not in self.stopwords and len(word) > 1 and not re.search(r'[嘻哈呜]', word)]
                    word_counter.update(words)
                
                # 存储每个月的Top5
                monthly_data[month_key] = word_counter.most_common(5)
            
            # 将数据转换为统一格式
            all_months = sorted(monthly_data.keys())
            top_words = {i+1: [] for i in range(5)}  # 存储每个排名的数据
            for month in all_months:
                words = monthly_data[month]
                for i in range(5):
                    if i < len(words):
                        top_words[i+1].append((month, words[i][0], words[i][1]))
                    else:
                        top_words[i+1].append((month, None, 0))
            
            results.append((all_months, top_words))
        
        # 创建两个图表
        your_fig = self._create_single_chart(results[0], self.your_name, COLORS['your_color'])
        ta_fig = self._create_single_chart(results[1], self.ta_name, COLORS['ta_color'])
        
        return your_fig, ta_fig

    def _create_single_chart(self, data, name, color):
        months, top_words = data
        fig = go.Figure()
        
        # 添加数据
        for rank in range(1, 6):
            x_vals = []
            y_vals = []
            text_vals = []
            for month, word, count in top_words[rank]:
                x_vals.append(month)
                y_vals.append(count)
                text_vals.append(f"{word}: {count}" if word else "")
            
            fig.add_trace(go.Bar(
                x=x_vals,
                y=y_vals,
                name=f"Top{rank}",
                text=text_vals,
                marker_color=color,
                opacity=0.9 - (rank-1)*0.15,
                hoverinfo="x+y+text"
            ))
        
        # 更新布局
        fig.update_layout(
            title=f"{name} 的月度关键词分布",
            xaxis_title="月份",
            yaxis_title="出现次数",
            height=400,
            # legend=dict(
            #     orientation="h",
            #     yanchor="bottom",
            #     y=1.02,
            #     xanchor="right",
            #     x=1
            # ),
            margin=dict(t=80),
            hovermode="x unified"
        )
        
        # 优化x轴显示
        fig.update_xaxes(
            tickangle=45,
            tickmode='array',
            tickvals=months,
            ticktext=[m.split('-')[1] + '月' for m in months]  # 显示为"01月","02月"
        )
        
        return fig

    def plot_calendar_heatmap(self, is_sender):
        df = self.df[self.df['IsSender'] == is_sender]
        daily_counts = df.groupby('date').size().reset_index(name='count')
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])
        daily_counts['period'] = daily_counts['date'].dt.day.apply(
            lambda x: f"{(x-1)//4*4+1}-{((x-1)//4+1)*4}" if x < 29 else "29-31"
        )
        
        fig = px.density_heatmap(
            daily_counts,
            x='period',
            y=daily_counts['date'].dt.month_name(),
            z='count',
            histfunc="sum",
            color_continuous_scale=COLORS['heatmap_color']
        )
        fig.update_layout(
            yaxis_title="月份",
            xaxis_title="日期区间",
            height=600
        )
        max_day = daily_counts.loc[daily_counts['count'].idxmax()]
        label = self.ta_name if is_sender == 0 else self.your_name
        analysis = f"📅 {label}在{max_day['date'].strftime('%Y-%m')}月的{max_day['date'].day}日达到单日峰值（{max_day['count']}条）"
        return fig, analysis
    
    def analyze_response_time(self):
        # 创建一个按 'StrTime' 列排序的临时数据框
        temp_df = self.df.sort_values('StrTime')
        
        # 计算每个发送者之间的连续消息之间的时间差（分钟）
        temp_df['time_diff'] = temp_df.groupby('IsSender')['StrTime'].diff().dt.total_seconds() / 60
        
        # 过滤出发送者发生变化且时间差为正的行
        response_df = temp_df[(temp_df['IsSender'].diff() != 0) & (temp_df['time_diff'] > 0)]
        
        # 创建一个箱线图，显示每个发送者的响应时间分布
        fig = px.box(response_df, x='IsSender', y='time_diff', 
                    color='IsSender',
                    color_discrete_map={0: COLORS['ta_color'], 1: COLORS['your_color']},
                    title='消息响应时间分布分析')
        
        # 更新图形布局，设置自定义的 y 轴标题、x 轴标题、图例等
        fig.update_layout(
            yaxis_title='响应时间（分钟）',
            xaxis_title='发送者',
            showlegend=False,
            xaxis={'ticktext': [f'{self.ta_name}的回复速度', f'{self.your_name}的回复速度'], 'tickvals': [0, 1]}
        )
        
        # 计算每个发送者的响应时间中位数
        median_times = response_df.groupby('IsSender')['time_diff'].median()
        
        # 生成一个分析字符串，包含每个发送者的响应时间中位数
        analysis = f"⏱ {self.ta_name}的回复中位时间为{median_times[0]:.1f}分钟，{self.your_name}的回复中位时间为{median_times[1]:.1f}分钟"
        
        # 返回图形和分析字符串
        return fig, analysis

    
    def _get_max_consecutive_days(self):
        dates = self.df['date'].unique()
        dates.sort()

        if len(dates) <= 1:
            return len(dates)
    
        diffs = np.diff(dates)
        consecutive = []
        count = 1

        for d in diffs:
            if d == timedelta(days=1):
                count +=1
            else:
                consecutive.append(count)
                count = 1
        consecutive.append(count)
        return max(consecutive) if consecutive else 0
    def plot_daily_trend(self):
        daily_counts = self.df.groupby('date').size()
        fig = go.Figure(data=[go.Scatter(x=daily_counts.index, y=daily_counts.values,
                                        mode='lines', line=dict(width=2, color=COLORS['your_color']))])
        fig.update_layout(xaxis_title="日期", yaxis_title="消息量", height=400)
        max_day = daily_counts.idxmax().strftime('%Y-%m-%d')
        analysis = f"📈 你们的聊天高峰出现在{max_day}，当日共发送{daily_counts.max()}条消息"
        return fig, analysis

    def plot_sentiment_analysis(self):
        self.df['sentiment'] = self.df['StrContent'].apply(
            lambda x: SnowNLP(str(x)).sentiments if len(str(x)) > 2 else 0.5)
        
        weekly_sentiment = self.df.groupby(['IsSender', pd.Grouper(key='StrTime', freq='W')])['sentiment'].mean().reset_index()
        
        fig = go.Figure()
        for is_sender in [0, 1]:
            subset = weekly_sentiment[weekly_sentiment['IsSender'] == is_sender]
            fig.add_trace(go.Scatter(
                x=subset['StrTime'],
                y=subset['sentiment'],
                mode='lines+markers',
                name=self.ta_name if is_sender == 0 else self.your_name,
                line=dict(color=COLORS['ta_color'] if is_sender == 0 else COLORS['your_color']) 
            ))
        
        fig.update_layout(
            xaxis_title="日期",
            yaxis_title="情感值",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        avg_sentiment = weekly_sentiment.groupby('IsSender')['sentiment'].mean()
        analysis = f"😊 {self.ta_name}平均情感值{avg_sentiment[0]:.2f}，{self.your_name}的平均情感值{avg_sentiment[1]:.2f}"
        return fig, analysis

    def plot_hourly_distribution(self):
        hour_counts = self.df.groupby('hour').size()
        fig = go.Figure(data=[go.Bar(x=hour_counts.index, y=hour_counts.values,
                                    marker_color=COLORS['your_color'])])
        fig.update_layout(xaxis_title="小时", yaxis_title="消息量", height=400)
        peak_hour = hour_counts.idxmax()
        analysis = f"🌙 你们聊天中最活跃的时段为{peak_hour}点，共发送{hour_counts.max()}条消息"
        return fig, analysis

    def get_metrics(self):
        metrics = {}
        metrics['msg_count'] = self.df.groupby('IsSender').size().to_dict()
        metrics['peak_hour'] = self.df.groupby('hour').size().idxmax()
        metrics['max_consecutive_days'] = self._get_max_consecutive_days()
        
        # 最常聊天天数（消息量>50的天数）
        daily_counts = self.df.groupby('date').size()
        metrics['active_days'] = len(daily_counts[daily_counts > 50])
        return metrics

    def analyze_joint_topics(self, n_topics=2, top_words=5):
        """使用NMF进行联合主题分析"""
        monthly_data = []

        extra_stopwords = set([self.your_name, self.ta_name])  # 明确过滤双方昵称
        invalid_chars = {'嘻', '哈', '呜', '啦', '哦'}  # 新增无效字符黑名单

        # 按月份合并所有对话
        for month, group in self.df.groupby(pd.Grouper(key='StrTime', freq='ME')):
            # 合并当月所有消息
            combined_text = ' '.join(group['StrContent'].astype(str))
            # 清洗文本
            combined_text = re.sub(r"\[.+?\]", "", combined_text)
            words = [word for word in jieba.lcut(combined_text) 
                if (word not in self.stopwords) and 
                (len(word) > 1) and 
                (word not in extra_stopwords) and 
                (not any(char in invalid_chars for char in word))]
            monthly_data.append({
                'month': month.strftime("%Y-%m"),
                'text': ' '.join(words),
                'msg_count': len(group)
            })
        
        # 创建TF-IDF矩阵
        tfidf = TfidfVectorizer(
            max_features=1000,
            token_pattern=r'(?u)\b\w{2,}\b',  # 仅保留2字以上词语
            ngram_range=(1,2)  # 允许二元词组
        )
        dtm = tfidf.fit_transform([d['text'] for d in monthly_data])
        
        # 训练NMF模型
        nmf = NMF(
            n_components=n_topics,
            random_state=42,
            beta_loss='kullback-leibler',  # 更适合短文本
            solver='mu',
            max_iter=1000
        )
        nmf.fit(dtm)
        
        # 提取特征词
        feature_names = tfidf.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(nmf.components_):
            top_features = [feature_names[i] for i in topic.argsort()[:-top_words-1:-1]]
            topics.append({
                'topic_id': topic_idx+1,
                'keywords': '、'.join(top_features)
            })
        
        # 分配主题到月份
        topic_results = []
        for i, month_data in enumerate(monthly_data):
            weights = nmf.transform(dtm[i])[0]
            main_topic = weights.argmax() + 1
            topic_results.append({
                'month': month_data['month'],
                'msg_count': month_data['msg_count'],
                'main_topic': main_topic,
                'keywords': topics[main_topic-1]['keywords'],
                'weight': weights.max()
            })
        
        return pd.DataFrame(topic_results)
    
    def get_longest_chat_duration(self):
        """计算最长的一次聊天持续时间"""
        temp_df = self.df.sort_values('StrTime')
        temp_df['time_diff'] = temp_df['StrTime'].diff().dt.total_seconds() / 60
        temp_df['chat_id'] = (temp_df['time_diff'] > 10).cumsum()  # 设超过10分钟的间隔为新聊天
        
        chat_durations = temp_df.groupby('chat_id')['StrTime'].agg(['min', 'max'])
        chat_durations['duration'] = (chat_durations['max'] - chat_durations['min']).dt.total_seconds() / 60
        
        longest_chat = chat_durations.loc[chat_durations['duration'].idxmax()]
        return longest_chat['duration'], longest_chat['min'], longest_chat['max']

# Streamlit App
st.title("💬 WeChat 聊天记录分析")

# 自定义CSS样式
st.markdown("""
<style>
.subheader {
        margin-bottom: 4rem; 
}
.analysis-text {
    font-size: 16px !important;
    color: #DB7093 !important;
}
</style>
""", unsafe_allow_html=True)

# 初始化 session_state
if 'hide_names' not in st.session_state:
    st.session_state.hide_names = False
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# 昵称输入框
if not st.session_state.hide_names:
    col_name1, col_name2 = st.columns(2)
    with col_name1:
        your_name = st.text_input("TA如何称呼你", value="你")
    with col_name2:
        ta_name = st.text_input("你如何称呼TA", value="TA")

    # 添加一个按钮来隐藏输入框
    if st.button("确认"):
        st.session_state.hide_names = True
        st.session_state.your_name = your_name
        st.session_state.ta_name = ta_name
        st.rerun()
else:
    your_name = st.session_state.your_name
    ta_name = st.session_state.ta_name

# 文件上传框
uploaded_file = st.file_uploader("请上传微信聊天记录CSV文件", type="csv", help="需要首先从[memotrace](https://memotrace.cn/doc/posts/deploy/install.html)获取csv文件")

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    analyzer = WeChatAnalyzer(df, your_name=your_name, ta_name=ta_name)
    
    # 关键指标
    st.subheader("👀 核心数据概览")
    metrics = analyzer.get_metrics()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label=f"{ta_name}的消息量", value=metrics['msg_count'].get(0, 0))
    with col2:
        st.metric(label=f"{your_name}的消息量", value=metrics['msg_count'].get(1, 0))
    with col3:
        st.metric(label="最长连续天数", value=metrics['max_consecutive_days'])
    with col4:
        st.metric(label="火热聊天天数", value=metrics['active_days'], 
                help="单日消息量超过50条的天数")

    # 最长聊天持续时间
    longest_duration, start_time, end_time = analyzer.get_longest_chat_duration()
    st.markdown(f"**你们最长的一次聊天持续了 {longest_duration:.1f} 分钟**，从 {start_time.strftime('%Y-%m-%d %H:%M')} 到 {end_time.strftime('%Y-%m-%d %H:%M')}.")
    st.markdown("是在谈论什么话题呢？")
    
    # 关键词分析
    st.subheader("☁️ 词云")
    col1, col2 = st.columns(2)
    with col1:
        wc_ta, top_words_ta = analyzer.generate_wordcloud(0)
        plt.figure(figsize=(8, 5))
        plt.imshow(wc_ta, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        st.markdown(f"<div class='analysis-text'>{ta_name}的高频词：{'、'.join([w[0] for w in top_words_ta])}</div>", unsafe_allow_html=True)
    
    with col2:
        wc_my, top_words_my = analyzer.generate_wordcloud(1)
        plt.figure(figsize=(8, 5))
        plt.imshow(wc_my, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        st.markdown(f"<div class='analysis-text'>{your_name}的高频词：{'、'.join([w[0] for w in top_words_my])}</div>", unsafe_allow_html=True)

    def get_topic_data(_analyzer):
        return _analyzer.analyze_joint_topics()

    topic_df = get_topic_data(analyzer)

    # # 创建可视化
    # fig = px.scatter(
    #     topic_df,
    #     x='month',
    #     y='weight',
    #     size='msg_count',
    #     color='main_topic',
    #     hover_name='keywords',
    #     color_continuous_scale=COLORS['heatmap_color'],
    #     size_max=20,
    #     labels={'month':'月份', 'weight':'主题强度', 'msg_count':'消息量'}
    # )

    # fig.update_layout(
    #     height=500,
    #     xaxis=dict(tickangle=45),
    #     hoverlabel=dict(bgcolor="white")
    # )

    # st.plotly_chart(fig, use_container_width=True)

    st.markdown("  ")

    # 添加详细解读
    st.subheader("🧩 月度主题分析")
    for idx, row in topic_df.iterrows():
        st.markdown(f"""
        **{row['month']}月**  
        🔖 主要话题：{row['keywords']}  
        📊 消息量：{row['msg_count']}条 | 主题强度：{row['weight']:.2f}
        """)
        st.progress(row['weight'])

    # 月份高频词
    # st.markdown("### 📊 月度关键词")

    # 你的关键词分布
    your_fig, _ = analyzer.plot_monthly_keywords()
    st.plotly_chart(your_fig, use_container_width=True)

    # TA的关键词分布
    _, ta_fig = analyzer.plot_monthly_keywords()
    st.plotly_chart(ta_fig, use_container_width=True)

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