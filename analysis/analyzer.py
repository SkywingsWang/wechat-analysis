# analysis/analyzer.py
import pandas as pd
import numpy as np
import jieba
import re
import base64
import logging
from datetime import datetime
from typing import Optional, Dict, Tuple
from pathlib import Path
from io import BytesIO
from collections import Counter
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from snownlp import SnowNLP
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChartConfig:
    """图表样式配置类"""
    WORDCLOUD_SIZE = (1600, 1200)  # 词云尺寸 (width, height)
    WORDCLOUD_FONT_PATH = "static/fonts/simhei.ttf"  # 中文字体路径
    HEATMAP_COLORSCALE = 'reds'  # 热力图颜色方案
    PLOTLY_TEMPLATE = "plotly_dark"  # Plotly模板
    RESPONSE_TIME_PALETTE = ["#ff69b4", "#db7093"]  # 响应时间箱线图配色
    SENTIMENT_WINDOW = "W"  # 情感分析时间窗口（周）

class WeChatLoveAnalyzer:
    """微信聊天记录分析核心类"""
    
    def __init__(self, data_path: str, config: ChartConfig = ChartConfig()):
        """
        初始化分析器
        :param data_path: 聊天记录CSV文件路径
        :param config: 图表配置对象
        """
        self.config = config
        self.df = self._preprocess_data(data_path)
        self.stopwords = self._load_stopwords("CNstopwords.txt")
        self._setup_visualization()

    def _setup_visualization(self) -> None:
        """初始化可视化配置"""
        plt.style.use('ggplot')
        sns.set_palette(sns.color_palette(self.config.RESPONSE_TIME_PALETTE))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

    def _preprocess_data(self, data_path: str) -> pd.DataFrame:
        """数据预处理管道"""
        try:
            df = pd.read_csv(data_path, parse_dates=['StrTime'])
            df = df[df['Type'] == 1]  # 仅分析文本消息
            df = df.dropna(subset=['StrContent'])  # 移除空内容
            
            # 过滤小红书链接
            df = df[~df['StrContent'].str.contains(r'https://www\.xiaohongshu\.com', na=False)]
            
            # 时间特征提取
            df['date'] = df['StrTime'].dt.date
            df['hour'] = df['StrTime'].dt.hour
            df['weekday'] = df['StrTime'].dt.weekday
            df['month'] = df['StrTime'].dt.month
            return df
        except FileNotFoundError:
            logger.error(f"数据文件未找到: {data_path}")
            raise
        except KeyError as e:
            logger.error(f"数据列缺失: {str(e)}")
            raise

    @staticmethod
    def _load_stopwords(path: str) -> set:
        """加载停用词表"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return set(line.strip() for line in f)
        except FileNotFoundError:
            logger.warning(f"停用词表未找到: {path}, 使用空集合")
            return set()

    def generate_wordcloud(self, is_sender: int, save_path: Optional[str] = None) -> Optional[str]:
        """
        生成词云图
        :param is_sender: 发送者标识 (0-对方, 1-自己)
        :param save_path: 保存路径（可选）
        :return: 若未提供保存路径，返回Base64编码图片
        """
        texts = self.df[self.df['IsSender'] == is_sender]['StrContent']
        pattern = re.compile(r"\[.+?\]")  # 过滤表情符号
        
        words = []
        for text in texts:
            clean_text = pattern.sub('', str(text))
            words += [word for word in jieba.lcut(clean_text) 
                     if word not in self.stopwords and len(word) > 1]
        
        word_count = Counter(words)
        wc = WordCloud(
            font_path=self.config.WORDCLOUD_FONT_PATH,
            width=self.config.WORDCLOUD_SIZE[0],
            height=self.config.WORDCLOUD_SIZE[1],
            background_color='white',
            colormap='RdPu'
        )
        wc.generate_from_frequencies(word_count)
        
        plt.figure(figsize=(20, 15))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            return None
        else:
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            plt.close()
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')

    def plot_calendar_heatmap(self, is_sender: int, save_path: Optional[str] = None) -> Optional[str]:
        """
        生成交互式日历热力图
        :param is_sender: 发送者标识
        :param save_path: 保存路径（可选）
        :return: 若未提供保存路径，返回Base64编码图片
        """
        df = self.df[self.df['IsSender'] == is_sender]
        daily_counts = df.groupby('date').size().reset_index(name='count')
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])
        
        fig = px.density_heatmap(
            daily_counts,
            x=daily_counts['date'].dt.day,
            y=daily_counts['date'].dt.month_name(),
            z='count',
            histfunc="sum",
            color_continuous_scale=self.config.HEATMAP_COLORSCALE
        )
        fig.update_layout(
            yaxis_title="月份",
            xaxis_title="日期",
            height=800,
            template=self.config.PLOTLY_TEMPLATE
        )
        
        if save_path:
            fig.write_image(save_path)
            return None
        else:
            buffer = BytesIO()
            fig.write_image(buffer, format='png')
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')

    def analyze_response_time(self, save_path: Optional[str] = None) -> Optional[str]:
        """
        分析响应时间
        :param save_path: 保存路径（可选）
        :return: 若未提供保存路径，返回Base64编码图片
        """
        temp_df = self.df.sort_values('StrTime')
        temp_df['time_diff'] = temp_df.groupby('IsSender')['StrTime'].diff()
        response_df = temp_df[temp_df['IsSender'].diff() != 0]
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            x='IsSender', 
            y='time_diff', 
            data=response_df, 
            showfliers=False, 
            width=0.4
        )
        plt.ylabel('响应时间间隔', fontsize=12)
        plt.xticks([0, 1], ['TA的回复速度', '你的回复速度'])
        plt.title('消息响应时间分布分析', fontsize=14)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            return None
        else:
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')

    def create_love_report(self, save_path: Optional[str] = None) -> Optional[str]:
        """
        生成综合报告图表
        :param save_path: 保存路径（可选）
        :return: 若未提供保存路径，返回Plotly HTML div字符串
        """
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'xy'}, {'type': 'polar'}],
                   [{'type': 'xy'}, {'type': 'xy'}]],
            subplot_titles=("每日聊天趋势", "消息类型分布", 
                           "情感波动曲线", "时段分析")
        )
        
        # 每日趋势
        daily_counts = self.df.groupby('date').size()
        fig.add_trace(
            go.Scatter(
                x=daily_counts.index, 
                y=daily_counts.values,
                mode='lines', 
                name='每日消息量'
            ),
            row=1, col=1
        )
        
        # 消息类型雷达图
        type_counts = self.df.groupby(['IsSender', 'Type']).size().unstack()
        fig.add_trace(
            go.Scatterpolar(
                r=type_counts.loc[0].values,
                theta=['文字','图片','语音','视频','其他'],
                fill='toself',
                name='TA'
            ), 
            row=1, col=2
        )
        fig.add_trace(
            go.Scatterpolar(
                r=type_counts.loc[1].values,
                theta=['文字','图片','语音','视频','其他'],
                fill='toself',
                name='你'
            ), 
            row=1, col=2
        )
        
        # 情感分析
        try:
            self.df['sentiment'] = self.df['StrContent'].apply(
                lambda x: SnowNLP(str(x)).sentiments
            )
            weekly_sentiment = self.df.resample(
                self.config.SENTIMENT_WINDOW, 
                on="StrTime"
            )["sentiment"].mean()
            fig.add_trace(
                go.Scatter(
                    x=weekly_sentiment.index, 
                    y=weekly_sentiment.values,
                    mode='lines+markers', 
                    name='情感值'
                ),
                row=2, col=1
            )
        except Exception as e:
            logger.error(f"情感分析失败: {str(e)}")
        
        # 时段分析
        hour_counts = self.df.groupby('hour').size()
        fig.add_trace(
            go.Bar(
                x=hour_counts.index, 
                y=hour_counts.values,
                name='时段分布'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=1200, 
            width=1600, 
            title_text="爱情数据综合分析报告",
            template=self.config.PLOTLY_TEMPLATE,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            return None
        else:
            return fig.to_html(
                full_html=False,
                include_plotlyjs='cdn',
                div_id='love-report'
            )

    def get_metrics(self) -> Dict:
        """获取关键指标"""
        metrics = {}
        # 消息量对比
        msg_counts = self.df.groupby('IsSender').size()
        metrics['msg_count'] = {
            0: int(msg_counts.get(0, 0)),
            1: int(msg_counts.get(1, 0))
        }
        # 最活跃时段
        try:
            metrics['peak_hour'] = int(self.df.groupby('hour').size().idxmax())
        except ValueError:
            metrics['peak_hour'] = "无数据"
        return metrics