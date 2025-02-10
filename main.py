from analysis.analyzer import WeChatLoveAnalyzer

if __name__ == "__main__":
    # 初始化分析器
    analyzer = WeChatLoveAnalyzer("data/chat_records.csv")
    
    # 获取指标数据
    metrics = analyzer.get_metrics()
    print(f"消息统计: TA发送{metrics['msg_count'][0]}条，你发送{metrics['msg_count'][1]}条")
    
    # 生成词云图（返回Base64）
    ta_wordcloud = analyzer.generate_wordcloud(0)
    my_wordcloud = analyzer.generate_wordcloud(1)
    
    # 生成综合报告HTML片段
    report_html = analyzer.create_love_report(save_path="output.html")
    print("报告已输出")