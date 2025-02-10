# report_generator.py
import base64
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from tempfile import TemporaryDirectory

class ReportGenerator:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.env = Environment(loader=FileSystemLoader("templates"))
    
    def _generate_charts(self, tmpdir):
        """生成所有图表并返回上下文数据"""
        # 词云图
        ta_wc = self.analyzer.generate_wordcloud(0, f"{tmpdir}/ta_wc.png")
        my_wc = self.analyzer.generate_wordcloud(1, f"{tmpdir}/my_wc.png")
        
        # 热力图
        ta_heatmap = self.analyzer.plot_calendar_heatmap(0, f"{tmpdir}/ta_heatmap.png")
        my_heatmap = self.analyzer.plot_calendar_heatmap(1, f"{tmpdir}/my_heatmap.png")
        
        # 响应时间
        response_time = self.analyzer.analyze_response_time(f"{tmpdir}/response_time.png")
        
        # 综合报告
        love_report = self.analyzer.create_love_report(f"{tmpdir}/love_report.html")
        
        return {
            "ta_wordcloud": self._img_to_base64(f"{tmpdir}/ta_wc.png"),
            "my_wordcloud": self._img_to_base64(f"{tmpdir}/my_wc.png"),
            "ta_heatmap": self._img_to_base64(f"{tmpdir}/ta_heatmap.png"),
            "my_heatmap": self._img_to_base64(f"{tmpdir}/my_heatmap.png"),
            "response_time": self._img_to_base64(f"{tmpdir}/response_time.png"),
            "love_report": Path(f"{tmpdir}/love_report.html").read_text(encoding="utf-8")
        }

    def _img_to_base64(self, path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def generate_report(self, output_path="report.html"):
        with TemporaryDirectory() as tmpdir:
            context = {
                "metrics": self.analyzer.get_metrics(),
                **self._generate_charts(tmpdir)
            }
            
            template = self.env.get_template("base.html")
            html = template.render(context)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)
        return output_path