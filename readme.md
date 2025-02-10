├── data/                # 数据目录
|   └── chat_records.csv
├── analysis/            # 分析模块
│   ├── __init__.py
│   └── analyzer.py      # 核心分析类
├── templates/           # HTML模板
│   ├── base.html        # 基础模板
│   └── components/      # 组件模板
├── static/              # 静态资源
│   ├── css/
│   │   ├── style.css          # 主样式文件
|   │   └── mobile.css         # 移动端专用样式（可选）
│   ├── js/
|   │   ├── script.js          # 交互逻辑（如手势支持）
|   │   └── plotly-loader.js   # Plotly动态加载逻辑
|   └── fonts/
|       └── simhei.ttf         # 中文字体文件（词云专用）
├── report_generator.py  # 报告生成模块
└── main.py              # 入口文件