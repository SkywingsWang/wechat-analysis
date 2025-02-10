// 增强移动端交互
document.addEventListener('DOMContentLoaded', () => {
    // 自动调整图表尺寸
    const resizeCharts = () => {
        document.querySelectorAll('.js-plotly-plot').forEach(chart => {
            Plotly.relayout(chart, {
                'width': chart.offsetWidth,
                'height': chart.offsetWidth * 0.75
            });
        });
    };

    // 初始化调整
    resizeCharts();
    
    // 监听屏幕旋转
    window.addEventListener('orientationchange', resizeCharts);
    window.addEventListener('resize', resizeCharts);

    // 点击图表全屏
    document.querySelectorAll('.chart-container').forEach(container => {
        container.addEventListener('click', () => {
            container.classList.toggle('fullscreen');
        });
    });
});