import matplotlib.pyplot as plt
import json

def plot_json_data(json_data):
    """
    将JSON整数列表绘制为折线图
    
    参数:
    json_data: 包含整数的列表
    
    返回:
    折线图的图形对象
    """
    # 确保数据是列表且所有元素都是整数
    if not isinstance(json_data, list):
        raise TypeError("输入数据必须是列表")
    
    for i, item in enumerate(json_data):
        if not isinstance(item, int):
            raise TypeError(f"列表元素 {i} 不是整数: {item}")
    
    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制折线图
    ax.plot(json_data, marker='o', linestyle='-')
    
    # 设置图表标题和坐标轴标签
    ax.set_title('')
    ax.set_xlabel('Try-Time')
    ax.set_ylabel('Path-Length')
    
    # 添加网格线以便于查看
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 设置x轴刻度为整数索引
    ax.set_xticks(range(len(json_data)))
    
    return fig

# 示例使用
if __name__ == "__main__":
    # 示例JSON数据
    with open("path_len.json", 'r') as f:
        json_data = json.load(f)
    
    # 生成并显示图表
    fig = plot_json_data(json_data)
    plt.tight_layout()  # 确保布局合理
    plt.show()
    
    # 如果需要保存图表，可以取消下面这行的注释
    # fig.savefig('json_line_chart.png', dpi=300)