# -*- coding: utf-8 -*-
import os
import json
import gc
import glob
import logging
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import psutil

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from matplotlib import font_manager, rcParams

# 设置中文字体
font_path = "/usr/share/fonts/SourceHanSansSC-Regular.otf"
font_prop = font_manager.FontProperties(fname=font_path)
rcParams['font.family'] = font_prop.get_name()
rcParams['axes.unicode_minus'] = False

# 日志记录和内存检查
def setup_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.info

def check_memory_limit(limit_gb=400):
    mem = psutil.virtual_memory()
    used_gb = (mem.total - mem.available) / (1024**3)
    log(f"当前内存使用：{used_gb:.2f} GB")
    return used_gb < limit_gb

# 数据加载
def load_data(parquet_dir, catalog_path):
    all_files = glob.glob(os.path.join(parquet_dir, "*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in all_files], ignore_index=True)
    with open(catalog_path, "r", encoding="utf-8") as f:
        product_info = pd.DataFrame(json.load(f)["products"])
    id_to_category = product_info.set_index("id")["category"].to_dict()
    id_to_price = product_info.set_index("id")["price"].to_dict()
    return df, id_to_category, id_to_price

# 解析订单信息
def parse_orders(df, id_to_category, id_to_price):
    def extract(row):
        try:
            purchase = json.loads(row) if isinstance(row, str) else row
            items = purchase.get("items", [])
            return pd.Series({
                "order_item_ids": [i["id"] for i in items],
                "order_categories": [id_to_category.get(i["id"], "未知") for i in items],
                "order_prices": [id_to_price.get(i["id"], 0.0) for i in items],
                "order_date": purchase.get("purchase_date"),
                "payment_method": purchase.get("payment_method"),
                "payment_status": purchase.get("payment_status"),
                "avg_price": purchase.get("avg_price", 0.0),
            })
        except:
            return pd.Series({
                "order_item_ids": [], "order_categories": [],
                "order_prices": [], "order_date": None,
                "payment_method": None, "payment_status": None,
                "avg_price": 0.0
            })
    parsed = df["purchase_history"].apply(extract)
    df_orders = pd.concat([df, parsed], axis=1)
    return df_orders

# 类别映射和购物篮构造
def build_baskets(df_orders, category_map):
    sub_to_big = {sub: big for big, subs in category_map.items() for sub in subs}
    baskets = []
    for cats in df_orders["order_categories"]:
        big_cats = list(set(sub_to_big.get(cat, None) for cat in cats if sub_to_big.get(cat)))
        if big_cats:
            baskets.append(big_cats)
    return baskets

# Apriori 分析并返回规则
def mine_rules(baskets, min_support=0.01, min_confidence=0.5):
    te = TransactionEncoder()
    te_ary = te.fit(baskets).transform(baskets)
    df_te = pd.DataFrame(te_ary, columns=te.columns_)
    freq_items = apriori(df_te, min_support=min_support, use_colnames=True)
    rules = association_rules(freq_items, metric="confidence", min_threshold=min_confidence)
    return rules

# 可视化关联规则网络图
def draw_network(rules, title, filename, output_dir):
    G = nx.DiGraph()
    for _, row in rules.iterrows():
        for a in row['antecedents']:
            for c in row['consequents']:
                G.add_edge(a, c, weight=row['confidence'])

    pos = nx.spring_layout(G, k=1)
    plt.figure(figsize=(12, 8))
    weights = [d['weight'] for (_, _, d) in G.edges(data=True)]
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', width=weights, arrows=True,
            font_family=font_prop.get_name())
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)},
                                 font_color='red', font_family=font_prop.get_name())
    plt.title(title, fontproperties=font_prop)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

# 高价商品支付方式分布
def plot_high_value_payment(df_orders, output_dir):
    payments = [row['payment_method'] for _, row in df_orders.iterrows()
                if any(p > 5000 for p in row['order_prices'])]
    df = pd.DataFrame(payments, columns=["payment_method"])
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="payment_method", order=df["payment_method"].value_counts().index)
    plt.title("高价值商品支付方式分布", fontproperties=font_prop)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "high_value_payment_distribution.png"), dpi=300)
    plt.close()

# 月度订单数 & 类别热力图
def time_based_analysis(df_orders, output_dir):
    df_orders['order_date'] = pd.to_datetime(df_orders['order_date'], errors='coerce')
    df_orders['month'] = df_orders['order_date'].dt.month

    monthly_counts = df_orders.groupby('month').size()
    monthly_counts.plot(kind='bar')
    plt.title('月度订单数量', fontproperties=font_prop)
    plt.xlabel('月份')
    plt.ylabel('订单数')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "monthly_order_counts.png"), dpi=300)
    plt.close()

    # 热力图
    heat = defaultdict(int)
    for _, row in df_orders.iterrows():
        month = row['month']
        for cat in row['order_categories']:
            heat[(cat, month)] += 1
    df_heat = pd.DataFrame([{"category": k[0], "month": k[1], "count": v} for k, v in heat.items()])
    pivot = df_heat.pivot(index="category", columns="month", values="count").fillna(0)

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".0f")
    plt.title("商品类别在不同月份的购买频率", fontproperties=font_prop)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "monthly_category_heatmap.png"), dpi=300)
    plt.close()

# 主执行函数
def main():
    global log
    # 配置路径
    data_dir = "/home/xh/Data_mining"
    parquet_dir = os.path.join(data_dir, "30G_data_new")
    catalog_path = os.path.join(data_dir, "product_catalog.json")
    project_name = "1e0G_data_results"
    output_dir = os.path.join(project_name, "figures")
    log_path = f"logs/{project_name}.log"

    os.makedirs(output_dir, exist_ok=True)
    log = setup_logging(log_path)

    df, id_to_category, id_to_price = load_data(parquet_dir, catalog_path)
    if not check_memory_limit():
        log("❌ 内存过高，终止")
        return

    df_orders = parse_orders(df, id_to_category, id_to_price)
    del df; gc.collect()

    category_map = {
        '电子产品': ['智能手机', '笔记本电脑', '平板电脑', '智能手表', '耳机', '音响', '相机', '摄像机', '游戏机'],
        '服装': ['上衣', '裤子', '裙子', '内衣', '鞋子', '帽子', '手套', '围巾', '外套'],
        '食品': ['零食', '饮料', '调味品', '米面', '水产', '肉类', '蛋奶', '水果', '蔬菜'],
        '家居': ['家具', '床上用品', '厨具', '卫浴用品'],
        '办公': ['文具', '办公用品'],
        '运动户外': ['健身器材', '户外装备'],
        '玩具': ['玩具', '模型', '益智玩具'],
        '母婴': ['婴儿用品', '儿童课外读物'],
        '汽车用品': ['车载电子', '汽车装饰']
    }

    baskets = build_baskets(df_orders, category_map)
    rules = mine_rules(baskets)
    elec_rules = rules[rules['antecedents'].apply(lambda x: '电子产品' in x) |
                       rules['consequents'].apply(lambda x: '电子产品' in x)]
    draw_network(elec_rules, "电子产品相关的关联规则网络图", "电子产品_关联规则网络图.png", output_dir)

    plot_high_value_payment(df_orders, output_dir)
    time_based_analysis(df_orders, output_dir)

    log("✅ 分析完成")

if __name__ == "__main__":
    main()
