import re
import csv
from collections import Counter

from XiNanData import sentences

FILE_PREFIX = "XiNan"

def save_dataset():
    import pandas as pd

    # 创建DataFrame
    df = pd.DataFrame(sentences)

    # 保存为Excel文件
    file_name = FILE_PREFIX + "精品句子.xlsx"
    df.to_excel(file_name, index=False, sheet_name="精品句子集")

    print(f"文件已保存为: {file_name}")
    print(f"总句子数: {len(sentences)}条")

# 1. 提取词汇并统计词频
def extract_and_count(sentences):
    pattern = r"【(.*?)】"
    all_words = []
    for sentence in sentences:
        matches = re.findall(pattern, sentence)
        all_words.extend(matches)
    word_freq = Counter(all_words)
    return word_freq

# 2. 保存为三列TSV（词汇|词频|释义）
def save_sorted_tsv(word_freq, filename=FILE_PREFIX + "TF.tsv"):
    # 按词频降序排序
    sorted_words = word_freq.most_common()
    with open(filename, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["词汇", "词频", "释义"])  # 表头
        for word, freq in sorted_words:
            writer.writerow([f"【{word}】", freq, ""])  # 第三列留空

# 3. 统计词汇长度分布
def analyze_length_distribution(word_freq):
    length_freq = Counter()
    length_counts = Counter()
    for word in word_freq.elements():
        length_freq[len(word)] += word_freq[word]  # 按实际出现次数统计
        length_counts[len(word)] += 1
    return length_freq, length_counts

# 主流程
if __name__ == "__main__":
    # 统计词频
    word_freq = extract_and_count(sentences)
    
    # 保存TSV（按词频降序）
    save_sorted_tsv(word_freq)
    print("TSV文件已生成：词汇 | 词频 | 释义")
    
    # 长度分布统计
    length_freq, length_counts = analyze_length_distribution(word_freq)
    
    # 输出统计结果
    print("\n词频统计TOP3：")
    for word, freq in word_freq.most_common(3):
        print(f"【{word}】: {freq}次")
    
    print("\n词汇长度分布：")
    for length, count in sorted(length_freq.items()):
        print(f"{length}字词: {count}次（涉及{length_counts[length]}个词）")