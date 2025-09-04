import pandas as pd
import re
import dashscope
import requests
import os
import time
from tqdm import tqdm  # 用于显示进度条

def clean_dialect_text(text):
    """
    清洗方言文本：移除【】符号和（）及其内容
    示例：输入"那个娃儿【灵醒】得很，学东西一{哈（下）}就上手了。"
         输出"那个娃儿灵醒得很，学东西一哈就上手了。"
    """
    # 移除【】符号（保留中间内容）
    text = re.sub(r'【(.*?)】', r'\1', text)
    
    # 移除（）及其中的内容
    text = re.sub(r'（[^)]*）', '', text)

    # 移除{}符号（保留中间内容）
    text = re.sub(r'\{(.*?)\}', r'\1', text)
    return text.strip()

def generate_dialect_audio(text, index, api_key, output_dir="dialect_audios"):
    """
    生成方言音频文件并保存
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成文件名
    filename = f"dialect_{index:04d}.wav"
    save_path = os.path.join(output_dir, filename)
    if os.path.isfile(save_path):  # 使用isfile确保是文件而非目录
        return True, save_path  # 直接返回已有文件路径
    
    # 调用语音合成API
    response = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
        model="qwen-tts-2025-05-22",
        api_key=api_key,
        text=text,
        voice="Jada"
    )
    
    # 处理API响应
    if response.status_code == 200:
        audio_url = response.output.audio["url"]
        
        # 下载音频文件
        audio_response = requests.get(audio_url)
        if audio_response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(audio_response.content)
            return True, save_path
        else:
            return False, f"音频下载失败: HTTP {audio_response.status_code}"
    else:
        return False, f"API调用失败: {response.message}"

def main():
    # 配置参数
    API_KEY = input("输入Qwen API Key：")  # 替换为您的API密钥
    # EXCEL_FILE = "20250812西南官话精品句.xlsx"
    OUTPUT_DIR = "../ShangHaiData"
    from ShangHaiData import sentences
    
    # 处理每条方言并生成音频
    success_count = 0
    failure_messages = []
    
    print("开始生成方言音频...")
    for i, row in enumerate(tqdm(sentences)):
        original_text = row
        
        # 清洗文本
        cleaned_text = clean_dialect_text(original_text)
        
        # 生成音频
        success, result = generate_dialect_audio(cleaned_text, i, API_KEY, OUTPUT_DIR)
        
        if success:
            success_count += 1
        else:
            failure_messages.append(f"第{i}行失败: {result} | 原文: {cleaned_text}\t{original_text}")
    
    # 输出结果摘要
    print(f"\n处理完成! 成功: {success_count}/{len(sentences)}")
    if failure_messages:
        print("\n失败详情:")
        for msg in failure_messages:
            print(f" - {msg}")

if __name__ == "__main__":
    main()