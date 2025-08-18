import re
import os
import fcntl
import requests
import json
from typing import List, Dict, Tuple
import time
import importlib
# 方言单选题生成器配置文件

# 大模型API配置
API_URL = "https://api.siliconflow.cn/v1/chat/completions"  # 请替换为你的API地址
MODEL_NAME = "deepseek-ai/DeepSeek-V3"  # 请替换为你要使用的模型

# 其他配置
TIMEOUT = 300  # API请求超时时间（秒）

REGION = "XiNan"
# 输出文件配置
OUTPUT_FILENAME = f"{REGION}DialectQuizzes.json"  # 输出文件名 

class DialectQuizGenerator:
    def __init__(self, api_key: str, api_url: str, model_name: str = "gpt-3.5-turbo"):
        """
        初始化方言单选题生成器
        
        Args:
            api_key: 大模型API密钥
            api_url: 大模型API地址
            model_name: 使用的模型名称
        """
        self.api_key = api_key
        self.llm_api_url = api_url
        self.llm_model_name = model_name
        
        # 加载方言词汇字典
        self.dialect_dict = self._load_dialect_dict()
        
        # 加载句子数据
        self.sentences = self._load_sentences()
        
        # 提示词模板
        self.quiz_prompt_template = """请根据以下方言句子和词汇释义，生成2-3道单选题来测试对方言词汇和整个句子意思的理解。

句子：{sentence}

方言词汇释义：
{dialect_explanations}

要求：
1. 题目要重点考察对方言词汇的理解，但不要直接问某个词的意思，而是把对词汇理解的考察融入题目中
2. 题目要考察对整个句子意思的理解，不要在题干中加入任何语音转写的内容，包括对句子的转写或词汇本身
3. 每个题目提供4个选项，其中1个正确答案
4. 选项要合理，不能太明显
5. 题目要贴近生活实际
6. 返回格式：JSON格式，包含题目列表，每个题目包含question(题目)、options(选项列表)、answer(正确答案)、explanation(解释)

请直接返回JSON格式，不要其他内容。

示例：
"""
        self.example_json = """
{
    "questions": [
      {
          "question": "整句话的意思最接近以下哪一项？",
          "options": [
            "A. 你偶尔查看一下停电短信息，以了解电脑没电的原因",
            "B. 你住在一个叫三不子的地方，刚才电脑停电了",
            "C. 你刚才在三不子看短信息，我问你电脑是不是没电了",
            "D. 你让三不子帮你看短信息，因为你电脑是不是没电了"
          ],
          "answer": "A",
          "explanation": "说话人建议对方偶尔查看一下停电短信息，以了解电脑没电的原因"
      },
      {
        "question": "这句话的主要目的是？",
        "options": [
          "A. 提醒对方偶尔查看停电信息",
          "B. 要求对方立即查看停电信息",
          "C. 询问对方是否知道停电信息",
          "D. 告诉对方停电信息的内容"
        ],
        "answer": "A",
        "explanation": "说话人建议对方偶尔查看一下停电短信息，以了解电脑没电的原因。"
      }
    ]
}
"""

    def _load_dialect_dict(self) -> Dict[str, str]:
        """加载方言词汇字典"""
        dialect_dict = {}
        try:
            with open(f'{REGION}Dict.tsv', 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and '\t' in line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            word = parts[0].strip()
                            meaning = parts[1].strip()
                            dialect_dict[word] = meaning
            print(f"成功加载 {len(dialect_dict)} 个方言词汇")
        except Exception as e:
            print(f"加载方言词汇字典失败: {e}")
        return dialect_dict

    def _load_sentences(self) -> List[str]:
        """加载句子数据"""
        try:
            module_name = f'{REGION}Data'
            # 动态导入模块
            module = importlib.import_module(module_name)
            return module.sentences
        except Exception as e:
            print(f"加载句子数据失败: {e}")
            return []

    def _extract_dialect_words(self, sentence: str) -> List[str]:
        """从句子中提取方言词汇"""
        pattern = r'【([^】]+)】'
        return re.findall(pattern, sentence)

    def _get_dialect_explanations(self, sentence: str) -> str:
        """获取句子中方言词汇的释义"""
        dialect_words = self._extract_dialect_words(sentence)
        explanations = []
        
        for word in dialect_words:
            full_word = f"【{word}】"
            if full_word in self.dialect_dict:
                explanations.append(f"{full_word}: {self.dialect_dict[full_word]}")
            else:
                explanations.append(f"{full_word}: 未找到释义")
        
        return "\n".join(explanations) if explanations else "无方言词汇"

    def _call_llm_api(self, sentence: str, dialect_explanations: str) -> str:
        """调用大模型API生成单选题"""
        prompt = self.quiz_prompt_template.format(
            sentence=sentence,
            dialect_explanations=dialect_explanations
        ) + self.example_json
        
        payload = {
            "model": self.llm_model_name,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(self.llm_api_url, json=payload, headers=headers, timeout=TIMEOUT)
            response.raise_for_status()
            result = response.json()

            if result.get("choices") and result["choices"][0].get("message"):
                content = result["choices"][0]["message"].get("content", "").strip()
                return content if content else "[LLM返回空内容]"
            else:
                return f"[LLM返回格式错误: {response.text}]"
        except requests.exceptions.RequestException as e:
            return f"[LLM请求失败: {e}]"
        except Exception as e:
            return f"[LLM未知错误: {e}]"

    def _parse_quiz_response(self, response: str) -> Dict:
        """解析LLM返回的题目数据"""
        try:
            # 尝试提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                return {"error": "无法解析JSON格式", "raw_response": response}
        except json.JSONDecodeError as e:
            return {"error": f"JSON解析失败: {e}", "raw_response": response}
        except Exception as e:
            return {"error": f"解析失败: {e}", "raw_response": response}

    def generate_quiz_for_sentence(self, sentence: str, sentence_index: int = 0) -> Dict:
        """为单个句子生成单选题"""
        print(f"\n正在处理句子 {sentence_index:04d}: {sentence}")
        
        # 获取方言词汇释义
        dialect_explanations = self._get_dialect_explanations(sentence)
        print(f"方言词汇释义: {dialect_explanations}")
        
        # 调用LLM API
        # import pdb; pdb.set_trace()
        llm_response = self._call_llm_api(sentence, dialect_explanations)
        print(f"LLM响应: {llm_response}...")
        
        # 解析响应
        quiz_data = self._parse_quiz_response(llm_response)
        
        return {
            "sentence_id": f"{sentence_index:04d}",  # 添加句子编号，格式为0000, 0001, 0002...
            "sentence": sentence,
            "dialect_words": self._extract_dialect_words(sentence),
            "dialect_explanations": dialect_explanations,
            "quiz_data": quiz_data
        }

    def append_quiz_to_file(self, quiz: Dict, filename: str = OUTPUT_FILENAME):
        """将单个题目增量式追加到JSON文件"""
        try:
            # 检查文件是否存在并初始化
            if not os.path.exists(filename):
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
            
            # 读取现有数据
            with open(filename, 'r+', encoding='utf-8') as f:
                # 获取文件锁定，防止并发写入冲突
                fcntl.flock(f, fcntl.LOCK_EX)
                
                try:
                    # 读取现有数据
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = [data]
                    
                    # 追加新题目
                    data.append(quiz)
                    
                    # 写回文件
                    f.seek(0)
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    f.truncate()  # 截断文件，防止旧数据残留
                finally:
                    # 释放文件锁
                    fcntl.flock(f, fcntl.LOCK_UN)
                    
            print(f"题目已增量保存到 {filename}")
        except Exception as e:
            print(f"增量保存文件失败: {e}")

    def generate_all_quizzes(self, max_sentences: int = 1000) -> List[Dict]:
        """为所有句子生成单选题（增量式保存）"""
        all_quizzes = []
        
        # 限制处理的句子数量，避免API调用过多
        sentences_to_process = self.sentences[:max_sentences]
        
        for i, sentence in enumerate(sentences_to_process):
            print(f"\n=== 处理第 {i+1}/{len(sentences_to_process)} 个句子 ===")
            
            quiz_result = self.generate_quiz_for_sentence(sentence, i)
            all_quizzes.append(quiz_result)
            
            # 增量保存到文件
            self.append_quiz_to_file(quiz_result)
            
            # 添加延迟避免API调用过于频繁
            if i < len(sentences_to_process) - 1:
                time.sleep(1)
        
        return all_quizzes


    def save_quizzes_to_file(self, quizzes: List[Dict], filename: str = "dialect_quizzes.json"):
        """将生成的题目保存到文件"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(quizzes, f, ensure_ascii=False, indent=2)
            print(f"题目已保存到 {filename}")
        except Exception as e:
            print(f"保存文件失败: {e}")

    def print_quiz_summary(self, quizzes: List[Dict]):
        """打印题目摘要"""
        print(f"\n=== 题目生成摘要 ===")
        print(f"总共处理句子: {len(quizzes)}")
        
        success_count = 0
        error_count = 0
        
        for i, quiz in enumerate(quizzes, 1):
            if "error" not in quiz["quiz_data"]:
                success_count += 1
                print(f"句子 {i}: 成功生成题目")
            else:
                error_count += 1
                print(f"句子 {i}: 生成失败 - {quiz['quiz_data'].get('error', '未知错误')}")
        
        print(f"\n成功: {success_count}, 失败: {error_count}")


def main():
    """主函数"""
    try:
        
        API_KEY = input("your_api_key_here: ")

        # 创建生成器实例
        generator = DialectQuizGenerator(API_KEY, API_URL, MODEL_NAME)
        
        # 生成题目
        quizzes = generator.generate_all_quizzes()
        
        # 保存题目到文件
        # generator.save_quizzes_to_file(quizzes, OUTPUT_FILENAME)
        
        # 打印摘要
        generator.print_quiz_summary(quizzes)
        
        # 打印第一个成功的题目作为示例
        for quiz in quizzes:
            if "error" not in quiz["quiz_data"]:
                print(f"\n=== 示例题目 ===")
                print(f"句子编号: {quiz['sentence_id']}")
                print(f"句子: {quiz['sentence']}")
                print(f"方言词汇: {quiz['dialect_words']}")
                print(f"题目数据: {json.dumps(quiz['quiz_data'], ensure_ascii=False, indent=2)}")
                break
                
    except ImportError:
        print("错误：无法导入配置文件，请确保 config.py 文件存在")
    except Exception as e:
        print(f"程序运行出错: {e}")


if __name__ == "__main__":
    main() 