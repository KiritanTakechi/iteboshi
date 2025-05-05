# create_student_model_diag.py
import os
import sys
import torch
import transformers
from transformers import (
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperTokenizerFast,
    WhisperFeatureExtractor,
)
from transformers.utils import is_tokenizers_available

# --- 配置参数 ---
teacher_checkpoint = "openai/whisper-medium"
student_decoder_layers = 4
student_model_save_path = "./iteboshi_student_model" # 使用新的目录以避免混淆

# --- 主程序 ---
def main():
    print("--- 开始创建 Whisper 学生模型 (增强诊断) ---")
    print(f"教师模型 checkpoint: {teacher_checkpoint}")
    print(f"学生模型解码器层数: {student_decoder_layers}")
    print(f"模型保存路径: {student_model_save_path}")
    print("-" * 30)

    # 0. 检查环境和库
    print("[步骤 0/6] 检查环境和库...")
    print(f"Python 可执行文件: {sys.executable}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"Transformers 版本: {transformers.__version__}")
    try:
        import tokenizers
        print(f"Tokenizers 库版本: {tokenizers.__version__}")
        print(f"Tokenizers 库是否被 Transformers 检测到可用: {is_tokenizers_available()}")
        if not is_tokenizers_available():
             print("警告：Transformers 报告 tokenizers 库不可用，即使它可能已安装。")
             print("尝试继续，但可能无法加载 Fast Tokenizer。")
    except ImportError:
        print("错误：'tokenizers' 库未安装！无法生成 tokenizer.json。")
        print("请运行 'pip install tokenizers' 然后重试。")
        return
    print("-" * 30)

    # 1. 加载教师模型的配置 (不变)
    print("[步骤 1/6] 加载教师模型配置...")
    try:
        teacher_config = WhisperConfig.from_pretrained(teacher_checkpoint)
        print("教师配置加载成功。")
    except Exception as e:
        print(f"错误：加载教师模型配置失败: {e}")
        return
    print("-" * 30)

    # 2. 修改配置以创建学生模型配置 (不变)
    print("[步骤 2/6] 修改配置以创建学生模型配置...")
    student_config = WhisperConfig.from_dict(teacher_config.to_dict())
    student_config.decoder_layers = student_decoder_layers
    print(f"学生模型配置已修改: 解码器层数 = {student_config.decoder_layers}")
    print("-" * 30)

    # 3. 创建学生模型并加载部分权重 (不变)
    print("[步骤 3/6] 初始化学生模型结构并加载教师权重...")
    try:
        student_model = WhisperForConditionalGeneration.from_pretrained(
            teacher_checkpoint,
            config=student_config,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=True if torch.cuda.is_available() else False
        )
        print("学生模型创建成功。")
    except Exception as e:
        print(f"错误：创建学生模型时出错: {e}")
        return
    print("-" * 30)

    # 4. 显式加载 Fast Tokenizer 和 Feature Extractor
    print("[步骤 4/6] 显式加载 Fast Tokenizer 和 Feature Extractor...")
    tokenizer = None # 初始化为 None
    feature_extractor = None # 初始化为 None
    try:
        print(f"尝试显式加载 WhisperTokenizerFast for '{teacher_checkpoint}'...")
        # --- 直接加载 Fast Tokenizer ---
        tokenizer = WhisperTokenizerFast.from_pretrained(teacher_checkpoint)
        print(f"成功加载 Tokenizer，类型: {type(tokenizer)}")
        if not isinstance(tokenizer, WhisperTokenizerFast):
             print("严重警告：加载的 Tokenizer 不是 WhisperTokenizerFast 类型！")
             print("将无法生成 tokenizer.json。请检查 transformers 和 tokenizers 库的安装与兼容性。")
             # 你可以选择在这里退出，因为目标无法达成
             # return

        print(f"尝试显式加载 WhisperFeatureExtractor for '{teacher_checkpoint}'...")
        feature_extractor = WhisperFeatureExtractor.from_pretrained(teacher_checkpoint)
        print(f"成功加载 Feature Extractor，类型: {type(feature_extractor)}")

    except ImportError:
         print("错误：无法导入 WhisperTokenizerFast 或 WhisperFeatureExtractor。")
         print("这通常意味着 transformers 库安装不完整或版本有问题。")
         return
    except Exception as e:
        print(f"错误：显式加载 Tokenizer 或 Feature Extractor 时出错: {e}")
        return
    print("-" * 30)

    # 5. 保存 Tokenizer, Feature Extractor (和可选的 Processor)
    print("[步骤 5/6] 保存 Tokenizer 和 Feature Extractor...")
    if tokenizer is None or feature_extractor is None:
        print("错误：Tokenizer 或 Feature Extractor 未能成功加载，无法保存。")
        return

    try:
        print(f"确保目标目录存在: {student_model_save_path}")
        os.makedirs(student_model_save_path, exist_ok=True)

        # --- 直接单独保存 Tokenizer ---
        print(f"将 Fast Tokenizer 保存到: {student_model_save_path}")
        # 这个 save_pretrained 对于 Fast Tokenizer 应该生成 tokenizer.json
        tokenizer.save_pretrained(student_model_save_path)
        print("Fast Tokenizer 保存完成。")
        # 检查 tokenizer.json 是否真的生成了
        tokenizer_json_path = os.path.join(student_model_save_path, "tokenizer.json")
        if os.path.exists(tokenizer_json_path):
            print(f"成功：检测到文件 '{tokenizer_json_path}' 已生成！")
        else:
            print(f"失败：未能检测到文件 '{tokenizer_json_path}'。")
            print("这非常奇怪，请检查文件系统权限或是否有其他进程干扰。")
            # 列出目录内容以帮助调试
            try:
                print(f"目录 '{student_model_save_path}' 当前内容:")
                for item in os.listdir(student_model_save_path):
                    print(f"- {item}")
            except Exception as list_e:
                print(f"(无法列出目录内容: {list_e})")


        # --- 保存 Feature Extractor ---
        print(f"将 Feature Extractor 保存到: {student_model_save_path}")
        feature_extractor.save_pretrained(student_model_save_path)
        print("Feature Extractor 保存完成。")

        # --- 可选：如果你仍然需要 Processor 对象用于其他地方 ---
        # processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        # print("重新组合了 Processor 对象 (如果需要的话)。")
        # 注意：再次调用 processor.save_pretrained() 可能会覆盖文件，但理论上结果应该一致

    except Exception as e:
        print(f"错误：保存 Tokenizer 或 Feature Extractor 时出错: {e}")
        return
    print("-" * 30)

    # 6. 保存学生模型的最终状态 (模型权重和配置文件)
    print(f"[步骤 6/6] 将初始化的学生模型 (权重和配置) 保存到: {student_model_save_path}")
    try:
        # 再次确保目录存在
        os.makedirs(student_model_save_path, exist_ok=True)
        student_model.save_pretrained(student_model_save_path)
        print("学生模型权重和配置保存成功。")
    except Exception as e:
        print(f"错误：保存学生模型失败: {e}")
        return

    print("-" * 30)
    print("--- Whisper 学生模型创建和保存完成 (增强诊断) ---")
    print(f"所有文件已尝试保存到: {student_model_save_path}")
    print("请再次检查该目录的内容，特别是 'tokenizer.json' 文件是否存在。")

if __name__ == "__main__":
    main()