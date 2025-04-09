# create_student_model.py

import os
from transformers import WhisperConfig, WhisperForConditionalGeneration

# --- 配置参数 ---
# 作为基础的预训练 Whisper 模型 checkpoint (用于加载配置和部分权重)
teacher_checkpoint = "openai/whisper-base"
# 学生模型解码器层数
student_decoder_layers = 2
# 保存学生模型结构和初始权重的位置
student_model_save_path = "./iteboshi" # 本地路径

# --- 主程序 ---

def main():
    print(f"基于 {teacher_checkpoint} 创建学生模型...")

    # 1. 加载教师模型的配置
    print("加载教师模型配置...")
    try:
        teacher_config = WhisperConfig.from_pretrained(teacher_checkpoint)
        print("教师配置加载成功。")
    except Exception as e:
        print(f"加载教师模型配置失败: {e}")
        return

    # 2. 修改配置以创建学生模型配置
    student_config = WhisperConfig.from_dict(teacher_config.to_dict()) # 复制配置
    # student_config.forced_decoder_ids = None
    student_config.decoder_layers = student_decoder_layers
    print(f"学生模型配置已修改: 解码器层数 = {student_config.decoder_layers}")

    # 3. 创建学生模型
    # 使用 from_pretrained 加载教师模型的权重
    # 同时传入修改后的 student_config
    # ignore_mismatched_sizes=True 允许加载权重，即使层数不匹配
    #   - 编码器、嵌入层等匹配的权重会被加载
    #   - 解码器层由于数量不同，其权重会被随机初始化（或者部分加载，取决于实现细节，但主要是随机）
    print("尝试从教师模型 checkpoint 初始化学生模型...")
    try:
        student_model = WhisperForConditionalGeneration.from_pretrained(
            teacher_checkpoint,
            config=student_config,
            ignore_mismatched_sizes=True,
            # low_cpu_mem_usage=True # 尝试减少加载时的内存使用
        )
        print("学生模型创建成功。部分权重已从教师模型加载，不匹配部分（主要是解码器）已重新初始化。")
    except Exception as e:
        print(f"创建学生模型时出错: {e}")
        # 如果内存不足，加载教师模型可能会失败
        print("如果遇到内存不足 (OOM) 问题，可能无法直接加载教师模型权重。")
        print("备选方案：仅使用配置初始化学生模型（权重完全随机）：")
        # student_model = WhisperForConditionalGeneration(config=student_config)
        # print("已使用配置创建了完全随机权重的学生模型。")
        return # 暂时退出，让用户决定如何处理

    # 4. 保存学生模型的初始状态（模型权重和配置文件）
    print(f"将初始化的学生模型保存到: {student_model_save_path}")
    try:
        os.makedirs(student_model_save_path, exist_ok=True)
        student_model.save_pretrained(student_model_save_path)
        print("学生模型保存成功。")
    except Exception as e:
        print(f"保存学生模型失败: {e}")

if __name__ == "__main__":
    main()
