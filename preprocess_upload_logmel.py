# preprocess_upload_logmel.py

import os
import torch
from datasets import load_dataset, Audio, DatasetDict
from transformers import WhisperFeatureExtractor
from huggingface_hub import create_repo
import logging

# --- 配置参数 ---
# 你在 Hugging Face Hub 上的用户名或组织名
hf_username = "kiritan"
# 上一步上传的原始数据集仓库名称
hf_raw_repo_name = "iteboshi.raw"
# 你想创建的用于存储 Log-Mel 特征数据集的仓库名称
hf_logmel_repo_name = "iteboshi.logmel"
# 用于特征提取的 Whisper 模型 checkpoint (例如 'openai/whisper-base', 'openai/whisper-small' 等)
whisper_checkpoint = "openai/whisper-base"
# 使用的身份验证令牌
hf_token = True
# 并行处理的进程数（根据你的 CPU 和内存调整）
num_proc = max(1, os.cpu_count() * 3 // 4) # type: ignore
# 处理时的批次大小 (影响内存)
batch_size = 32

# --- 设置日志 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 主程序 ---

def main():
    logger.info("开始预处理流程...")

    # 1. 加载特征提取器
    logger.info(f"加载 Whisper 特征提取器: {whisper_checkpoint}")
    try:
        feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_checkpoint)
    except Exception as e:
        logger.error(f"加载特征提取器失败: {e}")
        return

    # 2. 加载上一步上传的原始数据集
    raw_repo_id = f"{hf_username}/{hf_raw_repo_name}"
    logger.info(f"加载原始数据集: {raw_repo_id}")
    try:
        dataset: DatasetDict = load_dataset(raw_repo_id) # type: ignore
        logger.info("原始数据集加载成功:")
        logger.info(dataset)
    except Exception as e:
        logger.error(f"加载原始数据集失败: {e}")
        return

    # 3. 确保音频采样率为 16kHz (Whisper 模型的要求)
    logger.info("将音频列转换为 16kHz 采样率...")
    try:
        # 使用 cast_column 进行转换，这通常是惰性求值的，在 map 中实际执行
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        logger.info("音频列类型已设置为 16kHz。")
    except Exception as e:
        logger.error(f"转换音频采样率时出错: {e}")
        return

    # 4. 定义预处理函数
    def prepare_dataset(batch):
        # 加载音频数据 (cast_column 后这里会自动重采样)
        audio_arrays = [x["array"] for x in batch["audio"]]
        # 计算 log-Mel 声谱图特征
        try:
            batch["input_features"] = feature_extractor(audio_arrays, sampling_rate=16000).input_features
        except Exception as e:
            logger.warning(f"处理批次中的某个音频时出错: {e}. 跳过该批次的部分数据可能导致问题。")
            # 尝试为这个批次生成空的或者占位的特征，避免 map 失败
            batch["input_features"] = [torch.zeros((feature_extractor.n_mels, 3000)) for _ in audio_arrays]

        # 返回包含 input_features 的字典
        return batch

    # 5. 应用预处理函数
    logger.info(f"开始使用 map 函数进行预处理 (num_proc={num_proc}, batch_size={batch_size})...")
    # batched=True 加速处理
    # remove_columns 移除不再需要的原始音频数据和可能存在的其他元数据，节省空间
    # writer_batch_size 控制写入缓存的大小，有助于控制内存峰值
    try:
        # 确定需要保留的列，通常是 'input_features' 和文本标签列 ('sentence' in Common Voice)
        # 其他列如 'client_id', 'path', 'up_votes' 等都可以移除
        columns_to_remove = [col for col in dataset['train'].column_names if col not in ['input_features', 'sentence']] # 假设文本列是 'sentence'
        if 'audio' not in columns_to_remove: # 确保 audio 列被移除
             columns_to_remove.append('audio')

        processed_dataset = dataset.map(
            prepare_dataset,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=columns_to_remove,
            writer_batch_size=batch_size * 10 # 调整这个值可能影响性能和内存
        )
        logger.info("预处理完成。处理后的数据集结构:")
        logger.info(processed_dataset)
    except Exception as e:
        logger.error(f"数据集 map 操作失败: {e}", exc_info=True) # 添加 exc_info 获取更详细的回溯
        return

    # 6. 上传处理后的数据集
    logmel_repo_id = f"{hf_username}/{hf_logmel_repo_name}"
    logger.info(f"准备将处理后的数据集上传到: {logmel_repo_id}")

    try:
        create_repo(logmel_repo_id, repo_type="dataset", exist_ok=True, token=hf_token if isinstance(hf_token, str) else None)
        logger.info(f"仓库 {logmel_repo_id} 已确认存在或已创建。")
    except Exception as e:
        logger.warning(f"创建或检查仓库时出错: {e}")

    try:
        logger.info("开始上传处理后的数据集...")
        processed_dataset.push_to_hub(logmel_repo_id, token=hf_token if isinstance(hf_token, str) else None)
        logger.info("处理后的数据集上传成功！")
    except Exception as e:
        logger.error(f"上传处理后的数据集失败: {e}")

if __name__ == "__main__":
    print("重要提示：请确保您已通过 'huggingface-cli login' 登录 Hugging Face Hub。")
    main()
