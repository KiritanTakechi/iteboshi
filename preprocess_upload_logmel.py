# preprocess_upload_logmel.py

import os
import numpy as np
from datasets import load_dataset, Audio, DatasetDict, Features, Array2D
from transformers import WhisperFeatureExtractor
from huggingface_hub import create_repo, HfFolder
import logging

# --- 配置参数 ---
# 你在 Hugging Face Hub 上的用户名或组织名
hf_username = "kiritan"
# 上一步上传的原始数据集仓库名称
hf_raw_repo_name = "iteboshi.raw"
# 选择生成的 Log-Mel 特征维度 (80 for base/small/medium, 128 for large)
num_mel_bins = 80 # <--- 在这里选择 80 或 128
# 你想创建的用于存储 Log-Mel 特征数据集的仓库名称
hf_logmel_repo_name = f"iteboshi.logmel.{num_mel_bins}bins" # 修改这里以匹配 num_mel_bins
# 用于特征提取的 Whisper 模型 checkpoint (会影响默认的 Mel bins，但会被下面覆盖)
whisper_checkpoint = "openai/whisper-large-v2"
# 使用的身份验证令牌 (True 会尝试自动获取, 也可以直接填入字符串 token)
hf_token = HfFolder.get_token() # 尝试从缓存获取 token
# **修改**: 只处理这些 splits
splits_to_process = ['train', 'validation', 'test']
# 并行处理的进程数（根据你的 CPU 和内存调整）
num_proc = max(1, os.cpu_count() * 3 // 4) # type: ignore
# 处理时的批次大小 (影响内存)
batch_size = 64 # 根据内存调整，128 bins 特征更大
# --- 设置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 主程序 ---
def main():
    logger.info("开始预处理流程...")
    logger.info(f"配置: num_mel_bins={num_mel_bins}, whisper_checkpoint='{whisper_checkpoint}'")
    logger.info(f"目标 Log-Mel 仓库: '{hf_username}/{hf_logmel_repo_name}'")
    logger.warning(f"请确保目标仓库名 '{hf_logmel_repo_name}' 与设置的 num_mel_bins ({num_mel_bins}) 匹配！")

    # 检查 token
    if not hf_token:
        logger.warning("未找到 Hugging Face Hub token。可能无法加载私有数据或上传结果。请运行 'huggingface-cli login'")
        # 可以在这里决定是否退出
        # sys.exit("无法继续，缺少 Hugging Face token。")

    # 1. 加载特征提取器 (明确指定 num_mel_bins)
    logger.info(f"加载 Whisper 特征提取器 (强制 num_mel_bins={num_mel_bins})...")
    try:
        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            whisper_checkpoint,
            num_mel_bins=num_mel_bins
        )
        logger.info(f"特征提取器加载成功。确认特征维度 (feature_size): {feature_extractor.feature_size}")
        if feature_extractor.feature_size != num_mel_bins:
             logger.warning(f"配置请求 {num_mel_bins} bins，但加载的 feature_extractor 实际维度为 {feature_extractor.feature_size}。请检查 checkpoint 和配置。")
             # 根据需要决定是否在此处停止
    except Exception as e:
        logger.error(f"加载特征提取器失败: {e}", exc_info=True)
        return

    # 2. 加载原始数据集的指定 splits
    raw_repo_id = f"{hf_username}/{hf_raw_repo_name}"
    raw_dataset_dict = DatasetDict()
    logger.info(f"尝试从仓库 {raw_repo_id} 加载 splits: {splits_to_process}")
    for split in splits_to_process:
        try:
            # 使用 stream=True 可以在下载时就开始处理，节省磁盘空间，但 map 可能受限
            # 先不使用 stream=True，确保 map 能正常工作
            dataset_split = load_dataset(raw_repo_id, split=split, token=hf_token) # type: ignore
            raw_dataset_dict[split] = dataset_split
            logger.info(f"成功加载 split: '{split}' (大小: {len(raw_dataset_dict[split])})")
            logger.info(f"'{split}' split 的特征: {raw_dataset_dict[split].features}") # 打印特征以供检查
        except Exception as e:
            logger.warning(f"加载 split '{split}' 失败，将跳过该 split: {e}")

    if not raw_dataset_dict:
        logger.error("未能成功加载任何指定的 splits ('train', 'validation', 'test')，无法继续。")
        return
    logger.info(f"成功加载的原始数据集 splits: {list(raw_dataset_dict.keys())}")

    # 3. 确保音频采样率为 16kHz
    logger.info("将所有加载的 splits 的音频列转换为 16kHz 采样率...")
    try:
        # 使用 cast_column 进行转换
        for split_name in raw_dataset_dict.keys():
            logger.info(f"转换 '{split_name}' split 的音频列...")
            # 检查是否存在 audio 列
            if "audio" not in raw_dataset_dict[split_name].features:
                 logger.error(f"错误：Split '{split_name}' 中未找到 'audio' 列。请检查原始数据集 {raw_repo_id}。")
                 #可以选择移除这个split或者终止
                 del raw_dataset_dict[split_name]
                 logger.warning(f"已从待处理列表中移除缺少 'audio' 列的 split: '{split_name}'")
                 continue # 继续处理下一个split

            raw_dataset_dict[split_name] = raw_dataset_dict[split_name].cast_column(
                "audio", Audio(sampling_rate=16000)
            )
        logger.info("音频列类型已设置为 16kHz。")
        # 再次检查是否还有可处理的 splits
        if not raw_dataset_dict:
             logger.error("所有 splits 都因缺少 'audio' 列而被移除，无法继续。")
             return

    except Exception as e:
        logger.error(f"转换音频采样率时出错: {e}", exc_info=True)
        return

    # 4. 定义预处理函数
    def prepare_dataset(batch):
        # 加载音频数据 (cast_column 后这里会自动重采样)
        audio_arrays = [x["array"] for x in batch["audio"]]
        sampling_rates = [x["sampling_rate"] for x in batch["audio"]] # 获取实际采样率用于调试

        processed_features = []
        max_length_seconds = 30.0 # Whisper 处理的最大音频秒数
        expected_sequence_length = int(max_length_seconds * feature_extractor.sampling_rate / feature_extractor.hop_length) # 3000 for 16kHz

        for i, (audio_array, sr) in enumerate(zip(audio_arrays, sampling_rates)):
            if sr != 16000:
                 logger.warning(f"警告：在 prepare_dataset 中遇到非 16kHz ({sr}Hz) 的音频，这不应该发生，请检查 cast_column。索引: {i}")
                 # 可以尝试强制重采样，但这会很慢
                 # 或者跳过这个样本
                 # processed_features.append(torch.zeros((num_mel_bins, expected_sequence_length))) # 使用占位符
                 # continue
            try:
                # 计算 log-Mel 声谱图特征
                # 注意：feature_extractor 返回的是列表，即使只有一个输入
                # 需要确保输入是 numpy array 或 list of floats
                input_features = feature_extractor(
                    audio_array,
                    sampling_rate=16000, # 明确告知采样率
                    return_tensors="pt" # 直接返回 PyTorch 张量
                ).input_features[0] # 获取批次中的第一个（也是唯一一个）结果

                processed_features.append(input_features.cpu().numpy())
            except Exception as e:
                logger.warning(f"处理单个音频样本时出错 (索引 {i}，原始采样率 {sr}): {e}. 将使用零填充代替。")
                # 使用与 feature extractor 输出维度匹配的零填充
                placeholder_array = np.zeros((num_mel_bins, expected_sequence_length), dtype=np.float32)
                processed_features.append(placeholder_array)


        # 正确做法：返回一个只包含新列或所需列的新字典
        result_batch = {}
        result_batch["input_features"] = processed_features
        if "sentence" in batch: # 保留 sentence 列
            result_batch["sentence"] = batch["sentence"]
        # 保留其他可能需要的列，比如 id 等 (如果原始数据集有的话)
        # if "id" in batch:
        #    result_batch["id"] = batch["id"]
        return result_batch


    # 5. 应用预处理函数到每个 split
    processed_dataset_dict = DatasetDict()
    logger.info(f"开始使用 map 函数进行预处理 (num_proc={num_proc}, batch_size={batch_size})...")

    # 动态确定要保留的列 (除了 audio 和将被移除的列)
    example_split_name = list(raw_dataset_dict.keys())[0]
    original_features = raw_dataset_dict[example_split_name].features
    columns_to_keep = ["input_features"] # 新增的列
    if "sentence" in original_features: # 保留常见的文本列
        columns_to_keep.append("sentence")
    # 你可以在这里添加其他需要保留的列名
    # columns_to_keep.append("id")

    logger.info(f"预处理后将保留的列: {columns_to_keep}")
    columns_to_remove = [col for col in original_features.keys() if col not in columns_to_keep and col != "input_features"]
    # 确保 audio 肯定被移除
    if "audio" not in columns_to_remove:
        columns_to_remove.append("audio")
    logger.info(f"预处理时将尝试移除的列: {columns_to_remove}")


    # 定义输出数据集的特征结构 (重要！)
    # 需要知道 input_features 的具体形状，它是一个二维列表 (mel_bins, seq_len)
    # seq_len 通常是 3000 (对应 30 秒音频)
    output_features = Features({
        'input_features': Array2D(shape=(num_mel_bins, 3000), dtype='float32'), # 需要安装 pyarrow
        **( # 使用字典解包添加其他保留的列及其类型
             {col: original_features[col] for col in columns_to_keep if col != "input_features"}
        )
    })
    logger.info(f"定义的输出特征结构: {output_features}")


    for split_name, ds in raw_dataset_dict.items():
        logger.info(f"--- 开始处理 split: '{split_name}' ---")
        try:
            # 对当前 split 应用 map
            processed_ds = ds.map(
                prepare_dataset,
                batched=True,
                batch_size=batch_size,
                num_proc=num_proc,
                remove_columns=columns_to_remove, # 移除不再需要的原始列
                features=output_features, # **显式指定输出特征**
                writer_batch_size=batch_size * 10, # 调整这个值可能影响性能和内存
                desc=f"Processing {split_name}" # 添加描述
            )
            processed_dataset_dict[split_name] = processed_ds
            logger.info(f"--- 处理完成 split: '{split_name}' ---")
            logger.info(f"处理后的 '{split_name}' split 结构: {processed_ds.features}")
            logger.info(f"处理后的 '{split_name}' split 大小: {processed_ds.shape}")
            # 检查一个样本
            if processed_ds.shape > 0:
                 sample = processed_ds[0]
                 logger.info(f"处理后 '{split_name}' 的第一个样本 input_features 长度: {sample['input_features'].shape}")
                 # 如果定义为 Array2D 或 Tensor，可以使用 .shape
            else:
                logger.warning(f"处理后的 split '{split_name}' 为空！")

        except Exception as e:
            logger.error(f"处理 split '{split_name}' 时失败: {e}", exc_info=True)
            logger.warning(f"将跳过上传 split: {split_name}")

    if not processed_dataset_dict:
        logger.error("没有成功处理任何 split，无法上传。")
        return

    logger.info("所有可用 splits 处理完成。")
    logger.info(f"最终准备上传的 Processed DatasetDict (splits: {list(processed_dataset_dict.keys())}):")
    logger.info(processed_dataset_dict)

    # 6. 上传处理后的数据集
    logmel_repo_id = f"{hf_username}/{hf_logmel_repo_name}"
    logger.info(f"准备将处理后的数据集上传到: {logmel_repo_id}")
    try:
        create_repo(logmel_repo_id, repo_type="dataset", exist_ok=True, token=hf_token)
        logger.info(f"仓库 {logmel_repo_id} 已确认存在或已创建。")
    except Exception as e:
        logger.warning(f"创建或检查仓库时出错: {e}")

    try:
        logger.info("开始上传处理后的数据集...")
        # 上传包含所有成功处理的 splits 的 DatasetDict
        processed_dataset_dict.push_to_hub(logmel_repo_id, token=hf_token)
        logger.info("处理后的数据集上传成功！")
    except Exception as e:
        logger.error(f"上传处理后的数据集失败: {e}", exc_info=True)

if __name__ == "__main__":
    print("重要提示：请确保您已通过 'huggingface-cli login' 登录 Hugging Face Hub。")
    main()
