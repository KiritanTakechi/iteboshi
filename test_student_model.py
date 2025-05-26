# test_comparison.py
import math
import os
import time
import warnings
import evaluate
import torch

# 导入 polars
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    print("警告: 未找到 'polars' 库。如需导出 CSV，请运行 'pip install polars'。")

from datasets import load_dataset, Audio
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq, PretrainedConfig
from huggingface_hub import HfFileSystem, model_info # 新增：用于尝试获取 Hub 模型大小

# --- 配置参数 ---
# 模型 ID
MODEL_OFFICIAL = ["openai/whisper-base", "openai/whisper-tiny", "openai/whisper-small", "openai/whisper-medium"]
# 对于学生模型，我们假设它被保存在一个可访问的本地路径
# 如果 MODEL_STUDENT = "kiritan/iteboshi" 是 Hub ID，我们需要一个本地路径来计算大小
# 假设学生模型在训练后保存在如 "./iteboshi_student_model_trained" 或类似路径
# 为了演示，我们先假设它是一个Hub ID，然后尝试获取Hub上的大小，或者如果能映射到本地就计算本地的
STUDENT_MODEL_HUB_ID = ["kiritan/iteboshi-tiny", "kiritan/iteboshi-small", "kiritan/iteboshi-medium"] # 学生模型在 Hub 上的 ID
# 如果你知道学生模型的确切本地保存路径 (例如，训练脚本的输出目录)，请在这里指定
# 这将用于精确计算学生模型的大小
STUDENT_MODEL_LOCAL_PATH = "./iteboshi_student_model_trained" # 假设的本地路径，你需要根据实际情况修改

# 数据集参数 (保持不变)
SOURCE_DATASET_NAME = "mozilla-foundation/common_voice_11_0"
LANGUAGE_CODE = "zh-CN"
TEST_SPLIT_NAME = "test"
NUM_TEST_SAMPLES = 3000
AUDIO_COLUMN = "audio"
TEXT_COLUMN = "sentence"

# 推理参数 (保持不变)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PIPELINE_BATCH_SIZE = 16
MAP_BATCH_SIZE = PIPELINE_BATCH_SIZE * 2

# Hugging Face token (保持不变)
HUGGING_FACE_TOKEN = None # 或者 HfFolder.get_token()

# CSV 导出文件名 (保持不变)
CSV_EXPORT_FILENAME = "comparison_results.csv"


# --- Helper Functions ---

def get_directory_size_mb(directory_path):
    """计算目录的总大小 (MB)"""
    total_size_bytes = 0
    if not os.path.isdir(directory_path):
        print(f"  警告: 路径 '{directory_path}' 不是一个有效的目录，无法计算大小。")
        return float('nan')
    try:
        for dirpath, dirnames, filenames in os.walk(directory_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size_bytes += os.path.getsize(fp)
        return total_size_bytes / (1024 * 1024)  # 转换为 MB
    except Exception as e:
        print(f"  警告: 计算目录 '{directory_path}' 大小时出错: {e}")
        return float('nan')

def get_hub_model_size_mb(model_id, token=None):
    """尝试从 Hugging Face Hub 获取模型（主要是 safetensors 或 pytorch_model.bin）的大小 (MB)"""
    total_size_bytes = 0
    try:
        fs = HfFileSystem(token=token)
        # 列出模型仓库中的所有文件及其大小
        files_info = fs.ls(model_id, detail=True)
        relevant_files = [
            "model.safetensors",
            "pytorch_model.bin",
            # 你可以添加其他你认为重要的模型文件，比如权重分片 "pytorch_model-00001-of-0000X.bin"
        ]
        # 优先选择 safetensors
        if any(fi["name"].endswith("model.safetensors") for fi in files_info):
            for fi in files_info:
                if fi["name"].endswith("model.safetensors"): # 通常只有一个
                    total_size_bytes += fi["size"]
                    print(f"  Hub: 找到 model.safetensors, 大小: {fi['size'] / (1024*1024):.2f} MB")
                    break # 假设只有一个
        # 否则，查找 pytorch_model.bin (包括分片)
        elif any(fi["name"].endswith("pytorch_model.bin") or ("pytorch_model" in fi["name"] and ".bin" in fi["name"]) for fi in files_info):
            found_main_bin = False
            for fi in files_info:
                if fi["name"].endswith("pytorch_model.bin"): # 主文件
                    total_size_bytes += fi["size"]
                    print(f"  Hub: 找到 pytorch_model.bin, 大小: {fi['size'] / (1024*1024):.2f} MB")
                    found_main_bin = True
                    break # 如果有主文件，通常不和分片混用
            if not found_main_bin: # 检查分片
                for fi in files_info:
                     # 简单匹配分片文件名模式
                    if "pytorch_model-" in fi["name"] and fi["name"].endswith(".bin") and "of" in fi["name"]:
                        total_size_bytes += fi["size"]
                        print(f"  Hub: 找到分片 {os.path.basename(fi['name'])}, 大小: {fi['size'] / (1024*1024):.2f} MB")
        else:
            print(f"  Hub: 未在模型仓库 '{model_id}' 中找到主要的模型权重文件 (safetensors/bin)。")
            # 作为后备，可以考虑使用 model_info().size，但这通常是整个仓库的大小
            try:
                info = model_info(model_id, token=token)
                if info.size:
                    print(f"  Hub: model_info().size (整个仓库): {info.size / (1024*1024):.2f} MB. 这可能不准确。")
                    # total_size_bytes = info.size # 取决于你是否想用这个作为后备
            except Exception:
                pass # 忽略 model_info 错误

        if total_size_bytes > 0:
            return total_size_bytes / (1024 * 1024)
        return float('nan')
    except Exception as e:
        print(f"  警告: 从 Hub 获取模型 '{model_id}' 大小时出错: {e}")
        return float('nan')


def load_model_and_processor(model_id_or_path, use_auth_token=None, is_student_model=False):
    """加载指定 ID 或路径的模型和处理器，并返回模型大小信息"""
    print(f"\n正在加载模型和处理器: {model_id_or_path} ...")
    model_size_mb = float('nan')
    try:
        processor = AutoProcessor.from_pretrained(model_id_or_path, use_auth_token=use_auth_token)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id_or_path, use_auth_token=use_auth_token)
        model.to(DEVICE)
        model.eval()
        print(f"模型 {model_id_or_path} 加载成功并移至 {DEVICE}.")

        # 计算模型大小
        if is_student_model and os.path.isdir(STUDENT_MODEL_LOCAL_PATH): # 优先使用学生模型的本地路径
            print(f"  计算学生模型 '{model_id_or_path}' (来自本地路径 '{STUDENT_MODEL_LOCAL_PATH}') 的大小...")
            model_size_mb = get_directory_size_mb(STUDENT_MODEL_LOCAL_PATH)
        elif os.path.isdir(model_id_or_path): # 如果 model_id_or_path 本身是目录
            print(f"  计算本地模型 '{model_id_or_path}' 的大小...")
            model_size_mb = get_directory_size_mb(model_id_or_path)
        else: # 对于 Hub 模型，尝试从 Hub API 获取
            print(f"  尝试从 Hub API 获取模型 '{model_id_or_path}' 的权重文件大小...")
            model_size_mb = get_hub_model_size_mb(model_id_or_path, token=use_auth_token)

        if not math.isnan(model_size_mb):
            print(f"  模型大小: {model_size_mb:.2f} MB")
        else:
            print(f"  未能确定模型 '{model_id_or_path}' 的大小。")

        return model, processor, model_size_mb
    except Exception as e:
        print(f"!!! 加载模型/处理器 {model_id_or_path} 失败: {e}")
        print("请检查:")
        print(f"  1. 模型 ID/路径 '{model_id_or_path}' 是否正确。")
        print(f"  2. 网络连接是否正常 (如果从 Hub 加载)。")
        print(f"  3. 如果是私有模型，是否已登录或提供了有效的 HUGGING_FACE_TOKEN。")
        return None, None, float('nan')


def evaluate_model(model_id_or_path, dataset, text_column, audio_column, device, pipeline_batch_size, map_batch_size, use_auth_token=None, is_student_model=False):
    """使用指定模型评估数据集，返回 WER, CER, 平均推理时间和峰值 GPU 显存, 模型大小"""
    print(f"\n--- 开始评估模型: {model_id_or_path} ---")
    model, processor, model_size_mb = load_model_and_processor(model_id_or_path, use_auth_token, is_student_model)

    metrics = {
        "WER": float('nan'),
        "CER": float('nan'),
        "Avg Inference Time (s)": float('nan'),
        "Peak GPU Memory (MiB)": float('nan'),
        "Model Size (MB)": model_size_mb # 从加载函数获取
    }

    if model is None or processor is None:
        print(f"由于模型或处理器加载失败，跳过对 {model_id_or_path} 的评估。")
        return metrics

    # 确定 pipeline 的语言和任务参数
    # 对于 OpenAI Whisper 模型，语言需要明确指定，任务通常是 transcribe
    # 对于学生模型 kiritan/iteboshi，如果它是在 openai/whisper-medium 基础上微调的，也应该遵循这个逻辑
    # 或者，如果它的 processor config 中保存了 language 和 task，可以尝试读取
    language = None
    task = "transcribe" # 默认为转录

    config = getattr(model, "config", None)
    if config and hasattr(config, "forced_decoder_ids") and config.forced_decoder_ids:
        # 尝试从 forced_decoder_ids 推断语言 (这是一种间接方式)
        # WhisperTokenizerFast.get_decoder_prompt_ids(language="english", task="transcribe")
        # 但更可靠的是直接使用或从 processor 获取
        pass

    # 尝试从 processor 获取语言，这是更推荐的方式
    # WhisperProcessor.tokenizer.language / .task
    tok_lang = getattr(processor.tokenizer, "language", None)
    tok_task = getattr(processor.tokenizer, "task", None)

    if "openai/whisper" in model_id_or_path or model_id_or_path == STUDENT_MODEL_HUB_ID : # 假设学生模型行为类似
        language = "chinese" # 明确为中文
        task = "transcribe"
        print(f"为 OpenAI Whisper 系列或学生模型 {model_id_or_path} 设置: language='{language}', task='{task}'")
    elif tok_lang and tok_task:
        language = tok_lang
        task = tok_task
        print(f"从 {model_id_or_path} 的处理器推断: language='{language}', task='{task}'")
    else:
        print(f"警告: 未能从 {model_id_or_path} 的处理器中自动推断语言和任务。将使用默认 task='transcribe'，语言可能需要手动在 generate_kwargs 中设置。")
        # 如果 language 仍为 None，pipeline 可能会报错或使用默认语言，这可能不是我们想要的

    # --- GPU 显存监测准备 (仅在 GPU 上运行时) ---
    peak_memory_mib = float('nan')
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats(device=device)
            print(f"已在设备 {device} 上重置 GPU 峰值显存统计。")
        except Exception as e:
            print(f"警告: 重置设备 {device} 的显存统计失败: {e}")

    asr_pipeline = None
    try:
        print("创建 ASR pipeline...")
        generate_kwargs = {"task": task}
        if language: # 只有在 language 确定时才加入
            generate_kwargs["language"] = language

        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
            chunk_length_s=30, # Whisper 的标准块长度
            batch_size=pipeline_batch_size,
            generate_kwargs=generate_kwargs
        )
        print(f"Pipeline 创建成功，生成参数: {generate_kwargs}")
    except Exception as e:
        print(f"!!! 创建 Pipeline 失败 ({model_id_or_path}): {e}")
        del model
        del processor
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return metrics

    predictions = []
    references = [] # 初始化为空，将在 map 或 fallback 中填充
    print(f"开始对约 {len(dataset)} 个样本进行推理 (Pipeline Batch Size: {pipeline_batch_size}, Map Batch Size: {map_batch_size})...")
    total_inference_time = 0
    actual_num_samples_processed = 0

    inference_start_time = time.time()
    try:
        # 确保 dataset 包含 text_column
        if text_column not in dataset.column_names:
            raise ValueError(f"错误: 数据集中未找到指定的文本列 '{text_column}'。可用列: {dataset.column_names}")

        def predict_batch(batch):
            # 'array' 和 'sampling_rate' 应该在 audio 列的字典中
            audio_inputs = [item["array"] for item in batch[audio_column]]
            # pipeline 调用返回字典列表，每个字典含 'text'
            outputs = asr_pipeline(audio_inputs)
            return {"prediction_text": [pred['text'] for pred in outputs]}

        # 使用 map 进行推理, 移除所有不必要的列，只保留 text_column 和新生成的 prediction_text
        # 这样可以避免因其他列类型不匹配导致的错误
        columns_to_remove_for_map = [col for col in dataset.column_names if col != text_column]

        results_map = dataset.map(
            predict_batch,
            batched=True,
            batch_size=map_batch_size,
            remove_columns=columns_to_remove_for_map
        )
        predictions = results_map["prediction_text"]
        references = results_map[text_column] # map 之后，这个 text_column 应该还在
        actual_num_samples_processed = len(predictions)
        total_inference_time = time.time() - inference_start_time
    except Exception as e:
        print(f"!!! 使用 map 进行批量推理时出错 ({model_id_or_path}): {e}")
        print("将尝试逐个样本进行推理 (这会比较慢)...")
        predictions = []
        references_list = [] # 重置
        sample_inference_times = []
        fallback_start_time = time.time()
        for i, item in enumerate(dataset):
            try:
                audio_input = item[audio_column]["array"]
                reference_text = item[text_column] # 确保这里能获取到
                sample_start = time.time()
                result = asr_pipeline(audio_input) # pipeline 处理单个样本
                sample_end = time.time()
                predictions.append(result["text"])
                references_list.append(reference_text)
                sample_inference_times.append(sample_end - sample_start)
                if (i + 1) % 50 == 0 or (i + 1) == len(dataset):
                    print(f"  Fallback: 已处理 {i+1}/{len(dataset)} 个样本...")
            except Exception as inner_e:
                print(f"  Fallback: 处理样本 {i} 时出错: {inner_e}")
                predictions.append("")
                references_list.append(item.get(text_column, "")) # 安全获取
        total_inference_time = time.time() - fallback_start_time
        references = references_list
        actual_num_samples_processed = len(predictions)

    if device.startswith("cuda"):
        try:
            peak_memory_bytes = torch.cuda.max_memory_reserved(device=device)
            peak_memory_mib = peak_memory_bytes / (1024 * 1024)
            print(f"模型 {model_id_or_path} 推理期间峰值 GPU 显存使用 (Reserved): {peak_memory_mib:.2f} MiB")
            metrics["Peak GPU Memory (MiB)"] = peak_memory_mib
        except Exception as e:
             print(f"警告: 获取设备 {device} 的峰值显存失败: {e}")
             # metrics["Peak GPU Memory (MiB)"] 已经是 NaN 了

    if actual_num_samples_processed == 0:
        print(f"警告: 模型 '{model_id_or_path}' 未成功处理任何样本。所有指标将为 N/A。")
    else:
        if len(predictions) != len(references):
            print(f"!!! 严重错误: 预测数量 ({len(predictions)}) 与参考数量 ({len(references)}) 不匹配 ({model_id_or_path}) 即使在 fallback 之后。")
            # 这是个严重问题，可能需要停止或只处理匹配的部分
            min_len = min(len(predictions), len(references))
            predictions = predictions[:min_len]
            references = references[:min_len]
            actual_num_samples_processed = min_len
            print(f"    已将样本数截断至 {actual_num_samples_processed} 以计算指标。")

        avg_inference_time = total_inference_time / actual_num_samples_processed
        metrics["Avg Inference Time (s)"] = avg_inference_time
        print(f"推理完成 ({model_id_or_path})。处理了 {actual_num_samples_processed} 个样本。总耗时: {total_inference_time:.2f} 秒。")
        print(f"平均每个样本推理时间: {avg_inference_time:.4f} 秒。")

        print("正在计算 WER 和 CER...")
        wer_score = float('nan')
        cer_score = float('nan')
        try:
            wer_metric = evaluate.load("wer")
            cer_metric = evaluate.load("cer")
            cleaned_references = []
            cleaned_predictions = []
            valid_sample_count = 0
            for r_text, p_text in zip(references, predictions):
                ref_strip = str(r_text).strip() if r_text is not None else ""
                pred_strip = str(p_text).strip() if p_text is not None else ""
                if ref_strip:
                    cleaned_references.append(ref_strip)
                    cleaned_predictions.append(pred_strip)
                    valid_sample_count += 1

            if valid_sample_count == 0:
                print("警告: 没有有效的非空参考文本用于计算指标。")
            else:
                print(f"基于 {valid_sample_count} 个有效样本计算指标...")
                wer_score = 100 * wer_metric.compute(predictions=cleaned_predictions, references=cleaned_references)
                cer_score = 100 * cer_metric.compute(predictions=cleaned_predictions, references=cleaned_references)
                print(f"计算完成: WER = {wer_score:.2f}%, CER = {cer_score:.2f}%")
        except Exception as e:
            print(f"!!! 计算 WER/CER 时出错 ({model_id_or_path}): {e}")
            # wer_score, cer_score 保持 NaN

        metrics["WER"] = wer_score
        metrics["CER"] = cer_score

    print(f"清理模型 {model_id_or_path} 的资源...")
    del model
    del processor
    if asr_pipeline: del asr_pipeline
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("资源清理完毕。")
    return metrics


def main():
    print(f"开始对比测试...")
    print(f"设备: {DEVICE}")
    print(f"数据集: {SOURCE_DATASET_NAME} ({LANGUAGE_CODE})")
    print(f"测试集分割: {TEST_SPLIT_NAME}")
    print(f"测试样本数 (请求): {'全部' if NUM_TEST_SAMPLES is None else NUM_TEST_SAMPLES}")
    print(f"学生模型 Hub ID: {STUDENT_MODEL_HUB_ID}")
    print(f"学生模型本地路径 (用于大小计算): {STUDENT_MODEL_LOCAL_PATH if os.path.isdir(STUDENT_MODEL_LOCAL_PATH) else '未指定或无效'}")
    print("-" * 40)

    print("正在加载和准备数据集...")
    actual_num_samples = 0
    test_dataset_raw = None
    try:
        split_expr = f"{TEST_SPLIT_NAME}"
        if NUM_TEST_SAMPLES is not None and NUM_TEST_SAMPLES > 0 : # 确保 NUM_TEST_SAMPLES > 0
            split_expr += f"[:{NUM_TEST_SAMPLES}]"
        elif NUM_TEST_SAMPLES == 0: # 如果是0，则不加载任何样本，用于快速测试脚本逻辑
            print("警告: NUM_TEST_SAMPLES 设置为 0，将不加载实际数据。")
            split_expr += "[:0]"

        test_dataset_raw = load_dataset(
            SOURCE_DATASET_NAME,
            LANGUAGE_CODE,
            split=split_expr,
            use_auth_token=HUGGING_FACE_TOKEN,
            trust_remote_code=True,
            # cache_dir="./hf_cache" # 可选: 指定缓存目录
        )

        if AUDIO_COLUMN not in test_dataset_raw.column_names:
             raise ValueError(f"音频列 '{AUDIO_COLUMN}' 在数据集中未找到。可用列: {test_dataset_raw.column_names}")
        test_dataset_processed = test_dataset_raw.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))

        if TEXT_COLUMN not in test_dataset_processed.column_names:
            raise ValueError(f"文本列 '{TEXT_COLUMN}' 在数据集中未找到。可用列: {test_dataset_processed.column_names}")

        print("数据集加载和准备成功:")
        print(test_dataset_processed)
        actual_num_samples = len(test_dataset_processed)
        print(f"实际加载的测试样本数: {actual_num_samples}")
        if actual_num_samples == 0 and (NUM_TEST_SAMPLES is None or NUM_TEST_SAMPLES > 0):
            print("警告: 实际加载的样本数为0，但请求的样本数大于0。请检查数据集或 split_expr。")
            # 如果这里返回，将无法进行评估
            # return

    except Exception as e:
        print(f"!!! 加载或准备数据集失败: {e}")
        print("无法继续进行模型评估。")
        return

    results = {}
    models_to_test = MODEL_OFFICIAL + STUDENT_MODEL_HUB_ID # 使用 Hub ID 来加载模型

    for model_id_or_path in models_to_test:
        is_student = (model_id_or_path == STUDENT_MODEL_HUB_ID)
        results[model_id_or_path] = evaluate_model(
            model_id_or_path=model_id_or_path,
            dataset=test_dataset_processed, # 传递处理后的数据集
            text_column=TEXT_COLUMN,
            audio_column=AUDIO_COLUMN,
            device=DEVICE,
            pipeline_batch_size=PIPELINE_BATCH_SIZE,
            map_batch_size=MAP_BATCH_SIZE,
            use_auth_token=HUGGING_FACE_TOKEN,
            is_student_model=is_student
        )

    print("\n\n--- 最终评估结果对比 ---")
    print(f"数据集: {SOURCE_DATASET_NAME} ({LANGUAGE_CODE}) - {TEST_SPLIT_NAME}")
    print(f"测试样本数 (请求/实际): {'全部' if NUM_TEST_SAMPLES is None else NUM_TEST_SAMPLES}/{actual_num_samples}")
    print("-" * 115)
    header = f"{'Model':<45} | {'WER (%)':<10} | {'CER (%)':<10} | {'Avg Inf Time/Sample (s)':<25} | {'Peak Mem (MiB)':<15} | {'Size (MB)':<10}"
    print(header)
    print("-" * 115)

    data_for_export = []
    for model_id_key, metrics_dict in results.items():
        wer = metrics_dict.get('WER', float('nan'))
        cer = metrics_dict.get('CER', float('nan'))
        avg_time = metrics_dict.get('Avg Inference Time (s)', float('nan'))
        peak_mem = metrics_dict.get('Peak GPU Memory (MiB)', float('nan'))
        model_size = metrics_dict.get('Model Size (MB)', float('nan'))

        wer_str = f"{wer:.2f}" if not math.isnan(wer) else "N/A"
        cer_str = f"{cer:.2f}" if not math.isnan(cer) else "N/A"
        time_str = f"{avg_time:.4f}" if not math.isnan(avg_time) else "N/A"
        mem_str = f"{peak_mem:.2f}" if not math.isnan(peak_mem) else ("N/A (CPU)" if DEVICE == "cpu" else "N/A")
        size_str = f"{model_size:.2f}" if not math.isnan(model_size) else "N/A"

        print(f"{model_id_key:<45} | {wer_str:<10} | {cer_str:<10} | {time_str:<25} | {mem_str:<15} | {size_str:<10}")

        data_for_export.append({
            "Model": model_id_key,
            "WER (%)": wer,
            "CER (%)": cer,
            "Avg Inf Time/Sample (s)": avg_time,
            "Peak Mem (MiB)": peak_mem,
            "Model Size (MB)": model_size
        })
    print("-" * 115)

    if POLARS_AVAILABLE:
        print(f"\n正在将结果导出到 CSV 文件: {CSV_EXPORT_FILENAME} ...")
        try:
            df_results = pl.DataFrame(data_for_export)
            column_order = ["Model", "WER (%)", "CER (%)", "Avg Inf Time/Sample (s)", "Peak Mem (MiB)", "Model Size (MB)"]
            df_results = df_results.select(column_order)
            # 对于 float_precision，Polars write_csv 可能没有这个参数，而是全局设置或自动处理
            # 将 NaN 写入为 CSV 中的空字符串
            df_results.write_csv(CSV_EXPORT_FILENAME, null_value="", float_precision=2, include_header=False) # 保持 float_precision 由 Polars 默认处理
            print(f"结果已成功保存到 {CSV_EXPORT_FILENAME}")
        except Exception as e:
            print(f"!!! 使用 Polars 导出到 CSV 时出错: {e}")
    elif not POLARS_AVAILABLE:
         print(f"\n跳过 CSV 导出，因为未安装 'polars' 库。")

    print("\n测试完成.")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings("ignore", message="Using the update method is deprecated")
    warnings.filterwarnings("ignore", category=UserWarning, module='datasets.arrow_dataset')
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, message=".*TypedStorage is deprecated.*")
    print("重要提示：")
    print("1. 如果学生模型是私有模型, 请确保已通过 'huggingface-cli login' 登录。")
    print("2. Common Voice 数据集下载可能需要一些时间，并且需要 'trust_remote_code=True'。")
    print(f"3. 将使用设备: {DEVICE}。GPU 推理速度远快于 CPU。")
    print(f"4. Pipeline Batch Size: {PIPELINE_BATCH_SIZE}, Map Batch Size: {MAP_BATCH_SIZE} (可根据显存调整)。")
    print("5. 将尝试测量 GPU 峰值显存使用 (仅在 CUDA 设备上)。")
    if os.path.isdir(STUDENT_MODEL_LOCAL_PATH):
        print(f"6. 学生模型大小将从本地路径 '{STUDENT_MODEL_LOCAL_PATH}' 计算。")
    else:
        print(f"6. 警告: 学生模型本地路径 '{STUDENT_MODEL_LOCAL_PATH}' 未找到或无效，其大小可能无法精确计算。")
        print(f"   将尝试从 Hub ID '{STUDENT_MODEL_HUB_ID}' 获取其大小。")

    if POLARS_AVAILABLE:
        print(f"7. 评估结果将自动保存到 '{CSV_EXPORT_FILENAME}' 文件中。")
    else:
        print(f"7. (可选) 安装 'polars' (pip install polars) 以将结果保存到 CSV 文件。")
    main()