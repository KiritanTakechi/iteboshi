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
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq

# --- 配置参数 ---
# 模型 ID
MODEL_OFFICIAL = ["openai/whisper-base", "openai/whisper-tiny", "openai/whisper-small", "openai/whisper-medium"]
MODEL_STUDENT = "kiritan/iteboshi"
# 数据集参数
SOURCE_DATASET_NAME = "mozilla-foundation/common_voice_11_0"
LANGUAGE_CODE = "zh-CN" # Common Voice 的中文简体代码
TEST_SPLIT_NAME = "test" # 使用测试集
NUM_TEST_SAMPLES = 10000   # 测试样本数量 (设置为 None 测试所有样本) - 减少样本数以便快速测试
AUDIO_COLUMN = "audio"
TEXT_COLUMN = "sentence" # Common Voice 中包含文本的列名通常是 'sentence'
# 推理参数
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PIPELINE_BATCH_SIZE = 16 # Pipeline 内部的批处理大小，根据显存调整
MAP_BATCH_SIZE = PIPELINE_BATCH_SIZE * 2 # dataset.map 的批处理大小
# Hugging Face token
HUGGING_FACE_TOKEN = None
# CSV 导出文件名
CSV_EXPORT_FILENAME = "comparison_results.csv"

# --- Helper Functions ---

def load_model_and_processor(model_id, use_auth_token=None):
    """加载指定 ID 的模型和处理器"""
    print(f"\n正在加载模型和处理器: {model_id} ...")
    try:
        processor = AutoProcessor.from_pretrained(model_id, use_auth_token=use_auth_token)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, use_auth_token=use_auth_token)
        model.to(DEVICE)
        model.eval()
        print(f"模型 {model_id} 加载成功并移至 {DEVICE}.")
        return model, processor
    except Exception as e:
        print(f"!!! 加载模型/处理器 {model_id} 失败: {e}")
        print("请检查:")
        print(f"  1. 模型 ID '{model_id}' 是否正确。")
        print(f"  2. 网络连接是否正常。")
        print(f"  3. 如果是私有模型，是否已登录或提供了有效的 HUGGING_FACE_TOKEN。")
        return None, None

def evaluate_model(model_id, dataset, text_column, audio_column, device, pipeline_batch_size, map_batch_size, use_auth_token=None):
    """使用指定模型评估数据集，返回 WER, CER, 平均推理时间和峰值 GPU 显存"""
    print(f"\n--- 开始评估模型: {model_id} ---")
    model, processor = load_model_and_processor(model_id, use_auth_token)
    # 初始化指标为 NaN，确保即使加载失败也有完整的键
    metrics = {
        "WER": float('nan'),
        "CER": float('nan'),
        "Avg Inference Time (s)": float('nan'),
        "Peak GPU Memory (MiB)": float('nan') # 新增指标
    }
    if model is None or processor is None:
        return metrics # 返回包含 NaN 的字典

    # 确定 pipeline 的语言和任务参数
    if "openai/whisper" in model_id:
        language = "chinese"
        task = "transcribe"
        print(f"为 {model_id} 设置: language='{language}', task='{task}'")
    else:
        language = getattr(processor.tokenizer, "language", None)
        task = getattr(processor.tokenizer, "task", "transcribe")
        if not language:
             print(f"警告: 未能从 {model_id} 的处理器中自动推断语言。可能需要手动指定。")
        print(f"从 {model_id} 的处理器推断: language='{language}', task='{task}'")

    # --- GPU 显存监测准备 (仅在 GPU 上运行时) ---
    peak_memory_mib = float('nan')
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        # 确保在正确的设备上重置
        try:
            torch.cuda.reset_peak_memory_stats(device=device)
            print(f"已在设备 {device} 上重置 GPU 峰值显存统计。")
        except Exception as e:
            print(f"警告: 重置设备 {device} 的显存统计失败: {e}")


    asr_pipeline = None # 初始化 pipeline 变量
    try:
        print("创建 ASR pipeline...")
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
            chunk_length_s=30,
            batch_size=pipeline_batch_size,
            generate_kwargs={
                "language": language,
                "task": task,
            }
        )
        print("Pipeline 创建成功。")
    except Exception as e:
        print(f"!!! 创建 Pipeline 失败 ({model_id}): {e}")
        del model
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return metrics # 返回包含 NaN 的字典

    predictions = []
    references = dataset[text_column]
    print(f"开始对 {len(dataset)} 个样本进行推理 (Pipeline Batch Size: {pipeline_batch_size}, Map Batch Size: {map_batch_size})...")
    total_inference_time = 0
    actual_num_samples_processed = 0 # 记录实际处理的样本数

    # --- 推理过程开始 ---
    inference_start_time = time.time()
    try:
        def predict_batch(batch):
            audio_inputs = [item["array"] for item in batch[audio_column]]
            outputs = asr_pipeline(audio_inputs)
            return {"prediction_text": [pred['text'] for pred in outputs]}

        results_map = dataset.map(
            predict_batch,
            batched=True,
            batch_size=map_batch_size,
            remove_columns=[col for col in dataset.column_names if col != text_column] # 保留 text_column 以便后续对齐
        )
        # 从 map 结果中提取预测和对齐的参考
        predictions = results_map["prediction_text"]
        references = results_map[text_column] # 使用 map 结果中的 reference 以保证对齐
        inference_end_time = time.time() # 记录 map 结束时间
        total_inference_time = inference_end_time - inference_start_time
        actual_num_samples_processed = len(predictions) # 更新实际处理样本数

    except Exception as e:
        print(f"!!! 使用 map 进行批量推理时出错 ({model_id}): {e}")
        print("将尝试逐个样本进行推理 (这会比较慢)...")
        predictions = []
        references_list = [] # 如果 map 失败，需要重新收集 references
        sample_inference_times = []
        fallback_start_time = time.time()
        for i, item in enumerate(dataset):
            try:
                audio_input = item[audio_column]["array"]
                reference_text = item[text_column]
                sample_start = time.time()
                result = asr_pipeline(audio_input)
                sample_end = time.time()
                predictions.append(result["text"])
                references_list.append(reference_text)
                sample_inference_times.append(sample_end - sample_start)
                if (i + 1) % 50 == 0:
                    print(f"  已处理 {i+1}/{len(dataset)} 个样本...")
            except Exception as inner_e:
                print(f"  处理样本 {i} 时出错: {inner_e}")
                predictions.append("") # 添加空预测以保持对齐
                references_list.append(item[text_column]) # 保持引用对齐
        fallback_end_time = time.time()
        total_inference_time = fallback_end_time - fallback_start_time # 或者 sum(sample_inference_times)
        references = references_list # 更新 references 列表
        actual_num_samples_processed = len(predictions) # 更新实际处理样本数
    # --- 推理过程结束 ---

    # --- 获取峰值显存 (仅在 GPU 上运行时) ---
    if device.startswith("cuda"):
        try:
            peak_memory_bytes = torch.cuda.max_memory_reserved(device=device)
            peak_memory_mib = peak_memory_bytes / (1024 * 1024) # 转换为 MiB
            print(f"模型 {model_id} 推理期间峰值 GPU 显存使用 (Reserved): {peak_memory_mib:.2f} MiB")
            metrics["Peak GPU Memory (MiB)"] = peak_memory_mib # 更新显存指标
        except Exception as e:
             print(f"警告: 获取设备 {device} 的峰值显存失败: {e}")
             metrics["Peak GPU Memory (MiB)"] = float('nan') # 获取失败则记录 NaN


    if len(predictions) != len(references):
       # 这个警告理论上不应再出现，因为 map 保留了 text_column 或 fallback 手动收集了
       print(f"!!! 内部错误: 预测数量 ({len(predictions)}) 与参考数量 ({len(references)}) 不匹配 ({model_id})。")
       # 尽量对齐以计算指标
       min_len = min(len(predictions), len(references))
       predictions = predictions[:min_len]
       references = references[:min_len]
       actual_num_samples_processed = min_len # 修正处理的样本数

    num_samples_processed = actual_num_samples_processed # 使用实际处理的数量
    avg_inference_time = total_inference_time / num_samples_processed if num_samples_processed > 0 else 0
    metrics["Avg Inference Time (s)"] = avg_inference_time # 更新时间指标
    print(f"推理完成 ({model_id})。处理了 {num_samples_processed} 个样本。总耗时: {total_inference_time:.2f} 秒。")
    print(f"平均每个样本推理时间: {avg_inference_time:.4f} 秒。")

    # 计算 WER 和 CER
    print("正在计算 WER 和 CER...")
    wer_score = float('nan')
    cer_score = float('nan')
    if num_samples_processed > 0: # 只有在成功处理了样本后才计算指标
        try:
            wer_metric = evaluate.load("wer")
            cer_metric = evaluate.load("cer")
            # 清理文本并过滤空参考
            cleaned_references = []
            cleaned_predictions = []
            valid_sample_count = 0
            for r, p in zip(references, predictions):
                ref_strip = r.strip() if isinstance(r, str) else ""
                pred_strip = p.strip() if isinstance(p, str) else ""
                if ref_strip: # 只对有非空参考文本的样本计算指标
                    cleaned_references.append(ref_strip)
                    cleaned_predictions.append(pred_strip)
                    valid_sample_count += 1

            if valid_sample_count == 0:
                print("警告: 没有有效的非空参考文本用于计算指标。")
            else:
                print(f"基于 {valid_sample_count} 个有效样本计算指标...")
                # 计算 WER
                wer_score = 100 * wer_metric.compute(predictions=cleaned_predictions, references=cleaned_references)
                # 计算 CER
                cer_score = 100 * cer_metric.compute(predictions=cleaned_predictions, references=cleaned_references)
                print(f"计算完成: WER = {wer_score:.2f}%, CER = {cer_score:.2f}%")
        except Exception as e:
            print(f"!!! 计算 WER/CER 时出错 ({model_id}): {e}")
            wer_score = float('nan') # 出错则重置为 NaN
            cer_score = float('nan')

    metrics["WER"] = wer_score # 更新 WER 指标
    metrics["CER"] = cer_score # 更新 CER 指标

    # 清理资源
    print(f"清理模型 {model_id} 的资源...")
    del model
    del processor
    if asr_pipeline: # 确保 pipeline 已创建才删除
        del asr_pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("资源清理完毕。")

    return metrics # 返回包含所有指标的字典

# --- 主程序 ---
def main():
    print(f"开始对比测试...")
    print(f"设备: {DEVICE}")
    print(f"数据集: {SOURCE_DATASET_NAME} ({LANGUAGE_CODE})")
    print(f"测试集分割: {TEST_SPLIT_NAME}")
    print(f"测试样本数: {'全部' if NUM_TEST_SAMPLES is None else NUM_TEST_SAMPLES}")
    print("-" * 40)

    # 1. 加载和准备数据集
    print("正在加载和准备数据集...")
    actual_num_samples = 0 # 初始化
    test_dataset = None
    try:
        split_expr = f"{TEST_SPLIT_NAME}"
        if NUM_TEST_SAMPLES is not None:
            split_expr += f"[:{NUM_TEST_SAMPLES}]"
        test_dataset = load_dataset(
            SOURCE_DATASET_NAME,
            LANGUAGE_CODE,
            split=split_expr,
            use_auth_token=HUGGING_FACE_TOKEN,
            trust_remote_code=True,
        )
        if AUDIO_COLUMN not in test_dataset.column_names:
             raise ValueError(f"音频列 '{AUDIO_COLUMN}' 在数据集中未找到。可用列: {test_dataset.column_names}")
        test_dataset = test_dataset.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))
        if TEXT_COLUMN not in test_dataset.column_names:
            raise ValueError(f"文本列 '{TEXT_COLUMN}' 在数据集中未找到。可用列: {test_dataset.column_names}")
        print("数据集加载和准备成功:")
        print(test_dataset)
        actual_num_samples = len(test_dataset)
        print(f"实际测试样本数: {actual_num_samples}")
    except Exception as e:
        print(f"!!! 加载或准备数据集失败: {e}")
        print("无法继续进行模型评估。")
        return # 数据集加载失败则退出

    # 2. 依次评估每个模型
    results = {}
    models_to_test = MODEL_OFFICIAL + [MODEL_STUDENT]
    for model_id in models_to_test:
        # 传递数据集引用
        results[model_id] = evaluate_model(
            model_id=model_id,
            dataset=test_dataset, # 传递原始数据集
            text_column=TEXT_COLUMN,
            audio_column=AUDIO_COLUMN,
            device=DEVICE,
            pipeline_batch_size=PIPELINE_BATCH_SIZE,
            map_batch_size=MAP_BATCH_SIZE,
            use_auth_token=HUGGING_FACE_TOKEN
        )

    # 3. 打印最终对比结果
    print("\n\n--- 最终评估结果对比 ---")
    print(f"数据集: {SOURCE_DATASET_NAME} ({LANGUAGE_CODE}) - {TEST_SPLIT_NAME}")
    print(f"测试样本数 (请求/实际): {'全部' if NUM_TEST_SAMPLES is None else NUM_TEST_SAMPLES}/{actual_num_samples}")
    print("-" * 95)
    header = f"{'Model':<35} | {'WER (%)':<10} | {'CER (%)':<10} | {'Avg Inf Time/Sample (s)':<25} | {'Peak Mem (MiB)':<15}"
    print(header)
    print("-" * 95)

    # 用于导出 CSV 的数据列表
    data_for_export = []

    for model_id, metrics in results.items():
        # 确保 metrics 包含所有键，即使 evaluate_model 提前失败
        wer = metrics.get('WER', float('nan'))
        cer = metrics.get('CER', float('nan'))
        avg_time = metrics.get('Avg Inference Time (s)', float('nan'))
        peak_mem = metrics.get('Peak GPU Memory (MiB)', float('nan'))

        wer_str = f"{wer:.2f}" if not math.isnan(wer) else "N/A"
        cer_str = f"{cer:.2f}" if not math.isnan(cer) else "N/A"
        time_str = f"{avg_time:.4f}" if not math.isnan(avg_time) else "N/A"
        mem_str = f"{peak_mem:.2f}" if not math.isnan(peak_mem) else ("N/A (CPU)" if DEVICE == "cpu" else "N/A") # 更精确的 N/A

        print(f"{model_id:<35} | {wer_str:<10} | {cer_str:<10} | {time_str:<25} | {mem_str:<15}")

        # 添加到导出列表，使用原始数值 (NaN 会被 Polars 处理)
        data_for_export.append({
            "Model": model_id,
            "WER (%)": wer,
            "CER (%)": cer,
            "Avg Inf Time/Sample (s)": avg_time,
            "Peak Mem (MiB)": peak_mem
        })
    print("-" * 95)

    # 4. 使用 Polars 导出结果到 CSV (如果可用)
    if POLARS_AVAILABLE:
        print(f"\n正在将结果导出到 CSV 文件: {CSV_EXPORT_FILENAME} ...")
        try:
            # 从字典列表创建 Polars DataFrame
            df_results = pl.DataFrame(data_for_export)

            # 定义列的顺序（可选，但更好）
            column_order = ["Model", "WER (%)", "CER (%)", "Avg Inf Time/Sample (s)", "Peak Mem (MiB)"]
            df_results = df_results.select(column_order) # 确保列按指定顺序

            # 将 NaN 写入为 CSV 中的空字符串（常见做法）
            df_results.write_csv(CSV_EXPORT_FILENAME, null_value="") # 或者保持默认，取决于后续处理需求
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
    warnings.filterwarnings("ignore", category=FutureWarning) # 忽略一些 transformers/datasets 的未来警告
    warnings.filterwarnings("ignore", category=UserWarning, message=".*TypedStorage is deprecated.*") # 忽略 torch 存储警告

    print("重要提示：")
    print("1. 如果 'kiritan/iteboshi' 是私有模型, 请确保已通过 'huggingface-cli login' 登录。")
    print("2. Common Voice 数据集下载可能需要一些时间，并且需要 'trust_remote_code=True'。")
    print(f"3. 将使用设备: {DEVICE}。GPU 推理速度远快于 CPU。")
    print(f"4. Pipeline Batch Size: {PIPELINE_BATCH_SIZE}, Map Batch Size: {MAP_BATCH_SIZE} (可根据显存调整)。")
    print("5. 将尝试测量 GPU 峰值显存使用 (仅在 CUDA 设备上)。")
    if POLARS_AVAILABLE:
        print(f"6. 评估结果将自动保存到 '{CSV_EXPORT_FILENAME}' 文件中。")
    else:
        print(f"6. (可选) 安装 'polars' (pip install polars) 以将结果保存到 CSV 文件。")

    main()