# test_comparison.py

import math
import os
import time
import warnings

import evaluate
import torch
from datasets import load_dataset, Audio
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq

# --- 配置参数 ---
# 模型 ID
MODEL_OFFICIAL = "openai/whisper-medium"
MODEL_STUDENT = "kiritan/iteboshi" # 你的模型 Hub ID

# 数据集参数
SOURCE_DATASET_NAME = "mozilla-foundation/common_voice_11_0"
LANGUAGE_CODE = "zh-CN" # Common Voice 的中文简体代码
TEST_SPLIT_NAME = "test" # 使用测试集
NUM_TEST_SAMPLES = 1000   # 测试样本数量 (设置为 None 测试所有样本)
AUDIO_COLUMN = "audio"
TEXT_COLUMN = "sentence" # Common Voice 中包含文本的列名通常是 'sentence'

# 推理参数
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PIPELINE_BATCH_SIZE = 16 # Pipeline 内部的批处理大小，根据显存调整
MAP_BATCH_SIZE = PIPELINE_BATCH_SIZE * 2 # dataset.map 的批处理大小

# Hugging Face token (加载公共模型/数据集通常不需要)
# 如果 kiritan/iteboshi 是私有的，需要设置 token 或提前 login
# HUGGING_FACE_TOKEN = "YOUR_HF_TOKEN" # 或者 None
HUGGING_FACE_TOKEN = None # 假设模型是公开的

# --- Helper Functions ---

def load_model_and_processor(model_id, use_auth_token=None):
    """加载指定 ID 的模型和处理器"""
    print(f"\n正在加载模型和处理器: {model_id} ...")
    try:
        # 使用 AutoProcessor 和 AutoModelForSpeechSeq2Seq 提高兼容性
        processor = AutoProcessor.from_pretrained(model_id, use_auth_token=use_auth_token)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, use_auth_token=use_auth_token)

        # 将模型移至设备并设置为评估模式
        model.to(DEVICE)
        model.eval()
        print(f"模型 {model_id} 加载成功并移至 {DEVICE}.")
        return model, processor
    except Exception as e:
        print(f"!!! 加载模型/处理器 {model_id} 失败: {e}")
        print("请检查:")
        print(f"  1. 模型 ID '{model_id}' 是否正确。")
        print(f"  2. 网络连接是否正常。")
        print(f"  3. 如果是私有模型，是否已登录 (huggingface-cli login) 或提供了有效的 HUGGING_FACE_TOKEN。")
        return None, None

def evaluate_model(model_id, dataset, text_column, audio_column, device, pipeline_batch_size, map_batch_size, use_auth_token=None):
    """使用指定模型评估数据集，返回 WER, CER 和平均推理时间"""
    print(f"\n--- 开始评估模型: {model_id} ---")

    model, processor = load_model_and_processor(model_id, use_auth_token)
    if model is None or processor is None:
        return {"WER": float('nan'), "CER": float('nan'), "Avg Inference Time (s)": float('nan')}

    # 确定 pipeline 的语言和任务参数
    if "openai/whisper" in model_id:
        # 对于官方 Whisper 多语言模型，需要明确指定语言
        language = "chinese"  # Whisper 能识别的简体中文代码
        task = "transcribe"
        print(f"为 {model_id} 设置: language='{language}', task='{task}'")
    else:
        # 尝试从学生模型的 processor 推断，如果保存正确的话
        language = getattr(processor.tokenizer, "language", None)
        task = getattr(processor.tokenizer, "task", "transcribe") # 默认为转录
        if not language:
             print(f"警告: 未能从 {model_id} 的处理器中自动推断语言。可能需要手动指定。")
             # language = "chinese" # 可以考虑在此处设置默认值或引发错误
        print(f"从 {model_id} 的处理器推断: language='{language}', task='{task}'")

    try:
        print("创建 ASR pipeline...")
        # 强制指定 language 和 task 参数给 generate_kwargs
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
            chunk_length_s=30, # Whisper 标准设置
            batch_size=pipeline_batch_size,
            generate_kwargs={
                "language": language,
                "task": task,
                # forced_decoder_ids 等通常由模型配置自动处理
            }
        )
        print("Pipeline 创建成功。")
    except Exception as e:
        print(f"!!! 创建 Pipeline 失败 ({model_id}): {e}")
        # 清理已加载的模型
        del model
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"WER": float('nan'), "CER": float('nan'), "Avg Inference Time (s)": float('nan')}

    predictions = []
    # 直接从原始数据集中获取所有参考文本
    references = dataset[text_column]

    print(f"开始对 {len(dataset)} 个样本进行推理 (Pipeline Batch Size: {pipeline_batch_size}, Map Batch Size: {map_batch_size})...")
    total_inference_time = 0

    # 使用 dataset.map 进行批量推理以提高效率
    try:
        # 定义用于 map 的函数
        def predict_batch(batch):
            # Pipeline 输入需要是音频数组列表
            audio_inputs = [item["array"] for item in batch[audio_column]]
            # Pipeline 返回字典列表，提取 'text'
            outputs = asr_pipeline(audio_inputs)
            return {"prediction_text": [pred['text'] for pred in outputs]}

        # 记录 map 操作的执行时间
        map_start_time = time.time()
        # 执行 map 操作
        # Note: 'remove_columns' helps save memory but ensure 'text_column' remains if needed later,
        # or better, get references before map.
        results = dataset.map(
            predict_batch,
            batched=True,
            batch_size=map_batch_size,
            remove_columns=dataset.column_names # 移除旧列以节省内存
        )
        map_end_time = time.time()
        total_inference_time = map_end_time - map_start_time
        predictions = results["prediction_text"] # 获取所有预测结果

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
                result = asr_pipeline(audio_input) # Pipeline 处理单个样本
                sample_end = time.time()

                predictions.append(result["text"])
                references_list.append(reference_text)
                sample_inference_times.append(sample_end - sample_start)

                if (i + 1) % 50 == 0: # 每 50 个样本打印一次进度
                    print(f"  已处理 {i+1}/{len(dataset)} 个样本...")

            except Exception as inner_e:
                print(f"  处理样本 {i} 时出错: {inner_e}")
                predictions.append("") # 添加空预测以保持对齐
                references_list.append(item[text_column]) # 保持引用对齐

        fallback_end_time = time.time()
        total_inference_time = fallback_end_time - fallback_start_time # 或者 sum(sample_inference_times)
        references = references_list # 更新 references 列表

    if len(predictions) != len(references):
       print(f"!!! 警告: 预测数量 ({len(predictions)}) 与参考数量 ({len(references)}) 不匹配 ({model_id})。指标可能不准确。")
       # 尝试对齐，取最短长度
       min_len = min(len(predictions), len(references))
       predictions = predictions[:min_len]
       references = references[:min_len]

    num_samples_processed = len(references)
    avg_inference_time = total_inference_time / num_samples_processed if num_samples_processed > 0 else 0

    print(f"推理完成 ({model_id})。总耗时: {total_inference_time:.2f} 秒。")
    print(f"平均每个样本推理时间: {avg_inference_time:.4f} 秒。")

    # 计算 WER 和 CER
    print("正在计算 WER 和 CER...")
    wer_score = float('nan')
    cer_score = float('nan')
    try:
        wer_metric = evaluate.load("wer")
        cer_metric = evaluate.load("cer")

        # 清理参考文本和预测文本中的换行符等，并过滤空参考，以获得更准确的指标
        cleaned_references = [r.strip() for r in references if r and r.strip()]
        cleaned_predictions = [p.strip() for p, r in zip(predictions, references) if r and r.strip()]

        if not cleaned_references:
            print("警告: 没有有效的非空参考文本用于计算指标。")
        else:
            print(f"基于 {len(cleaned_references)} 个有效样本计算指标...")
            # 计算 WER (Word Error Rate) - 对中文意义不大，但可以计算
            wer_score = 100 * wer_metric.compute(predictions=cleaned_predictions, references=cleaned_references)
            # 计算 CER (Character Error Rate) - 对中文更重要
            cer_score = 100 * cer_metric.compute(predictions=cleaned_predictions, references=cleaned_references)
            print(f"计算完成: WER = {wer_score:.2f}%, CER = {cer_score:.2f}%")

    except Exception as e:
        print(f"!!! 计算 WER/CER 时出错 ({model_id}): {e}")

    # 清理 GPU 显存
    print(f"清理模型 {model_id} 的资源...")
    del model
    del processor
    del asr_pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("资源清理完毕。")

    return {
        "WER": wer_score,
        "CER": cer_score,
        "Avg Inference Time (s)": avg_inference_time
    }

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
    try:
        # 构建 split 表达式
        split_expr = f"{TEST_SPLIT_NAME}"
        if NUM_TEST_SAMPLES is not None:
            split_expr += f"[:{NUM_TEST_SAMPLES}]" # 使用切片语法选择样本

        # 加载数据集
        test_dataset = load_dataset(
            SOURCE_DATASET_NAME,
            LANGUAGE_CODE,
            split=split_expr,
            use_auth_token=HUGGING_FACE_TOKEN,
            trust_remote_code=True, # Common Voice 需要设置 True
        )

        # 确认音频列存在并转换为 16kHz
        if AUDIO_COLUMN not in test_dataset.column_names:
             raise ValueError(f"音频列 '{AUDIO_COLUMN}' 在数据集中未找到。可用列: {test_dataset.column_names}")
        test_dataset = test_dataset.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))

        # 确认文本列存在
        if TEXT_COLUMN not in test_dataset.column_names:
            raise ValueError(f"文本列 '{TEXT_COLUMN}' 在数据集中未找到。可用列: {test_dataset.column_names}")

        print("数据集加载和准备成功:")
        print(test_dataset)
        actual_num_samples = len(test_dataset) # 获取实际加载的样本数
        print(f"实际测试样本数: {actual_num_samples}")

    except Exception as e:
        print(f"!!! 加载或准备数据集失败: {e}")
        return

    # 2. 依次评估每个模型
    results = {}
    models_to_test = [MODEL_OFFICIAL, MODEL_STUDENT]

    for model_id in models_to_test:
        # 注意：每次评估都会重新加载数据集的一个副本（如果map修改了它）
        # 或者确保 evaluate_model 不会永久修改传入的 dataset 对象
        # 这里我们假设 evaluate_model 使用 map(remove_columns=...)，所以需要传递原始数据集的引用
        # 如果担心 dataset 被修改，可以在循环开始时重新加载或复制
        current_dataset_instance = test_dataset # 传递数据集引用
        results[model_id] = evaluate_model(
            model_id=model_id,
            dataset=current_dataset_instance,
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
    print(f"测试样本数: {actual_num_samples}")
    print("-" * 70)
    # 使用 f-string 对齐打印表头
    print(f"{'Model':<35} | {'WER (%)':<10} | {'CER (%)':<10} | {'Avg Inf Time/Sample (s)':<25}")
    print("-" * 70)

    for model_id, metrics in results.items():
        # 检查 NaN 值，使用 math.isnan
        wer_str = f"{metrics['WER']:.2f}" if not math.isnan(metrics['WER']) else "N/A"
        cer_str = f"{metrics['CER']:.2f}" if not math.isnan(metrics['CER']) else "N/A"
        time_str = f"{metrics['Avg Inference Time (s)']:.4f}" if not math.isnan(metrics['Avg Inference Time (s)']) else "N/A"
        # 使用 f-string 对齐打印结果行
        print(f"{model_id:<35} | {wer_str:<10} | {cer_str:<10} | {time_str:<25}")

    print("-" * 70)
    print("测试完成.")


if __name__ == "__main__":
    # 设置 Tokenizers 并行化环境变量 (推荐)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 忽略特定类型的警告 (可选)
    warnings.filterwarnings("ignore", message="Using the update method is deprecated")
    warnings.filterwarnings("ignore", category=UserWarning, module='datasets.arrow_dataset')

    # 打印提示信息
    print("重要提示：")
    print("1. 如果 'kiritan/iteboshi' 是私有模型, 请确保已通过 'huggingface-cli login' 登录。")
    print("2. Common Voice 数据集下载可能需要一些时间，并且需要 'trust_remote_code=True'。")
    print(f"3. 将使用设备: {DEVICE}。GPU 推理速度远快于 CPU。")
    print(f"4. Pipeline Batch Size: {PIPELINE_BATCH_SIZE}, Map Batch Size: {MAP_BATCH_SIZE} (可根据显存调整)。")

    # 运行主函数
    main()