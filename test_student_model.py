# test_student_model.py

import torch
import evaluate
from datasets import load_dataset, Audio, DatasetDict
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import time
import warnings

# --- 配置参数 ---
hf_username = "kiritan" # 你的用户名
hf_student_repo_name = "iteboshi" # 你上传的模型名称
trained_model_path = f"{hf_username}/{hf_student_repo_name}" # Hub ID

# processor_checkpoint = "openai/whisper-base" # 不再需要单独指定基础 checkpoint

# 原始数据集名称和语言 (用于加载测试集的原始音频和文本)
source_dataset_name = "mozilla-foundation/common_voice_11_0"
language_code = "zh-CN"
# 测试集的 split 名称
test_split_name = "test"
# 测试样本数量 (None 表示使用整个测试集)
num_test_samples = 100
# Hugging Face token (加载公共模型通常不需要，但私有模型需要)
hf_token = True # 或者 None
# 推理设备 ('cuda:0', 'cpu', etc.)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# 推理批处理大小 (使用 pipeline 时)
pipeline_batch_size = 16

# --- 主程序 ---
def main():
    print(f"开始测试 Hugging Face Hub 上的模型: {trained_model_path}")
    print(f"使用设备: {device}")

    # 1. 加载 Processor 和训练好的学生模型
    print(f"从 Hub 加载 Processor 和模型: {trained_model_path}...")
    try:
        # 从训练好的模型仓库加载 Processor，它包含了正确的 tokenizer 和 feature extractor 配置
        processor = WhisperProcessor.from_pretrained(
            trained_model_path,
        )
        # 从训练好的模型仓库加载模型
        model = WhisperForConditionalGeneration.from_pretrained(
            trained_model_path,
        ).to(device) # type: ignore

        model.eval() # 设置为评估模式

        # 确认 Processor 的语言和任务设置是否正确加载
        print(f"Processor 加载成功。语言: {processor.tokenizer.language}, 任务: {processor.tokenizer.task}") # type: ignore
        # 确认模型配置中的解码设置
        print(f"模型配置加载成功。Forced decoder IDs length: {len(model.config.forced_decoder_ids) if model.config.forced_decoder_ids else 0}")

    except Exception as e:
        print(f"从 Hub 加载 Processor 或模型失败: {e}")
        print("请确保:")
        print(f"  1. 模型仓库 '{trained_model_path}' 存在。")
        print("  2. 如果仓库是私有的，你已通过 'huggingface-cli login' 登录或提供了有效的 'hf_token'。")
        print("  3. 仓库中包含必要的模型文件 (pytorch_model.bin), 配置文件 (config.json), 以及 processor 文件 (preprocessor_config.json, vocab.json, added_tokens.json 等)。")
        return

    # 2. 加载原始测试数据集 (与之前相同，注意 trust_remote_code)
    print(f"加载原始数据集 '{source_dataset_name}' ({language_code}) 的 '{test_split_name}' split...")
    try:
        split_expr = f"{test_split_name}"
        if num_test_samples is not None:
             split_expr += f"[:{num_test_samples}]"

        test_dataset_raw: DatasetDict = load_dataset(
            source_dataset_name,
            language_code,
            split=split_expr,
            trust_remote_code=True, # Common Voice 需要
        ) # type: ignore

        test_dataset_raw = test_dataset_raw.cast_column("audio", Audio(sampling_rate=16000))
        print("测试数据集加载成功:")
        print(test_dataset_raw)
    except Exception as e:
        print(f"加载测试数据集失败: {e}")
        return

    # 3. 使用 pipeline 进行推理
    print("方法 1: 使用 Transformers Pipeline 进行推理...")
    try:
        # Pipeline 现在直接使用从 Hub 加载的 model 和 processor
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model, # 直接使用加载的模型对象
            tokenizer=processor.tokenizer, # type: ignore
            feature_extractor=processor.feature_extractor, # type: ignore
            device=device,
            chunk_length_s=30,
            batch_size=pipeline_batch_size,
            # **重要：传递解码参数给 pipeline**
            generate_kwargs={
                "language": processor.tokenizer.language, # 使用 processor 中加载的语言 # type: ignore
                "task": processor.tokenizer.task,         # 使用 processor 中加载的任务 # type: ignore
                # forced_decoder_ids 和 suppress_tokens 通常由模型配置自动处理
                # 如果 pipeline 未正确应用，可以在这里强制指定：
                # "forced_decoder_ids": model.config.forced_decoder_ids,
                # "suppress_tokens": model.config.suppress_tokens,
            }
        )
        print("Pipeline 创建成功。")
    except Exception as e:
        print(f"创建 Pipeline 失败: {e}")
        asr_pipeline = None

    predictions = []
    references = []
    if asr_pipeline:
        start_time = time.time()
        print(f"开始对 {len(test_dataset_raw)} 个样本进行转录...")
        warnings.filterwarnings("ignore", category=UserWarning, module='datasets.arrow_dataset')

        # 迭代处理或使用 map (与之前相同)
        try:
            # 使用 map 进行批量处理
            # 注意 pipeline 的 batch_size 和 map 的 batch_size 是不同的概念
            # 这里 map 的 batch_size 控制一次传递多少样本给 map function
            # pipeline 的 batch_size 控制 pipeline 内部一次处理多少样本
            results = test_dataset_raw.map(
                lambda batch: {"prediction": asr_pipeline([x["array"] for x in batch["audio"]])},
                batched=True,
                batch_size=pipeline_batch_size * 2 # map 的 batch size 可以稍大
            )
            # pipeline 返回的是字典列表，需要提取 'text'
            predictions = [item['text'] if isinstance(item, dict) and 'text' in item else '' for item in results["prediction"]]
            references = results["sentence"]
        except Exception as e:
             print(f"使用 map 和 pipeline 进行批量处理时出错: {e}")
             print("将回退到逐个处理...")
             predictions = []
             references = []
             for item in test_dataset_raw:
                try:
                    # Pipeline 输入可以是 numpy array
                    audio_input = item["audio"]["array"]
                    result = asr_pipeline(audio_input)
                    predictions.append(result["text"]) # type: ignore
                    references.append(item["sentence"])
                except Exception as inner_e:
                    print(f"处理样本 {item.get('path', 'N/A')} 时出错: {inner_e}.")
                    predictions.append("") # 添加空预测以保持对齐
                    references.append(item["sentence"])

        warnings.resetwarnings()

        end_time = time.time()
        print(f"Pipeline 推理完成，耗时: {end_time - start_time:.2f} 秒")

        # 打印样本结果 (与之前相同)
        print("\n--- 样本转录对比 ---")
        num_samples_to_show = min(5, len(predictions))
        for i in range(num_samples_to_show):
             # 清理可能存在的换行符以便比较
             ref_clean = references[i].replace('\n', ' ').strip() # type: ignore
             pred_clean = predictions[i].replace('\n', ' ').strip()
             print(f"引用: {ref_clean}")
             print(f"预测: {pred_clean}\n")


        # 4. 计算 WER (与之前相同)
        print("计算 WER...")
        try:
            wer_metric = evaluate.load("wer")
            # 过滤掉空的引用或预测，避免影响 WER 计算 (可选，取决于评估策略)
            filtered_preds = [p for p, r in zip(predictions, references) if r]
            filtered_refs = [r for r in references if r]
            if len(filtered_preds) != len(filtered_refs):
                 print("警告：过滤后预测和引用的数量不匹配，可能存在问题。")

            if not filtered_refs:
                print("没有有效的引用文本用于计算 WER。")
                wer_score = float('nan')
            else:
                wer_score = 100 * wer_metric.compute(predictions=filtered_preds, references=filtered_refs) # type: ignore
                print(f"测试集 WER (基于 {len(filtered_refs)} 个有效样本): {wer_score:.2f}%")

        except Exception as e:
            print(f"计算 WER 时出错: {e}")

    else:
        print("Pipeline 创建失败，无法进行测试。")


if __name__ == "__main__":
    print("重要提示：如果模型仓库是私有的，请确保已通过 'huggingface-cli login' 登录。")
    main()
