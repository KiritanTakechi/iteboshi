# train_student_model.py

import os
import torch
import evaluate # 使用 evaluate 库计算 WER
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import wandb

from datasets import load_dataset, DatasetDict
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from huggingface_hub import create_repo # 用于预先创建仓库 (可选)

# --- 配置参数 ---
# 你在 Hugging Face Hub 上的用户名或组织名
hf_username = "kiritan" # 同上
# 包含 Log-Mel 特征的数据集仓库名称
hf_logmel_repo_name = "iteboshi.logmel" # 同第 3 步
# 学生模型所在的本地路径 (来自第 4 步) 或 Hugging Face Hub ID
student_model_path = "./iteboshi"
# 用于初始化 Processor (Tokenizer+FeatureExtractor) 的 Whisper checkpoint
processor_checkpoint = "openai/whisper-base"
# 训练输出目录 (本地临时保存)
training_output_dir = "./iteboshi_temp" # 可以是临时目录
# **新增：上传到 Hub 的学生模型仓库名称**
hf_student_repo_name = "iteboshi" # 自定义你的模型名称

wandb_project_name = "iteboshi"

# 训练参数
per_device_train_batch_size = 32  # 每个设备的训练批次大小
per_device_eval_batch_size = 24   # 每个设备的评估批次大小
gradient_accumulation_steps = 1   # 梯度累积步数 (有效批次大小 = train_batch_size * grad_acc_steps)
learning_rate = 2e-5              # 学习率
warmup_steps = 500                # 预热步数
max_steps = 4000                  # 最大训练步数
gradient_checkpointing = True     # 是否启用梯度检查点 (节省显存)
bf16 = True                       # 是否启用混合精度训练 (需要兼容 GPU)
evaluation_strategy = "steps"     # 评估策略
eval_steps = 1000                 # 每隔多少步评估一次
save_steps = 1000                 # 每隔多少步保存一次 checkpoint
logging_steps = 25                # 每隔多少步记录一次日志 (包括 wandb)
hf_token = True                   # Hugging Face Hub token (依赖 huggingface-cli login)
num_proc = max(1, os.cpu_count() * 3 // 4) # type: ignore
num_proc_tokenizer = max(1, os.cpu_count() * 3 // 4) # type: ignore

# 这里根据一些关键参数自动生成一个名字
wandb_run_name = f"学生_{student_model_path.split('/')[-1]}_lr{learning_rate}_有效批次{per_device_train_batch_size*gradient_accumulation_steps}"

# --- 数据收集器 ---
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 分离输入特征和标签
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # 使用 feature_extractor 对 input_features 进行填充
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # 使用 tokenizer 对 labels 进行填充
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 将标签中的 padding token (通常是 tokenizer.pad_token_id) 替换为 -100，以便损失函数忽略它们
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # 如果解码器输入的开头是 bos token，移除它 (因为模型会自动添加)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        # 将处理后的标签添加到批次中
        batch["labels"] = labels

        return batch

# --- 主程序 ---
def main():
    print("开始训练学生模型（已集成 Weights & Biases）...")

    # **设置 WandB 环境变量 (推荐方式)**
    # 这会告诉 Trainer 将日志发送到哪个 wandb 项目
    os.environ["WANDB_PROJECT"] = wandb_project_name
    # os.environ["WANDB_LOG_MODEL"] = "checkpoint" # 可选: 将模型 checkpoint 作为 artifact 上传到 wandb (可能会占用较多存储空间)

    # --- 检查 GPU ---
    use_cuda = torch.cuda.is_available()
    global bf16, gradient_checkpointing # 允许修改全局变量
    if not use_cuda:
        print("警告：未检测到 CUDA GPU。训练将在 CPU 上进行，速度会非常慢。")
        bf16 = False
        # gradient_checkpointing 在 CPU 上可能无效或报错，保险起见禁用
        # gradient_checkpointing = False
        print("已禁用 bf16 混合精度训练。")
    else:
        print(f"检测到 CUDA GPU: {torch.cuda.get_device_name(0)}")
        # 检查显存大小，如果显存非常小，即使有 GPU 也可能需要禁用 bf16 或 gradient_checkpointing
        # gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        # if gpu_mem_gb < 10: # 示例：如果显存小于 10GB
        #     print(f"警告：GPU 显存 ({gpu_mem_gb:.1f}GB) 可能较小，如果遇到 OOM 问题，请尝试减小批次大小或禁用 bf16/gradient_checkpointing。")


    # 1. 加载 Processor (Tokenizer + Feature Extractor)
    print(f"加载 Processor: {processor_checkpoint}")
    try:
        # 明确指定语言和任务，这些信息会随模型一起保存和上传
        processor = WhisperProcessor.from_pretrained(processor_checkpoint, language="Chinese", task="transcribe")
        processor.tokenizer.set_prefix_tokens(language="chinese", task="transcribe") # 再次确认设置 # type: ignore
        print(f"Processor 加载成功。语言: {processor.tokenizer.language}, 任务: {processor.tokenizer.task}") # type: ignore
    except Exception as e:
        print(f"加载 Processor 失败: {e}")
        return

    # 2. 加载 Log-Mel 特征数据集
    logmel_repo_id = f"{hf_username}/{hf_logmel_repo_name}"
    print(f"加载 Log-Mel 数据集: {logmel_repo_id}")
    try:
        # 定义明确要加载的 splits
        splits_to_load = ["train", "validation"] # 可以根据需要加入 "test"
        dataset = DatasetDict()
        for split in splits_to_load:
            try:
                # 使用 split 参数加载特定的 split
                dataset[split] = load_dataset(logmel_repo_id, split=split)
                print(f"成功加载 split: {split} (大小: {len(dataset[split])})")
            except ValueError as e:
                print(f"加载 split '{split}' 失败 (可能不存在于仓库中): {e}")

            # 检查是否成功加载了 'train' 和 'validation'
        if "train" not in dataset:
            print(f"错误：未能从仓库 {logmel_repo_id} 加载 'train' split。无法继续训练。")
            return
        if "validation" not in dataset:
            print(f"警告：未能从仓库 {logmel_repo_id} 加载 'validation' split。将无法进行训练中评估。")
            # 或者直接 return 如果验证集是必需的

        print("Log-Mel 数据集所需 splits 加载成功:")
        print(dataset) # 现在只包含明确加载的 splits
    except Exception as e:
        print(f"加载 Log-Mel 数据集失败: {e}")
        print("请确保您已通过 'huggingface-cli login' 登录，并且该仓库存在且您有读取权限。")
        return

    # 3. Tokenize 文本标签 ('sentence' 列)
    print("Tokenizing 文本标签...")
    def prepare_labels(batch):
        # 检查 'sentence' 列是否存在
        if "sentence" not in batch:
             print("警告：批次中未找到 'sentence' 列。")
             return batch
        # 确保 'sentence' 列是字符串列表
        sentences = batch["sentence"]
        if not isinstance(sentences, list): sentences = [sentences]
        # 处理可能存在的 None 或非字符串数据
        sentences = [str(s) if s is not None else "" for s in sentences]

        # 使用 tokenizer 处理文本，不在 map 中进行填充 (padding)，由 DataCollator 处理
        batch["labels"] = processor.tokenizer(sentences, padding=False, truncation=True).input_ids # type: ignore
        return batch

    try:
        # 确定要移除的列：移除所有列，除了 'input_features' 和（临时）'sentence'
        columns_to_remove = [col for col in dataset["train"].column_names if col != 'input_features']
        if 'sentence' in columns_to_remove: columns_to_remove.remove('sentence') # 确保 sentence 在处理前不被移除

        dataset = dataset.map(
            prepare_labels,
            batched=True,
            num_proc=num_proc_tokenizer,
            remove_columns=columns_to_remove # 移除不再需要的列，包括处理完的 'sentence' 列
        )

        print("文本标签 Tokenization 完成。")
        print("处理后的数据集结构 (包含 input_features 和 labels):")
        print(dataset)
    except Exception as e:
        print(f"Tokenizing 标签时出错: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. 实例化数据收集器
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 5. 加载评估指标 (WER - 字错误率)
    print("加载 WER 评估指标...")
    try:
        metric = evaluate.load("wer")
    except Exception as e:
        print(f"加载 WER 指标失败: {e}")
        return

    # 6. 定义计算指标的函数 (用于评估)
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # 将标签中的 -100 替换回 pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id # type: ignore

        # 使用 processor 解码预测和标签 ID
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True) # type: ignore
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True) # type: ignore

        # 计算 WER
        try:
            wer = 100 * metric.compute(predictions=pred_str, references=label_str) # type: ignore
        except Exception as e:
            print(f"计算 WER 时出错: {e}")
            print("预测样本:", pred_str[:2])
            print("参考样本:", label_str[:2])
            wer = float('inf') # 返回一个无效值

        return {"wer": wer}

    # 7. 加载学生模型
    print(f"加载学生模型: {student_model_path}")
    try:
        model = WhisperForConditionalGeneration.from_pretrained(
            student_model_path,
            # low_cpu_mem_usage=True, # 如果加载模型时 CPU 内存不足，可以尝试开启
        )
        # **重要：设置解码相关的配置，这些配置会随模型一起保存和上传**
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="chinese", task="transcribe") # type: ignore
        model.config.suppress_tokens = [] # 这里可以添加需要抑制生成的 token ID 列表
        # 确保在模型加载后启用梯度检查点 (如果配置为 True)
        if gradient_checkpointing:
            # 检查模型是否支持梯度检查点
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                print("梯度检查点已启用。")
            else:
                 print("警告：模型不支持 gradient_checkpointing_enable 方法。")

        generation_max_len = getattr(model.config, 'max_target_positions', None) # 尝试获取
        if generation_max_len is None:
            generation_max_len = getattr(model.config, 'max_decoder_position_embeddings', 448) # 备选或默认值
            print(f"警告：未找到 'max_target_positions'，使用 'max_decoder_position_embeddings' 或默认值: {generation_max_len}")
        else:
            print(f"从模型配置获取 generation_max_length: {generation_max_len}")


        print("学生模型加载成功。")
    except Exception as e:
        print(f"加载学生模型失败: {e}")
        return

    # 8. 配置训练参数
    print("配置训练参数 (已集成 WandB)...")
    hub_model_id = f"{hf_username}/{hf_student_repo_name}" # Hugging Face Hub 上的完整仓库 ID
    print(f"训练完成后模型将上传到 Hugging Face Hub: {hub_model_id}")
    print(f"训练过程将记录到 WandB 项目: '{wandb_project_name}', 运行名称: '{wandb_run_name}'")

    # (可选) 尝试预先创建 Hugging Face Hub 仓库，如果不存在
    try:
        create_repo(hub_model_id, exist_ok=True, token=hf_token if isinstance(hf_token, str) else None)
        print(f"Hugging Face Hub 仓库 {hub_model_id} 已确认存在或已创建。")
    except Exception as e:
        print(f"警告：创建或检查 Hub 仓库时出错: {e}。如果仓库已存在且你有写入权限，训练仍可继续并尝试上传。")


    training_args = Seq2SeqTrainingArguments(
        output_dir=training_output_dir, # 本地输出目录 (保存 checkpoint 和最终模型)
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        gradient_checkpointing=gradient_checkpointing, # 再次确认应用 (虽然已在模型上启用)
        bf16=bf16,
        eval_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        save_strategy=evaluation_strategy, # 保存策略与评估策略一致
        save_steps=save_steps,
        logging_steps=logging_steps,       # 日志记录频率 (包括 wandb)
        report_to=["wandb"], # 记录到 WandB
        run_name=wandb_run_name,            # 指定 wandb 上的运行名称
        load_best_model_at_end=True,        # 训练结束后加载评估指标最佳的模型
        metric_for_best_model="wer",        # 用于选择最佳模型的指标 (WER 越低越好)
        greater_is_better=False,            # 指标值是否越大越好 (WER 是越小越好)
        predict_with_generate=True,         # 必须为 Seq2Seq 任务设置为 True，以便在评估时生成文本
        generation_max_length=generation_max_len, # 设置生成文本的最大长度
        push_to_hub=True,                   # 是否在保存 checkpoint 或训练结束时推送到 Hub
        hub_model_id=hub_model_id,          # 推送到的 Hub 仓库 ID
        hub_strategy="checkpoint",          # 推送策略: "end", "checkpoint", "all_checkpoints"
                                            # "checkpoint": 保存 checkpoint 时推送
                                            # "end": 只在训练结束时推送最佳模型
        hub_token=hf_token if isinstance(hf_token, str) else None, # 传递 Hub token
        save_total_limit=2,                 # 最多保留最近的几个 checkpoint
        # **可选 WandB 特定参数 (通常不需要在这里设置，通过环境变量或 wandb init 更好)**
        # wandb_project=wandb_project_name, # 也可以在这里设置项目名，但环境变量优先
    )

    # 9. 实例化 Trainer
    # Trainer 会自动检测到 training_args 中的 'wandb' 并进行初始化
    print("实例化 Seq2SeqTrainer...")
    trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"), # 使用 .get 以防数据集中没有 'validation' split
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            processing_class=processor, # 这里需要传入 processor 用于日志记录和可能的输入处理 # type: ignore
        )

    # 10. 开始训练
    print("开始训练...")
    try:
        # 如果需要从之前的 checkpoint 恢复训练:
        # train_result = trainer.train(resume_from_checkpoint=True) # 自动寻找最新的 checkpoint
        # train_result = trainer.train(resume_from_checkpoint='/path/to/specific/checkpoint') # 指定 checkpoint 路径
        train_result = trainer.train()

        print("训练完成。")

        # 保存最终的模型状态和指标到本地 output_dir (Trainer 默认会做)
        # trainer.save_model() # 如果 load_best_model_at_end=True, Trainer 会自动保存最佳模型
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics) # 记录训练最终指标 (也会发送到 wandb)
        trainer.save_metrics("train", metrics) # 保存训练最终指标到本地 json 文件
        trainer.save_state() # 保存 Trainer 状态 (包括随机种子等)
        print(f"最终训练状态和指标已保存到本地目录: {training_output_dir}")

        # 评估测试集 (可选)
        if "test" in dataset:
            if dataset.get("test"): # 确保 test split 存在且不为空
                print("评估测试集...")
                test_metrics = trainer.evaluate(eval_dataset=dataset["test"]) # type: ignore
                trainer.log_metrics("test", test_metrics) # 记录测试集指标 (也会发送到 wandb)
                trainer.save_metrics("test", test_metrics) # 保存测试集指标到本地 json 文件
            else:
                 print("数据集中 'test' split 为空，跳过测试集评估。")
        else:
            print("数据集中未找到 'test' split，跳过测试集评估。")

        # **手动触发一次最终上传到 Hub (可选, 作为保险)**
        # 如果 hub_strategy='end'，或者你想确保最终的最佳模型被明确推送
        print("准备将最终/最佳模型推送到 Hugging Face Hub...")
        try:
            # 这会上传 output_dir 中的模型文件、配置文件、tokenizer 文件等
            # 如果 load_best_model_at_end=True, 上传的是在验证集上表现最佳的模型
            trainer.push_to_hub(commit_message="训练结束，上传最终模型")
            print(f"模型成功上传到 Hugging Face Hub: {hub_model_id}")
        except Exception as e:
            print(f"上传模型到 Hub 时出错: {e}")
            print(f"你可以稍后手动上传保存在 {training_output_dir} 的模型文件。")

    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc() # 打印详细的错误堆栈信息
    finally:
        # **确保 wandb 运行结束** (虽然 Trainer 通常会处理，但加上更保险)
        # 检查 wandb 是否已被初始化
        if wandb.run is not None:
            wandb.finish()
            print("WandB 运行已结束。")

if __name__ == "__main__":
    print("重要提示：请确保已运行 'pip install wandb' 并通过 'wandb login' 登录。")
    print("重要提示：请确保您已通过 'huggingface-cli login' 使用具有写入权限的 token 登录 Hugging Face Hub。")
    main()
