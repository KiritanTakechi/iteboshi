# train_student_model.py

import os
import torch
import evaluate
from dataclasses import dataclass
from typing import Dict, List, Union, Optional
import wandb
import logging

# 明确导入所需组件
from datasets import load_dataset, DatasetDict
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from huggingface_hub import create_repo, HfFolder # 用于保存 token

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 配置参数 ---
# Hugging Face 配置
hf_username: str = "kiritan"

num_mel_bins = 80 # <--- 在这里选择 80 或 128

hf_logmel_repo_name: str = f"iteboshi.logmel.{num_mel_bins}bins"       # Log-Mel 特征数据集仓库
hf_student_repo_name: str = "iteboshi"             # 训练后上传的学生模型仓库
hf_token: Optional[str] = HfFolder.get_token()     # 尝试从缓存获取 token

# 模型与 Processor 配置
student_model_path: str = "./iteboshi"            # 初始学生模型路径
processor_checkpoint: str = "openai/whisper-medium" # 加载组件的源 checkpoint
model_language: str = "Chinese"                   # Whisper 指定的语言
model_task: str = "transcribe"                    # Whisper 指定的任务

# 训练超参数
training_output_dir: str = "./iteboshi_temp"       # 本地训练输出/缓存目录
per_device_train_batch_size: int = 4               # 显著减小以防止 OOM
per_device_eval_batch_size: int = 4                # 显著减小
gradient_accumulation_steps: int = 8               # 增加以维持有效批次大小 (8*4=32)
learning_rate: float = 2e-5
warmup_steps: int = 500
max_steps: int = 20000                             # 增加训练步数
gradient_checkpointing: bool = True
fp16: bool = False                                 # 推荐使用 fp16，更广泛支持
bf16: bool = True                                  # 除非确定需要且支持 bf16

# 评估与日志记录
evaluation_strategy: str = "steps"
eval_steps: int = 1000                             # 评估频率
save_steps: int = 1000                             # 保存频率
logging_steps: int = 25                            # 日志频率 (包括 loss 打印和 wandb)
save_total_limit: int = 2                          # 最多保留的 checkpoint 数量

# 数据处理配置
num_proc_tokenizer: int = max(1, os.cpu_count() * 3 // 4) # Tokenizer map 使用的进程数 # type: ignore

# WandB 配置
wandb_project_name: str = "iteboshi" # WandB 项目名
wandb_run_name: str = f"student_{hf_student_repo_name}_lr{learning_rate}_bs{per_device_train_batch_size*gradient_accumulation_steps}_{'fp16' if fp16 else 'bf16' if bf16 else 'fp32'}"

# --- 数据收集器 ---
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor # Data Collator 接收 Processor 很方便

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 分离输入特征和标签
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # 使用 feature_extractor 对 input_features 进行填充
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt") # type: ignore

        # 使用 tokenizer 对 labels 进行填充
        # type: ignore #
        # 用于忽略 Pyright 可能对 processor.tokenizer 访问的警告
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt") # type: ignore

        # 将标签中的 padding token 替换为 -100
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # 如果解码器输入的开头是 bos token，移除它
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item(): # type: ignore
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# --- Debug Trainer (用于检查 Labels) ---
class DebugTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        # 只在训练阶段且达到日志记录步数时打印，避免过多输出
        if model.training and labels is not None and self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
            try:
                logger.info("-" * 60)
                logger.info(f"Debugging Labels at step {self.state.global_step}:")
                logger.info(f"Labels shape: {labels.shape}")
                logger.info(f"Labels sample (first sample, first 50 tokens): {labels[0, :50].tolist()}") # 转为 list 打印
                num_non_ignored = (labels != -100).sum().item()
                total_tokens = labels.numel()
                logger.info(f"Number of non-ignored tokens in batch: {num_non_ignored}")
                logger.info(f"Total number of tokens in batch: {total_tokens}")
                if total_tokens > 0:
                    ratio = num_non_ignored / total_tokens
                    logger.info(f"Ratio of non-ignored tokens: {ratio:.4f}")
                    if ratio < 0.1: # 如果有效标签比例过低，发出警告
                         logger.warning(f"LOW non-ignored token ratio ({ratio:.4f}) at step {self.state.global_step}!")
                else:
                    logger.warning(f"Empty labels tensor encountered at step {self.state.global_step}!")
                logger.info("-" * 60)
            except Exception as e:
                logger.error(f"Error during label debugging at step {self.state.global_step}: {e}")

        # 调用原始的 compute_loss
        loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        # 也可以在这里打印 loss 值
        # if model.training and self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
        #     logger.info(f"Computed Loss at step {self.state.global_step}: {loss.item():.4f}")

        return loss

# --- 主程序 ---
def main():
    logger.info("开始训练学生模型 (明确使用 WhisperTokenizerFast)...")

    # 设置 WandB (推荐使用环境变量)
    os.environ["WANDB_PROJECT"] = wandb_project_name
    # os.environ["WANDB_LOG_MODEL"] = "checkpoint" # 上传模型到 wandb (可选)
    logger.info(f"WandB Project: {wandb_project_name}, Run Name: {wandb_run_name}")

    # --- 检查 GPU 和混合精度 ---
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        logger.warning("未检测到 CUDA GPU。训练将在 CPU 上进行，非常慢。")
        global fp16, bf16 # 允许修改全局变量
        fp16 = bf16 = False
        logger.warning("已禁用混合精度训练。")
    else:
        logger.info(f"检测到 CUDA GPU: {torch.cuda.get_device_name(0)}")
        if fp16:
            logger.info("使用 fp16 混合精度训练。")
        if bf16:
            logger.info("使用 bf16 混合精度训练。")


    # 1. 分别加载 Feature Extractor 和 TokenizerFast
    logger.info(f"从 '{processor_checkpoint}' 加载 Feature Extractor 和 Tokenizer...")
    try:
        feature_extractor: WhisperFeatureExtractor = WhisperFeatureExtractor.from_pretrained(processor_checkpoint)
        tokenizer: WhisperTokenizer = WhisperTokenizer.from_pretrained(
            processor_checkpoint, language=model_language, task=model_task
        )
        # 手动设置 prefix tokens (重要)
        tokenizer.set_prefix_tokens(language=model_language, task=model_task)

        # 组合成 Processor (主要为了方便传递给 DataCollator)
        processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

        logger.info(f"Feature Extractor type: {type(feature_extractor)}")
        logger.info(f"Tokenizer type: {type(tokenizer)}")
        logger.info(f"Tokenizer language: {tokenizer.language}, task: {tokenizer.task}")

    except ImportError:
         logger.error("加载 WhisperTokenizerFast 失败，请确保已安装 'tokenizers' 库 (`pip install tokenizers`)")
         return
    except Exception as e:
        logger.error(f"加载 Feature Extractor 或 TokenizerFast 失败: {e}", exc_info=True)
        return

    # 2. 加载 Log-Mel 特征数据集 (只加载 train/validation/test)
    logmel_repo_id = f"{hf_username}/{hf_logmel_repo_name}"
    logger.info(f"从仓库加载 Log-Mel 数据集的 'train' 'validation' 和 'test' splits: {logmel_repo_id}")
    try:
        splits_to_load = ["train", "validation", "test"]
        dataset = DatasetDict()
        for split in splits_to_load:
            try:
                dataset[split] = load_dataset(logmel_repo_id, split=split, token=hf_token) # 添加 token
                logger.info(f"成功加载 split: {split} (大小: {len(dataset[split])})")
            except ValueError as e: logger.warning(f"加载 split '{split}' 失败: {e}")
        if "train" not in dataset: raise ValueError(f"无法从 {logmel_repo_id} 加载 'train' split")
        logger.info(f"数据集加载成功: {dataset}")
    except Exception as e:
        logger.error(f"加载 Log-Mel 数据集时出错: {e}", exc_info=True)
        return

    # 3. Tokenize 文本标签
    logger.info("Tokenizing 文本标签...")
    min_sentence_len_chars = 3 # 设置一个最小句子字符长度阈值 (用于调试)
    def prepare_labels(batch):
        if "sentence" not in batch: return batch
        sentences = batch["sentence"]
        if not isinstance(sentences, list): sentences = [list]
        processed_sentences = [str(s).strip() if s is not None else "" for s in sentences]

        # 检查短句子/空句子
        short_sentences = [s for s in processed_sentences if len(s) < min_sentence_len_chars]
        if short_sentences:
            logger.debug(f"prepare_labels 遇到 {len(short_sentences)} 个短句子 (少于 {min_sentence_len_chars} 字符): {short_sentences[:5]}...") # 只显示前5个

        # 使用加载的 tokenizer 对象
        batch["labels"] = tokenizer(processed_sentences, padding=False, truncation=True).input_ids

        # 检查空的 Tokenizer 输出
        empty_labels_indices = [i for i, lbl_list in enumerate(batch["labels"]) if not lbl_list]
        if empty_labels_indices:
             problematic_sentences = [processed_sentences[i] for i in empty_labels_indices]
             logger.debug(f"Tokenizer 为 {len(empty_labels_indices)} 个句子产生了空标签列表: {problematic_sentences[:5]}...")

        return batch

    try:
        # 确定要移除的列
        column_names = dataset["train"].column_names
        columns_to_remove = [col for col in column_names if col not in ['input_features', 'sentence']]

        tokenized_dataset = dataset.map(
            prepare_labels,
            batched=True,
            num_proc=num_proc_tokenizer,
            remove_columns=columns_to_remove,
            desc="Tokenizing labels" # 添加描述
        )
        logger.info("文本标签 Tokenization 完成。")
        logger.info(f"Tokenized 数据集: {tokenized_dataset}")

        # # **(关键) 添加过滤步骤，移除有效标签过少的样本**
        # min_effective_label_len = 3 # 设置一个最小有效标签 token 数量（不含特殊 token 和 padding）
        # logger.info(f"开始过滤有效标签长度小于 {min_effective_label_len} 的样本...")

        # def has_enough_labels(example):
        #     # 移除特殊 tokens 列表需要根据具体 tokenizer 确认，这里简单检查非 -100 的数量
        #     # 更精确的方式是解码后检查长度，但会慢
        #     return sum(1 for token_id in example['labels'] if token_id != -100 and token_id < tokenizer.vocab_size) >= min_effective_label_len

        # filtered_dataset = DatasetDict()
        # for split_name, ds in tokenized_dataset.items():
        #     original_size = len(ds)
        #     filtered_dataset[split_name] = ds.filter(
        #         has_enough_labels,
        #         num_proc=num_proc_tokenizer, # 使用多进程加速过滤
        #         desc=f"Filtering {split_name} labels"
        #     )
        #     removed_count = original_size - len(filtered_dataset[split_name])
        #     logger.info(f"过滤后 {split_name} 集大小: {len(filtered_dataset[split_name])} (移除了 {removed_count} 个样本)")
        #     if removed_count > original_size * 0.5: # 如果移除了超过一半，发出警告
        #          logger.warning(f"注意：超过 50% 的样本在 {split_name} split 中被移除，请检查数据质量和过滤阈值！")

        # # 检查过滤后是否还有数据
        # if not filtered_dataset["train"]:
        #     logger.error("错误：过滤后训练集为空！请检查数据或降低过滤阈值。")
        #     return

        # tokenized_dataset = filtered_dataset # 使用过滤后的数据集

    except Exception as e:
        logger.error(f"Tokenizing 时出错: {e}", exc_info=True)
        return

    # 4. 实例化数据收集器
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    logger.info("Data Collator 实例化完成。")

    # --- 5. 加载评估指标 (WER 和 CER) ---
    logger.info("-" * 30 + " 步骤 5: 加载评估指标 " + "-" * 30)
    try:
        # 加载 WER 指标
        wer_metric = evaluate.load("wer")
        logger.info("WER 指标加载成功。")
        # 加载 CER 指标
        cer_metric = evaluate.load("cer")
        logger.info("CER 指标加载成功。")
    except Exception as e:
        logger.error(f"加载评估指标失败: {e}", exc_info=True)
        return

    # 6. 定义计算指标的函数 (使用加载的 tokenizer)
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str) # type: ignore
        cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str) # type: ignore
        return {"wer": wer, "cer": cer}
    logger.info("Compute Metrics 函数定义完成。")

    # 7. 加载学生模型 (使用加载的 tokenizer 设置 config)
    logger.info(f"加载学生模型: {student_model_path}")
    try:
        model: WhisperForConditionalGeneration = WhisperForConditionalGeneration.from_pretrained(student_model_path)

        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=model_language, task=model_task)
        model.config.suppress_tokens = [] # 根据需要添加
        if gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'): model.gradient_checkpointing_enable(); logger.info("梯度检查点已启用。")
            else: logger.warning("模型不支持 gradient_checkpointing_enable 方法。")
        # 获取 generation_max_length
        generation_max_len = getattr(model.config, 'max_length', None) # 优先尝试 max_length
        if generation_max_len is None:
            generation_max_len = getattr(model.config, 'max_target_positions', None)
        if generation_max_len is None:
            generation_max_len = getattr(model.config, 'max_decoder_position_embeddings', 448)
            logger.warning(f"使用备选或默认 generation_max_length: {generation_max_len}")
        else:
            logger.info(f"获取 generation_max_length: {generation_max_len}")

        logger.info(f"学生模型加载成功。模型位于: {model.device}") # Trainer 会自动处理设备
    except Exception as e: logger.error(f"加载学生模型失败: {e}", exc_info=True); return

    # 8. 配置训练参数
    logger.info("配置训练参数...")
    hub_model_id = f"{hf_username}/{hf_student_repo_name}"
    try:
        create_repo(hub_model_id, exist_ok=True, token=hf_token)
        logger.info(f"Hugging Face Hub 仓库 {hub_model_id} 已确认存在或创建。")
    except Exception as e: logger.warning(f"创建或检查 Hub 仓库时出错: {e}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=training_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        gradient_checkpointing=gradient_checkpointing,
        fp16=fp16, # 使用 fp16
        bf16=bf16, # 使用 bf16
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        save_strategy=evaluation_strategy,
        save_steps=save_steps,
        logging_steps=logging_steps,
        report_to=["wandb"],
        run_name=wandb_run_name,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=generation_max_len,
        push_to_hub=True,
        hub_model_id=hub_model_id,
        hub_strategy="checkpoint", # 每 save_steps 上传一次 checkpoint
        hub_token=hf_token,      # 传递 token
        save_total_limit=save_total_limit,
        dataloader_num_workers=4, # 增加数据加载进程 (根据 CPU 调整)
        remove_unused_columns=False, # DataCollator 会处理，设为 False 避免警告或错误
    )
    logger.info(f"训练参数配置完成: {training_args}")

    # 9. 实例化 Trainer (使用 DebugTrainer, 传入 feature_extractor)
    logger.info("实例化 DebugTrainer...")
    trainer = DebugTrainer(
            args=training_args,
            model=model,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset.get("validation"),
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            processing_class=processor,
        )
    logger.info("Trainer 实例化完成。")

    # 10. 开始训练
    logger.info("开始训练...")
    try:
        logger.info(f"训练集样本数 (过滤后): {len(tokenized_dataset['train'])}")
        if tokenized_dataset.get("validation"):
            logger.info(f"验证集样本数 (过滤后): {len(tokenized_dataset['validation'])}")

        train_result = trainer.train(resume_from_checkpoint=False) # 可以设置为 True 尝试恢复

        logger.info("训练完成。")
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info(f"最终训练状态和指标已保存到本地目录: {training_output_dir}")

        # 评估测试集 (如果你加载并处理了 "test" split)
        # if "test" in tokenized_dataset and tokenized_dataset.get("test"):
        #     logger.info("评估测试集...")
        #     test_metrics = trainer.evaluate(eval_dataset=tokenized_dataset["test"], metric_key_prefix="test")
        #     trainer.log_metrics("test", test_metrics)
        #     trainer.save_metrics("test", test_metrics)

        logger.info("准备将最终/最佳模型转换为 FP16 并推送到 Hugging Face Hub...")

        # trainer.model 现在是训练结束时加载的最佳模型（如果 load_best_model_at_end=True）
        # 或者训练结束时的最后一个模型
        logger.info(f"当前模型精度: {trainer.model.dtype}")
        if trainer.model.dtype != torch.float16:
            logger.info("将模型转换为 FP16...")
            try:
                trainer.model = trainer.model.half()
                # 转换后最好将模型移回 CPU，因为 push_to_hub 内部的 save_pretrained 在 CPU 上执行更安全
                # trainer.model.to('cpu') # 移到 CPU
                logger.info(f"模型已成功转换为 FP16 (dtype: {trainer.model.dtype})") # 并移至 {trainer.model.device}
            except Exception as e:
                logger.error(f"模型转换为 FP16 失败: {e}", exc_info=True)
                logger.warning("将尝试以上传原始精度的模型。")


        logger.info("准备将最终/最佳模型推送到 Hugging Face Hub...")
        try:
            trainer.push_to_hub(commit_message="训练结束，上传最终模型")


            logger.info("保存 FP16 模型对应的 Processor 到 Hub...")
            processor.push_to_hub(hub_model_id, commit_message="上传 FP16 模型对应的 Processor")

            logger.info(f"Processor 也已上传到 {hub_model_id}")

            logger.info(f"模型成功上传到 Hugging Face Hub: {hub_model_id}")
        except Exception as e:
            logger.error(f"上传模型到 Hub 时出错: {e}", exc_info=True)
            # 如果上传失败，尝试在本地保存 FP16 模型
            try:
                final_fp16_save_path = os.path.join(training_args.output_dir, "final_model_fp16")
                logger.warning(f"Hub 上传失败，尝试将 FP16 模型保存在本地: {final_fp16_save_path}")
                # trainer.save_model 会保存 trainer.model，此刻它已经是 FP16 了
                trainer.save_model(final_fp16_save_path)
                # 同时保存 processor 配置
                processor.save_pretrained(final_fp16_save_path)
                logger.info(f"FP16 模型和 Processor 已保存在本地: {final_fp16_save_path}")
            except Exception as save_e:
                logger.error(f"在 Hub 上传失败后，本地保存 FP16 模型也失败了: {save_e}", exc_info=True)

    except Exception as e:
        logger.error(f"训练过程中发生严重错误: {e}", exc_info=True)
    finally:
        if wandb.run is not None:
            wandb.finish()
            logger.info("WandB 运行已结束。")


if __name__ == "__main__":
    logger.info("运行训练脚本...")
    if not hf_token:
        logger.warning("未找到 Hugging Face Hub token。可能无法加载私有数据或上传模型。请运行 'huggingface-cli login'")
    main()
    logger.info("脚本执行完毕。")
