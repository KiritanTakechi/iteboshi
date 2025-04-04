# load_upload_raw_dataset.py

from datasets import load_dataset, DatasetDict
from huggingface_hub import create_repo

# --- 配置参数 ---
# Hugging Face Hub 上的源数据集名称
source_dataset_name = "mozilla-foundation/common_voice_11_0"
# 数据集语言代码
language_code = "zh-CN"
# 你在 Hugging Face Hub 上的用户名或组织名
hf_username = "kiritan"
# 你想创建的用于存储原始数据集的仓库名称
hf_raw_repo_name = "iteboshi.raw"
# 使用的身份验证令牌（如果 huggingface-cli login 配置好了，可以设为 True 或 None）
hf_token = True

# --- 主程序 ---

def main():
    print("开始加载 Common Voice 中文数据集...")
    # 加载指定语言的数据集，这里包含 train, validation, test 等 split
    # cache_dir 可以指定缓存位置，如果磁盘空间有限的话
    try:
        dataset: DatasetDict = load_dataset(source_dataset_name, language_code, trust_remote_code=True) # type: ignore
        print("数据集加载成功:")
        print(dataset)
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        print("请确保数据集名称和语言代码正确，并且网络连接正常。")
        return

    # 组合完整的仓库 ID
    target_repo_id = f"{hf_username}/{hf_raw_repo_name}"

    print(f"准备将数据集上传到 Hugging Face Hub: {target_repo_id}")

    # （可选）如果仓库不存在，可以尝试创建它
    try:
        create_repo(target_repo_id, repo_type="dataset", exist_ok=True, token=hf_token if isinstance(hf_token, str) else None)
        print(f"仓库 {target_repo_id} 已确认存在或已创建。")
    except Exception as e:
        print(f"创建或检查仓库时出错: {e}")
        print("请确保你的 Hugging Face token 有创建仓库的权限。")
        # 如果只是上传到现有仓库，可以忽略此错误，但上传可能会失败

    # 将数据集上传到 Hub
    # push_to_hub 会自动处理分块上传，相对内存友好
    try:
        print("开始上传数据集...")
        dataset.push_to_hub(target_repo_id, token=hf_token if isinstance(hf_token, str) else None)
        print("数据集上传成功！")
    except Exception as e:
        print(f"上传数据集时出错: {e}")
        print("请检查你的网络连接和 Hugging Face token 权限。")

if __name__ == "__main__":
    # 确保你已经通过 `huggingface-cli login` 登录
    print("重要提示：请确保您已通过 'huggingface-cli login' 登录 Hugging Face Hub。")
    main()
