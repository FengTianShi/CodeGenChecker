from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os

# 加载预训练模型和分词器
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base")

codes = []

current_directory = os.path.dirname(__file__)

generated_code_path = os.path.join(current_directory, "dataset", "generated")
human_code_path = os.path.join(current_directory, "dataset", "human")

generated_code_list = os.listdir(generated_code_path)
human_code_list = os.listdir(human_code_path)

for generated_code_name in generated_code_list:
    with open(
        os.path.join(generated_code_path, generated_code_name),
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:
        generated_code = f.read().strip()
        codes.append(generated_code)

for human_code_name in human_code_list:
    with open(
        os.path.join(human_code_path, human_code_name),
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:
        human_code = f.read().strip()
        codes.append(human_code)

labels = []
for i in range(len(generated_code_list)):
    labels.append(0)

for i in range(len(human_code_list)):
    labels.append(1)

# 编码数据
inputs = tokenizer(codes, padding=True, truncation=True, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
labels = torch.tensor(labels)

# 创建数据加载器
dataset = TensorDataset(input_ids, attention_mask, labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 设置优化器
optimizer = AdamW(model.parameters(), lr=3e-5)

# 微调模型
model.train()
for epoch in range(3):  # 对数据集进行迭代
    for batch in tqdm(dataloader):
        batch_input_ids, batch_attention_mask, batch_labels = batch

        # 模型输出
        outputs = model(
            batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels
        )
        loss = outputs.loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"epoch {epoch+1} completed")

# 保存模型
model.save_pretrained("./codebert_finetuned")

print("finetune completed")
