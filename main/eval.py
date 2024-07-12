import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
import os

# 加载微调后的模型和分词器
model = RobertaForSequenceClassification.from_pretrained("./codebert_finetuned")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

test_codes = []

current_directory = os.path.dirname(__file__)

generated_code_path = os.path.join(current_directory, "dataset", "gen_test")
human_code_path = os.path.join(current_directory, "dataset", "hu_test")

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
        test_codes.append(generated_code)

for human_code_name in human_code_list:
    with open(
        os.path.join(human_code_path, human_code_name),
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:
        human_code = f.read().strip()
        test_codes.append(human_code)

test_labels = []
for i in range(len(generated_code_list)):
    test_labels.append(0)

for i in range(len(human_code_list)):
    test_labels.append(1)

# Encode the test data
test_inputs = tokenizer(test_codes, padding=True, truncation=True, return_tensors="pt")
test_input_ids = test_inputs["input_ids"]
test_attention_mask = test_inputs["attention_mask"]
test_labels = torch.tensor(test_labels)

# Create test data loader
test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=2)

# Evaluate the model
model.eval()  # Set the model to evaluation mode
total_eval_accuracy = 0


# Function to calculate accuracy
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


for batch in test_dataloader:
    batch_input_ids, batch_attention_mask, batch_labels = batch

    with torch.no_grad():
        outputs = model(batch_input_ids, attention_mask=batch_attention_mask)

    logits = outputs.logits
    logits = logits.detach().cpu().numpy()
    label_ids = batch_labels.to("cpu").numpy()

    # Calculate accuracy
    total_eval_accuracy += flat_accuracy(logits, label_ids)

# Calculate average accuracy
avg_accuracy = total_eval_accuracy / len(test_dataloader)
print("Accuracy on the test set: {:.2f}".format(avg_accuracy))
