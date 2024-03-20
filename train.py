from transformers import AutoModel, AutoTokenizer
import torch
from torch.utils.data import DataLoader
import pandas as pd
import os

torch.cuda.empty_cache()

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4'

# 加载预训练模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).cuda()

# 添加微调步骤
# 假设你有一个对话的文本数据集，命名为conversations
conversations = pd.read_json('./train_data.json') 
num_epochs = 1

# 设置批量大小
batch_size = 8

# 准备数据
inputs = tokenizer(conversations["data"].apply(lambda x: x["text"]).tolist(), return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# 定义损失函数和优化器
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# 开始微调
for epoch in range(num_epochs):
    total_loss = 0.0
    for i in range(0, len(conversations), batch_size):
        batch_conversations = conversations[i:i+batch_size]
        inputs = tokenizer(batch_conversations["data"].apply(lambda x: x["text"]).tolist(), return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        optimizer.zero_grad()
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / (len(conversations) / batch_size)
    print(f"Epoch {epoch + 1}, Average Loss: {average_loss}")

# 保存微调后的模型
model.save_pretrained("./fine_tuned_chat_model")
