from transformers import AutoModel, AutoTokenizer
import torch
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '6'

# 加载预训练模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).cuda()

# 添加微调步骤
# 假设你有一个对话的文本数据集，命名为conversations
conversations = pd.read_json('./train_data.json') 
num_epochs = 1

# 准备数据
inputs = tokenizer(conversations, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# 定义损失函数和优化器
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# 开始微调
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# 保存微调后的模型
model.save_pretrained("./fine_tuned_chat_model")
