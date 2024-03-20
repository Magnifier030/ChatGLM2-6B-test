import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import os

# 初始化分布式训练环境
def init_distributed():
    # 初始化分布式训练
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.manual_seed(0)

# 加载预训练模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").cuda()

# 将模型包装成分布式数据并行模型
init_distributed()
model = DDP(model)

# 准备数据
conversations = pd.read_json('./train_data.json')
batch_size = 2

# 使用分布式采样器
train_sampler = DistributedSampler(conversations, num_replicas=torch.distributed.get_world_size(), rank=torch.distributed.get_rank())
train_loader = DataLoader(conversations, batch_size=batch_size, sampler=train_sampler)

# 定义损失函数和优化器
loss_function = nn.CrossEntropyLoss().cuda()
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

# 开始分布式训练
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        inputs = tokenizer(batch['data'].apply(lambda x: x["text"]).tolist(), return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 输出每个 epoch 的平均损失
    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Average Loss: {average_loss}")

# 保存微调后的模型
torch.save(model.module.state_dict(), "./fine_tuned_chat_model.pth")
