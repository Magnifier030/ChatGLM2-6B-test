FROM python:3.9

# 将工作目录设置为 /app
WORKDIR /app

# 复制当前目录中的所有文件到容器的 /app 目录
COPY . /app

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 当容器启动时，运行应用
CMD ["python", "./train.py"]