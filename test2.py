import pandas as pd
conversations = pd.read_json('./train_data.json') 
print(conversations['data'].tolist()[0])