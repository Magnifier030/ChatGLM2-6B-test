import pandas as pd
import json

def transform_dataformat(before_file, after_file):
   # read .json file
   with open(before_file,'r', encoding='utf-8') as f:
       data_json = json.load(f)

   # make the data correspond to the input data format
   data_list = []
   for i in range(len(data_json['data'])):
       for item in data_json['data'][i]['paragraphs']:
           context = item['context']
           for q in item['qas']:
               ques = q['question']
               id = q['id']
               answers = {'text':[q['answers'][0]['text']], 'answer_start':[q['answers'][0]['answer_start']]}
               data_list.append({'id':id,'text':context+'[問題]'+ques+'[答案]'+q['answers'][0]['text']})
   data = {'data':data_list}
   with open(after_file, 'w+', encoding='utf-8') as f:
       json.dump(data, f, indent=4)

transform_dataformat('./DRCD_training.json', './change/train_data.json')
# transform_dataformat('./DRCD_dev.json', './change/dev_data.json')
# transform_dataformat('./DRCD_test.json', './change/test_data.json')