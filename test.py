import os

# 指定檔案路徑
file_path = './file.txt'
# 建立空白檔案
open(file_path, 'w').close()
# 或者寫入內容到檔案中
with open(file_path, 'w') as file:
    file.write('這是檔案的內容。')