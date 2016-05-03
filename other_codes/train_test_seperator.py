import numpy as np

train_rate = 0.5
test_rate = 0.4
validate_rate = .1

data_path = '/media/bero/9214EFDB14EFBFF9/Users/BeRo/Google Drive/Bero/arshad project/Ms_Thesis/data set/SentiPersV1.0/separated/beroData.txt'
label_path = '/media/bero/9214EFDB14EFBFF9/Users/BeRo/Google Drive/Bero/arshad project/Ms_Thesis/data set/SentiPersV1.0/separated/beroLabel.txt'

data_file = open(data_path, "r")
label_file = open(label_path, "r")

tmp_data = data_file.readlines()
data_content = []
for s in tmp_data:
    s = s.split("\n")
    s = s[0]
    s = s.split("\r")
    s = s[0]
    if(s == "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"):
        continue
    else:
        data_content.append(s)

tmp_data = label_file.readlines()
label_content = []
for s in tmp_data:
    s = s.split("\n")
    s = s[0]
    s = s.split("\r")
    s = s[0]
    label_content.append(int(s))

data_file.close()
label_file.close()

Num_Of_train = int(len(label_content) * train_rate)
Num_Of_validate = int(len(label_content) * validate_rate)
Num_Of_test = len(label_content) - (Num_Of_train + Num_Of_validate)

train_data = []
train_label = []

validate_data = []
validate_label = []

test_data = []
test_label = []


permutation = np.random.permutation(len(label_content))
index = 0
for i in permutation:
    if index < Num_Of_train:
        train_data.append(data_content[i])
        train_label.append(label_content[i])
    elif index < Num_Of_train + Num_Of_validate:
        validate_data.append(data_content[i])
        validate_label.append(label_content[i])
    else:
        test_data.append(data_content[i])
        test_label.append(label_content[i])
    index += 1
