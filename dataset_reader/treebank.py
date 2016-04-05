__author__ = 'BeRo'


data_path = "C:/Users/BeRo/Google Drive/Bero/arshad project/Ms_Thesis/data set/stanfordSentimentTreebank/dictionary.txt"
label_path = "C:/Users/BeRo/Google Drive/Bero/arshad project/Ms_Thesis/data set/stanfordSentimentTreebank/sentiment_labels.txt"

data_file = open(data_path, "r")
data_content = data_file.readlines()
data_file.close()

label_file = open(label_path, "r")
label_content = label_file.readlines()
label_file.close()

data_sentences = []
data_sentences = [None] * len(data_content)

for s in data_content:
    s = s.split("\n")
    s = s[0]
    s = s.split("|")
    index = int(s[1])
    s = s[0]
    data_sentences[index] = s

i = 1
label_tags = []
while i < len(label_content):
    s = label_content[i]
    i += 1
    s = s.split("\n")
    s = s[0]
    s = s.split("|")
    label_tags.append(s[1])


destpath = 'C:/Users/BeRo/Google Drive/Bero/arshad project/Ms_Thesis/data set/stanfordSentimentTreebank/data_sentences.txt'
destfile = open(destpath, "w")
for s in data_sentences:
    destfile.write(s)
    destfile.write("\n")
destfile.close()

destpath = 'C:/Users/BeRo/Google Drive/Bero/arshad project/Ms_Thesis/data set/stanfordSentimentTreebank/label_tags.txt'
destfile = open(destpath, "w")
for s in label_tags:
    destfile.write(s)
    destfile.write("\n")
destfile.close()