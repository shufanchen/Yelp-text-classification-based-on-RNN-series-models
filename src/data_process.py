#coding=utf-8
import json
import pickle
from nltk.tokenize import WordPunctTokenizer
from collections import defaultdict

#使用nltk分词器
word_tokenizer = WordPunctTokenizer()

#记录每个单词及其出现的频率
word_freq = defaultdict(int)

# 读取数据集，并进行分词，统计每个单词出现次数，保存在word freq中
with open('../dataset/yelp_academic_dataset_review.json', 'rb') as f:
    for line in f:
        review = json.loads(line)
        words = word_tokenizer.tokenize(review['text'])
        for word in words:
            word_freq[word] += 1

    print("load finished")

# 将词频表保存下来
with open('word_freq.pickle', 'wb') as g:
    pickle.dump(word_freq, g)
    print(len(word_freq)) #159654
    print("word_freq save finished")

num_classes = 5
# 将词频排序，
sort_words = list(sorted(word_freq.items(), key=lambda x:-x[1]))
print(sort_words[:10], sort_words[-10:])

#构建vocablary，并将出现次数小于5的单词全部去除，视为UNKNOW
vocab = {}
i = 1
vocab['UNKNOW_TOKEN'] = 0
for word, freq in word_freq.items():
    if freq > 5:
        vocab[word] = i
        i += 1
print(i)
UNKNOWN = 0

data_x = []
data_y = []

max_word_in_sent = 300

#将所有的评论文件都转化为长度300的索引列表，也就是每篇评论统一都有300个单词
#不够的补零，多余的删除，并保存到最终的数据集文件之中
with open('../dataset/yelp_academic_dataset_review.json', 'rb') as f:
    for line in f:
        doc = []
        review = json.loads(line)
        words = word_tokenizer.tokenize(review['text'])
        word_to_index = []
        for i, word in enumerate(words):
            if i < max_word_in_sent:
                word_to_index.append(vocab.get(word, UNKNOWN))
        if len(word_to_index) < max_word_in_sent:
            word_to_index = word_to_index + [0] * (300 - len(word_to_index))
        doc.append(word_to_index)


        label = int(review['stars'])-1

        data_y.append([label])
        data_x.append(doc)
    pickle.dump((data_x, data_y), open('yelp_data_1.pickle', 'wb'))
    #print(len(data_x))  #229907
    length = len(data_x)
    train_x, val_x, test_x = data_x[:-10000], data_x[-10000:-1000], data_x[-1000:]
    train_y, val_y, test_y = data_y[:-10000], data_y[-10000:-1000], data_y[-1000:]
    pickle.dump((train_x, train_y), open('yelp_train.pickle', 'wb'))
    print(len(train_x))
    pickle.dump((val_x, val_y), open('yelp_val.pickle', 'wb'))
    print(len(val_x))
    pickle.dump((test_x, test_y), open('yelp_test.pickle', 'wb'))
    print(len(test_x))