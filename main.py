import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report
import pickle
import source.crnn as ModelSource

fp_correct = open('./results/crnn-max-all-features_correct.txt','wb')
fp_wrong = open('./results/crnn-max-all-features_wrong.txt','wb')


with open('./i2b2/i2b2-train.pickle', 'rb') as handle:
	W = pickle.load(handle)
	Y_onehot = pickle.load(handle)
	# wv = pickle.load(handle)
	word_list = pickle.load(handle)
	rev_word_list = pickle.load(handle)
	label_dict = pickle.load(handle)
	rev_label_dict = pickle.load(handle)


per = 0.8

num_total = len(W)
seq_len = len(W[0])
#seq_len=124
word_dict_size = len(word_list)
label_dict_size = len(label_dict)


W_train = W
Y_train = Y_onehot

model = ModelSource.Model(label_dict_size,seq_len,word_dict_size)

## Training the model
num_train = len(W_train)
y_true_list = []
y_pred_list = []
num_epochs = 10
N = 5
batch_size = 256
num_batches_per_epoch = int(num_train/batch_size)


def test_step(W_te, Y_te):
	n = len(W_te)
	num = int(n/batch_size) + 1
	sample = []
	for batch_num in range(num):
		start_index = batch_num*batch_size
		end_index = min((batch_num + 1) * batch_size, n)
		a=range(start_index, end_index)
		sample.append(a)
	pred = []
	for i in sample:
		p = model.test_step(W_te[i], Y_te[i])
		pred.extend(p)
	return pred

for j in range(num_epochs):
	acc = []		
	step = 0
	sam=[]
	for batch_num in range(num_batches_per_epoch):	
		start_index = batch_num*batch_size
		end_index = (batch_num + 1) * batch_size
		sam.append(range(start_index, end_index))
	
	for rang in sam:
		step,acc_cur  = model.train_step(W_train[rang], Y_train[rang])
		acc.append(acc_cur)
	
	acc = np.array(acc)
	print ("Average accuracy for epoch",j+1,"=",np.mean(acc))
print ("Training finished.")

##------------------------------------------------------------------------------------##
##TESTING


with open('./i2b2/i2b2-test.pickle', 'rb') as handle:
	sent_names=pickle.load(handle)
	sentences = pickle.load(handle)
	sent_lengths = pickle.load(handle)
	W_te = pickle.load(handle)
	Y_onehot = pickle.load(handle)
	# wv = pickle.load(handle)
	word_list = pickle.load(handle)
	rev_word_list = pickle.load(handle)
	label_dict = pickle.load(handle)
	rev_label_dict = pickle.load(handle)

print ("Test data loaded")

num_total = len(W_te)
seq_len = len(W_te[0])



pred = test_step(W_te,Y_onehot)

y_true = np.argmax(Y_onehot, 1)
y_pred = pred

print(classification_report(y_true, y_pred,[1,2,3,4,5,6,7,8],digits=4))