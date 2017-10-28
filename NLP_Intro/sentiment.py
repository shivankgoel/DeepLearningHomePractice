import numpy
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
'''
beautiful soup is a xml parser. 
positive.review and negative.review are xml files
'''
from bs4 import BeautifulSoup

'''
It converts all words into base words. 
Example. Dogs=dog cats=cat jumping = jump etc. 
'''
wordnet_lemmatizer = WordNetLemmatizer()

stopwords = (w.rstrip() for w in open('data/sentiment/stopwords.txt'))

positive_reviews = BeautifulSoup(open('data/sentiment/positive.review').read())
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(open('data/sentiment/negative.review').read())
negative_reviews = positive_reviews.findAll('review_text')

'''
We have more positive reviews than negative. 
So before training we need to cut-off
'''
np.random.shuffle(positive_reviews)
positive_reviews = positive_reviews[:len(negative_reviews)]

'''
Create a dictionary that map words to indices
'''
word_index_map = {}
current_index = 0

'''
Create a tokenizer that takes a string and return an array.
'''
def my_tokenizer(s):
	s = s.lower()
	#this will work better than string.split
	tokens = nltk.tokenize.word_tokenize(s)
	#throw out short words as they won't be useful
	tokens = [t for t in tokens if len(t)>2]
	#convert to base forms
	tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
	#remove stopwords
	tokens = [t for t in tokens if t not in stopwords]
	return tokens

positive_tokenized = []
negative_tokenized = []


for review in positive_reviews:
	tokens = my_tokenizer(review.txt)
	positive_tokenized.append(tokens)
	for token in tokens:
		if token not in word_index_map:
			word_index_map[token] = current_index
			current_index+=1


for review in negative_reviews:
	tokens = my_tokenizer(review.txt)
	negative_tokenized.append(tokens)
	for token in tokens:
		if token not in word_index_map:
			word_index_map[token] = current_index
			current_index+=1


def tokens_to_vector(tokens,label):
	x = np.zeros(len(word_index_map) + 1)
	for t in tokens:
		i = word_index_map(t)
		x[i]+=1
	x = x/x.sum()
	x[-1] = label
	return x


N = len(positive_tokenized) + len(negative_tokenized)
data = np.zeros(N,len(word_index_map)+1)
i = 0

for tokens in positive_tokenized:
	data[i,:] = tokens_to_vector(tokens,1)
	i+=1

for tokens in negative_tokenized:
	data[i,:] = tokens_to_vector(tokens,0)
	i+=1 

np.random.shuffle(data)
x = data[:,:-1]
y = data[:,-1]
xtrain = x[:-100,:]
ytrain = y[:-100]
xtest = x[-100:,:]
ytest = y[-100:]

model = LogisticRegression()
model.fit(xtrain,ytrain)
model.score(xtest,ytest)

threshhold = 0.5
for word,index in word_index_map.iteritems():
	weight = model.coef_[0][index]
	if weight > threshhold or weight < -threshhold:
		print word,weight

