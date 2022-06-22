import streamlit as st
import pandas as pd

import nltk
import re
import string
import pickle
import joblib
import timeit
import memory_profiler

nltk.download('stopwords')

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()
wn = nltk.WordNetLemmatizer()


# Security
#passlib,hashlib,bcrypt,scrypt
import hashlib
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False
# DB Management
import sqlite3 
conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data


def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data


@st.cache
def clean_text(text):
    text = ''.join([i for i in text if not i.isdigit()]) 
    text = "".join([i.lower() for i in text if i not in string.punctuation])
    text = ' '.join( [word for word in text.split() if len(word)>2] )
     
    tokens = re.split('\W+', text) 
    #words = [wn.lemmatize(word, 'v') for word in tokens]
    text = [ps.stem(word) for word in tokens if word not in stopwords] 
    text = [wn.lemmatize(word) for word in text] 
    
    text = " ".join(text)
    return text
@st.cache
def vectorizing(text):
	new_question = text
	tfidf_vectorizer = pickle.load(open("tfidf.pickle", "rb"))
	vectorized_question = tfidf_vectorizer.transform([new_question])
	return vectorized_question
@st.cache
def create_features(cleaned_text, vectorized_text):
	text = cleaned_text
	vectorized_text = vectorized_text
	label = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']
	toxic = ['fuck', 'shit', 'suck', 'stupid', 'bitch', 'idiot', 'asshol', 'gay', 'dick']
	severe_toxic = ['fuck', 'bitch', 'suck', 'shit', 'asshol', 'dick', 'cunt', 'faggot', 'cock']
	obscene =['fuck', 'shit', 'suck', 'bitch', 'asshol', 'dick', 'cunt', 'faggot', 'stupid']
	threat =['kill', 'die', 'fuck', 'shit', 'rape', 'hope', 'bitch', 'death', 'hell']
	insult = ['fuck', 'bitch', 'suck', 'shit', 'idiot', 'asshol', 'stupid', 'faggot', 'cunt']
	identity_hate = ['fuck', 'gay', 'nigger', 'faggot', 'shit', 'jew', 'bitch', 'homosexu', 'suck']
	contains_toxic = []
	contains_severe_toxic = []
	contains_obscene = []
	contains_threat = []
	contains_insult = []
	contains_identity_hate =[]
	for col in range(len(label)):
		toxic_list = vars()[label[col]]
		#st.write(toxic_list)
		value = "contains_"+label[col]
		
		check = any(substring in text for substring in toxic_list) 
		if check is True:
			vars()[value].append(1)
			#st.write("True")
		else:
			vars()[value].append(0)
			#st.write("False")
	inp = list([contains_toxic[0],contains_severe_toxic[0],contains_obscene[0], contains_threat[0], contains_insult[0], contains_identity_hate[0]])
	df = pd.DataFrame([inp], columns=['contains_toxic_word', 'contains_severe_toxic_word', 'contains_obscene_word', 'contains_threat_word', 'contains_insult_word', 'contains_identity_hate_word'])
	X = pd.concat([df, pd.DataFrame(vectorized_text.toarray())], axis=1)
	return X
def predict(features, model = 'Linear SVC'):
	start_time = timeit.default_timer()
	if model == 'Logistic Regression':
		svc_from_joblib = joblib.load('lintoxicmodel.pkl')
		accuracy = 71
	if model == 'Linear SVC':
		svc_from_joblib = joblib.load('svctoxicmodel.pkl') 
		accuracy = 73	
	if model == 'Naive Bayes':
		svc_from_joblib = joblib.load('bayestoxicmodel.pkl') 
		accuracy = 65	
	y = svc_from_joblib.predict(features)
	elapsed = timeit.default_timer() - start_time
	return y,elapsed, accuracy
def show_results(option):
    
	message = st.text_area('write a comment here:')
	if st.button('Predict'):
		#st.write(message)
		cleaned_text = clean_text(message)
		#st.write(cleaned_text)
		vectorized_text = vectorizing(cleaned_text)
		#st.write(vectorized_text)
		features = create_features(cleaned_text, vectorized_text)
		#st.write(features)
		prediction, elapsed, accuracy = predict(features, model = option)
		st.info("Time elapsed to predict is {:2f}". format(elapsed/60))
		st.info("Accuracy : {}%". format(accuracy))
		
		df = pd.DataFrame({
			"contains_toxic": prediction[:, 0],
			"contains_severe_toxic": prediction[:, 1],
			"contains_obscene": prediction[:, 2],
			"contains_threat": prediction[:, 3],
			"contains_insult":prediction[:, 4],
			"contains_identity_hate": prediction[:, 5]
			}, index=['Comment'])
		st.write(df.T)
def success(username):
    return st.success("Logged In as {}".format(username))

def main():
	"""Simple Login App"""

	st.title("Toxic Comment Analysis.")

	menu = ["Home","Login","SignUp"]
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home":
		st.subheader("Wecome to our ML and NLP powered project. Sign Up or Log in to continue")

	elif choice == "Login":
		st.subheader("Login Section")

		username = st.sidebar.text_input("User Name")
		password = st.sidebar.text_input("Password",type='password')
		if st.sidebar.checkbox("Login"):
			# if password == '12345':
			create_usertable()
			hashed_pswd = make_hashes(password)

			result = login_user(username,check_hashes(password,hashed_pswd))
			if result:
       			#st.success("Logged In as {}".format(username))
				success(username)
				option = st.selectbox('Which ML model would you like to use?',('Logistic Regression', 'Linear SVC', 'Naive Bayes'))
				st.write('You selected:', option)
				show_results(option)
       
			else:
				st.warning("Incorrect Username/Password")


	elif choice == "SignUp":
		st.subheader("Create New Account")
		new_user = st.text_input("Username")
		new_password = st.text_input("Password",type='password')

		if st.button("Signup"):
			create_usertable()
			add_userdata(new_user,make_hashes(new_password))
			st.success("You have successfully created a valid Account")
			st.info("Go to Login Menu to login")



if __name__ == '__main__':
	main()