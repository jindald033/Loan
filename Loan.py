import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
import streamlit as st
file = 'data.pkl'
fileobj = open(file, 'rb')

file1 = 'model1.pkl'
fileobj1 = open(file1, 'rb')
k=pickle.load(fileobj1)

m=pickle.load(fileobj)
y=pd.DataFrame(m)

X = y.iloc[:,0:2]
y1 = y.iloc[:,-1]
st.title("Suggestion App for Bank Customers")
i=int(st.number_input('Input ID'))
x=pd.DataFrame(X.loc[i]).T
#print(k)
prediction = int(k.predict(x))
st.write(prediction)

p = y1.map({0:'Lower Middle Class',1:'Upper Middle Class',2:'Well-Off'})
#st.write(p)
st.write(prediction)

if prediction==0:
    st.write('Lower Middle Class')
    st.header('We are suggesting you to avail our loan services.')
    r=st.radio('Are you intersted',options=['yes','no'])
    if r=='yes':
        st.write('Please visit our branch or call 1909090909')
    else:
        st.write('Thank You')
elif prediction==1:
    st.write('Upper Middle Class')
    st.header('We are suggesting you to avail our loan services and investment services')
    r = st.radio('Are you intersted', options=['yes', 'no'])
    if r=='yes':
        st.write('Please visit our branch or call 1909090909')
    else:
        st.write('Thank You')
else:
    st.write('Well-Off')
    st.header('We are suggesting you to avail our deposit services.')
    r = st.radio('Are you intersted', options=['yes', 'no'])
    if r == 'yes':
        st.write('Please visit our branch or call 1909090909')
    else:
        st.write('Thank You')