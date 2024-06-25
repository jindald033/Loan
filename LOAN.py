import pandas as pd
import streamlit as st
df = pd.read_csv("Churn_Modelling.csv")
df1 = df[['CreditScore','EstimatedSalary']]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df1_std = scaler.fit_transform(df1)
df1_std = pd.DataFrame(data = df1_std,columns = df1.columns)
from sklearn.cluster import KMeans
inertias = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df1)
    inertias.append(kmeans.inertia_)

kmeans = KMeans(n_clusters=3)
km = kmeans.fit(df1)
# Get cluster labels
labels = kmeans.labels_
df1['cluster_label'] = labels
X = df1.iloc[:,0:2]
y = df1.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 2)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)

st.title("Suggestion App for Bank Customers")
i=int(st.number_input('Input ID'))
x=pd.DataFrame(X.loc[i]).T
#print(k)
prediction = int(rfc.predict(x))
st.write(prediction)

p = y.map({0:'Lower Middle Class',1:'Upper Middle Class',2:'Well-Off'})
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
