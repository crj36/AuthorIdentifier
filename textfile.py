import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import arrays_storage as arrs
all_users = ['foo yang#5544','coa#1489','LukeLR#4309','jacky#1045','sewuh#6465','Sherry Baik#3979','mikeyyg#4901','Facade!#2123','AnthonyG#2225']
all_users_msg_count = []
for i in all_users:
    all_users_msg_count.append(0)
for i in arrs.all_authors:
    for x in range(len(all_users)):
        if all_users[x]==i:
            all_users_msg_count[x]+=1
# print(str(all_users_msg_count))
def distiguishability(user1, user2):
    X_used = []
    y_used = []
    for i in range(len(arrs.all_authors)):
        if (arrs.all_authors[i]==user1 or arrs.all_authors[i]==user2) and len(arrs.all_messages[i])>30:
            X_used.append(arrs.all_messages[i])
            y_used.append(arrs.all_authors[i])
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X_used).toarray()
    # print this to see how many messages tested on
    # print(len(y_used))

    X_train, X_test, y_train, y_test = train_test_split(X, y_used, test_size=0.2)

    # Create a decision tree classifier and fit it to the training data
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)


    # Evaluate the model's accuracy
    acc = accuracy_score(y_test, y_pred)
    return str(acc)
   

    # Split the data into training and test sets
    # from sklearn.model_selection import cross_val_score
    # from sklearn.svm import SVC
    # svm = SVC(kernel='linear', C=1, random_state=42)
    # scores = cross_val_score(svm, X, y_used, cv=10, error_score='raise')
    # return str(np.mean(scores))


user_tested = 'coa#1489'
print(user_tested+"'s distinguishability!")
for i in all_users:
    if not i== user_tested:
        print(i+': '+str(round(float(distiguishability(user_tested, i)),2)))
    

