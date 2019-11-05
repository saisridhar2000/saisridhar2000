import numpy as np
import pandas as pd
def do _calculation():
    #import numpy as np
    #import pandas as pd
    dataset=pd.read_csv('dataset2.csv')
    x=dataset.iloc[:,:5]
    x=dataset.iloc[:,:5].values
    y=dataset.iloc[:,-1:]
    y=dataset.iloc[:,-1:].values
    from sklearn.preprocessing import LabelEncoder
    lb=LabelEncoder()
    x[:,1]=lb.fit_transform(x[:,1])
    x[:,2]=lb.fit_transform(x[:,2])
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    model=Sequential()
    model.add(Dense(input_dim=5,init="random_uniform",activation="relu",output_dim=3))
    model.add(Dense(output_dim=3,init="random_uniform",activation="relu"))
    model.add(Dense(output_dim=1,init="random_uniform"))
    model.compile(optimizer='adam',loss='mse',metrics=['mse'])
    model.fit(x_train,y_train,epochs=1000,batch_size=10)
    model.predict(x_test)
    rv=model.predict(np.array([[35,1,7,13,0]]))
    if rv>=0.55:
        print("rh factor is positive")
    else:
        print("rh value is negative")
