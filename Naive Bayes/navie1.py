import pandas as pd
import numpy as np
d=pd.read_excel("iris_train.xlsx")
train_data = d.as_matrix()
x_train=train_data[:,0:4]
y_train=train_data[:,4]
d1=pd.read_excel("iris_test.xlsx")
test_data = d1.as_matrix()
x_test=test_data[:,0:4]
y_test=test_data[:,4]



def train_bayes(x_train,y_train):
    model_mean={}
    model_variance={}
    label=np.unique(y_train)
    for i in label:
        N=x_train[np.where(y_train==i),:].shape[1]
        avg=np.mean(x_train[np.where(y_train==i),:],1)
        model_mean[i]=avg
        model_variance[i]=a = np.sum(pow(x_train[np.where(y_train==i),:]-avg,2),1)/(N)
    return model_mean,model_variance

model_mean,model_variance=train_bayes(x_train,y_train)
print(model_mean)
len(model_mean)


def test_bayes(x_test,model_mean,model_variance):
    prediction=np.zeros([x_test.shape[0]])
    for data in range(x_test.shape[0]):
        probability=np.zeros([len(model_mean)])
        for i in range(len(model_mean)):
            mean=model_mean[i+1]
            variance=model_variance[i+1]
            exponent=np.exp(np.float32(-(pow(x_test[data,:]-mean,2)/(2*variance))))
            probability[i]=np.prod((1/(np.sqrt(np.float32(2*np.pi*variance)))*exponent),1)
            result_class=np.argmax(probability)
        prediction[data]=result_class+1
    return np.int_(prediction)


model_mean,model_variance=train_bayes(x_train,y_train)
prediction=test_bayes(x_test,model_mean,model_variance)


correct_pred=np.sum(prediction==y_test)
Accuracy=correct_pred*100/y_test.shape[0]
print(Accuracy)