import torch
import numpy as np

class LinearRegression:
    def __init__(self,lr=0.01,n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.weights=None
        self.bias=None

    def fit(self,X,XT,y):
        n_samples,n_features=X.shape
        temp=np.zeros(n_features)
        self.weights=torch.from_numpy(temp)
        self.bias=0

        for _ in range(self.n_iters):
            ans1=[]
            for a,b in zip(X,self.weights):
                temp=np.dot(a,b)
                ans1.append(temp)
            ans1=torch.tensor(ans1)
            y_pred=ans1+self.bias

            print("y_pred:")
            print(y_pred)

            print("y:")
            print(y)

            # y2=y_pred-y
            # ans2=[]
            # for a in XT:    # error
            #     temp2=np.dot(a,y2)
            #     ans2.append(temp2)
            # ans2=torch.tensor(ans2)
            # dw=(1/n_samples)*ans2
            # n=0
            # for i in y2:
            #     n+=i
            # db=(1/n_samples)*n

            # self.weights=self.weights-self.lr*dw
            # self.bias=self.bias-self.lr*db

    def predict(self, X):
        ans=[]
        for a,b in zip(X,self.weights):
            temp3=np.dot(a,b)
            ans.append(temp3)

        ans=torch.tensor(ans)
        return ans