import pandas as pd
from sklearn.linear_model import Perceptron

if __name__ == "__main__":
	diabetes = pd.read_csv('diabetes.csv').values
	
	x=diabetes[:, 0:8]
	y=diabetes[:, 8]
	
	model = Perceptron(random_state=1)
	model.fit(x,y)
	
	print("%.3f" % model.score(x,y))
