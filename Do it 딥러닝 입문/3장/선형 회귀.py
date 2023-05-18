from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt

diabetes = load_diabetes()

plt.scatter(diabetes.data[:,2], diabetes.target)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

x = diabetes.data[:,2]
y = diabetes.target