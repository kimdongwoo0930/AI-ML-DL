import matplotlib.pyplot as plt
import numpy as np
# plt.plot([1,2,3,4,5],[1,4,9,16,25])
# plt.show()

# plt.scatter([1,2,3,4,5],[1,4,9,16,25])
# plt.show()

x = np.random.randn(1000)
y = np.random.randn(1000)
plt.scatter(x,y)
plt.show()