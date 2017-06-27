from gen_data import *
from plotting import *

x, y = generate()

M = np.vstack((x,y)).T
mean = np.mean(M, axis=0)
C = M - mean

B = C.T.dot(C)
_, _ ,V = np.linalg.svd(B)

#plt.subplot(2,1,1)
plt.scatter(C[:,0],C[:,1], alpha = '0.5')
newline([0,0],V[:,0], c='orange')
plt.show()

save(C, 'foo')
dataset_import('foo')
