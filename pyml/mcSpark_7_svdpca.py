'''
    This code is intended to be run in the IPython shell. 
    You can enter each line in the shell and see the result immediately.
    The expected output in the Python console is presented as commented lines following the
    relevant code.
'''

# %pylab inline
# Populating the interactive namespace from numpy and matplotlib
############################################################################# 2
import matplotlib.pyplot as plt
path = "D:\\lfw\\Aaron_Eckhart\\Aaron_Eckhart_0001.jpg"
ae = plt.imread(path)
plt.imshow(ae)
plt.show()


############################################################################# 4 
tmpPath = "D:\\tmp\\aeGray.jpg"
aeGary = plt.imread(tmpPath)
plt.imshow(aeGary, cmap=plt.cm.gray)
plt.show()

############################################################################# 9
import numpy as np
pc = np.loadtxt("D:\\tmp\\pc.csv", delimiter=",")
print(pc.shape)
# (2500, 10)
def plot_gallery(images, h, w, n_row=2, n_col=5):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[:, i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title("Eigenface %d" % (i + 1), size=12)
        plt.xticks(())
        plt.yticks(())

plot_gallery(pc, 50, 50)
plt.show()

########################################################################### 14
s = np.loadtxt("D:\\tmp\\s.csv", delimiter=",")
print(s.shape)
plt.plot(s)
plt.show()
# (300,)
plt.plot(np.cumsum(s)) # 누적해서 더하가
plt.yscale('log')
plt.show()

#고유벡터 1부터 300까지
#어느 순간부터 약 50 부터는 더 늘려도 많이 늘어나지 않는다.
#클러스터링의 섬오브스퀘어랑 비슷한 결과네