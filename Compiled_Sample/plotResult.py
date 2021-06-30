import numpy as np
import pylab
import matplotlib.pyplot as plt

original_data = pylab.loadtxt('e0103.txt');
mat_profile = pylab.loadtxt('SCAMP_MatrixProfile_and_Index_1024_e0103.txt');

ind_org_data = np.arange(0, len(original_data), 1);
ind_mat_prof = np.arange(0, len(mat_profile), 1);

print("Original data length: " + str(len(original_data)))

plt.subplot(2, 1, 1)
plt.plot(ind_org_data, original_data)
plt.xlim([2500, 5000])

plt.xlabel('Samples')
plt.ylabel('Value')
plt.title('Input Time Series (electrocardiogram)')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(ind_mat_prof, mat_profile[:,0], 'C2')
plt.xlim([2500, 5000])
plt.xlabel('Samples')
plt.ylabel('Value')
plt.title('Matrix Profile')



plt.grid(True)

plt.show()
