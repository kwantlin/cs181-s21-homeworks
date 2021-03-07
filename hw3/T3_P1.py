import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

data = [3.3,3.5,3.1,1.8,3.0,0.74,2.5,2.4,1.6,2.1,2.4,1.3,1.7,0.19]
mu = 0
sigma = 1
tausq = 5
stdev = math.sqrt(sigma**2 + tausq)
x = np.linspace(-10, 10, 500)
line, = plt.plot(x, stats.norm.pdf(x, mu, stdev), label='Prior')
plt.legend(handles=[line])
plt.xlabel("x*")
plt.ylabel("p(x*|D)")
plt.savefig('base.png')
plt.show()

for i in range(len(data)):
    data_sum = sum(data[:i+1])
    data_avg = data_sum/(i+1)
    mu_map = data_sum/( ((sigma**2)/tausq) + (i+1) )
    print("Sum data:", data_sum)
    n = i+1
    sigma_n = math.sqrt(1/((n/(sigma**2)) + (1/tausq)))
    print("Posterior Sigma:", sigma_n)
    mu_n = (data_sum/(sigma**2)) * sigma_n**2
    print("Posterior Mu:", mu_n)
    # x = np.linspace(mu_n - 3*sigma_n, mu_n + 3*sigma_n, 100)
    line1, = plt.plot(x, stats.norm.pdf(x, mu_n, math.sqrt(sigma_n**2+sigma**2)), label='Posterior Predictive')
    line2, = plt.plot(x, stats.norm.pdf(x, data_avg, stdev),label='MLE estimate')
    line3, = plt.plot(x, stats.norm.pdf(x, mu_map, stdev), label='MAP estimate')
    plt.legend(handles=[line1, line2, line3])
    plt.xlabel("x*")
    plt.ylabel("p(x*|D)")
    plt.savefig("iter" + str(i+1) + ".png")
    plt.show()



