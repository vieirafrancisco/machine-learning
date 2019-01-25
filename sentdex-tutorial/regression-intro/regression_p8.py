from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def best_fit_slope(xs, ys):
    #m = sum((x_i-mean(xs))*(y_i-mean(ys)) [for x_i, y_i in zip(xs, ys)])/sum([(x_i-mean(xs))**2 for x_i in xs])
    m = (mean(xs)*mean(ys) - mean(xs*ys))/(mean(xs)**2 - mean(xs**2))
    return m

def y_intercept(m, xs, ys):
    b = mean(ys) - m*mean(xs)
    return b

m = best_fit_slope(xs, ys)
b = y_intercept(m, xs, ys)

regression_line = [(m*x_i)+b for x_i in xs]

print(m, b)

plt.plot(regression_line, 'r')
plt.scatter(xs, ys)

plt.show()