from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import random

#xs = np.array([1,2,3,4,5,6], dtype=np.float64)
#ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []

    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val-=step
        
    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def best_fit_slope(xs, ys):
    #m = sum([(x_i-mean(xs))*(y_i-mean(ys)) for x_i, y_i in zip(xs, ys)])/sum([(x_i-mean(xs))**2 for x_i in xs])
    m = (mean(xs)*mean(ys) - mean(xs*ys))/(mean(xs)**2 - mean(xs**2))
    return m

def y_intercept(m, xs, ys):
    b = mean(ys) - m*mean(xs)
    return b

def square_error(ys_origin, ys_pred):
    return sum([(y_i-y_j)**2 for y_i, y_j in zip(ys_origin, ys_pred)])

def coeficient_of_determination(ys_origin, ys_pred):
    se_pred = square_error(ys_origin, ys_pred)
    y_mean = [mean(ys_origin) for _ in ys_origin]
    se_mean = square_error(ys_origin, y_mean)
    r_squared = 1 - se_pred/se_mean
    return r_squared

xs, ys = create_dataset(40, 10, 2, correlation='pos')

m = best_fit_slope(xs, ys)
b = y_intercept(m, xs, ys)

regression_line = [(m*x_i)+b for x_i in xs]

r_squared = coeficient_of_determination(ys, regression_line)

print(r_squared, m, b)

x_pred = 8
y_pred = m*x_pred + b

plt.scatter(xs, ys)
plt.plot(regression_line, 'r')
plt.scatter(x_pred, y_pred, s=100)

plt.show()