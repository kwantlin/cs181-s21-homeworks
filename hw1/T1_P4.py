#####################
# CS 181, Spring 2021
# Homework 1, Problem 4
# Start Code
##################

import csv
import numpy as np
import matplotlib.pyplot as plt

csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
print("Years shape", years.shape)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985
# Plot the data.
plt.figure(1)
plt.plot(years, republican_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.figure(2)
plt.plot(years, sunspot_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Sunspots")
plt.figure(3)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()

# Create the simplest basis, with just the time and an offset.
X = np.vstack((np.ones(years.shape), years)).T
print("X shape", X.shape)
# TODO: basis functions
# Based on the letter input for part ('a','b','c','d'), output numpy arrays for the bases.
# The shape of arrays you return should be: (a) 24x6, (b) 24x12, (c) 24x6, (c) 24x26
# xx is the input of years (or any variable you want to turn into the appropriate basis).
def make_basis(xx,part='a',is_years=True):
    newx = xx.copy()
    if part == 'a' and is_years:
        print("ENTERED 1A CONDITION")
        newx[:,1] = (newx[:,1] - np.array([1960]*len(newx)))/40

    if part == "a" and not is_years:
        print("ENTERED 2A CONDITION")
        newx[:,-1] = newx[:,1]/20
    # print("Newx", newx[:,1])
    res = [newx[:,0]]
    if part == 'a':
        for i in range(1, 6):
            res.append(list(np.power(newx[:,1], i)))
    elif part == 'b':
        for i in range(1960, 2011, 5):
            res.append(list(np.exp( (np.power(newx[:,1]-i, 2) * (-1)) / (25) )))
    elif part == 'c':
        for i in range(1, 6):
            res.append(list(np.cos(newx[:,1]/i)))
    elif part == 'd':
        for i in range(1, 26):
            res.append(list(np.cos(newx[:,1]/i)))
    res = np.array(res).T
    return res

# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
def find_weights(X,Y):
    w = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
    return w

def residual(Y, Y_hat):
    res = 0
    # print("Y shape", Y.shape)
    # print("Y_hat shape", Y_hat.shape)
    for i in range(len(Y)):
        res += (Y_hat[i] - Y[i]) **2
    return res

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!

grid_years = np.linspace(1960, 2005, 200)
grid_X = np.vstack((np.ones(grid_years.shape), grid_years))


# TODO: plot and report sum of squared error for each basis

# Plot the data and the regression line.

# 1a
print("1A")
# print("X shape", X.shape)
# print("grid_X shape", grid_X.shape)
X_1a = make_basis(X, 'a')
# print("X_1a shape", X_1a.shape)
test = np.dot(X_1a.T, X_1a)
w = find_weights(X_1a, Y)
# print("w shape", w.shape)
grid_X_1a = make_basis(grid_X.T, 'a')
# print("grid_X_1a shape", grid_X_1a.shape)
grid_Yhat  = np.dot(grid_X_1a, w)
plt.figure(4)
plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.show()
Y_hat = np.dot(X_1a, w)
print("1a res:", residual(Y, Y_hat))

#1b
print("1B")
X_1a = make_basis(X, 'b')
test = np.dot(X_1a.T, X_1a)
w = find_weights(X_1a, Y)
grid_X_1a = make_basis(grid_X.T, 'b')
grid_Yhat  = np.dot(grid_X_1a, w)
plt.figure(5)
plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.show()
Y_hat = np.dot(X_1a, w)
print("1b res:", residual(Y, Y_hat))

#1c
print("1C")
X_1a = make_basis(X, 'c')
test = np.dot(X_1a.T, X_1a)
w = find_weights(X_1a, Y)
grid_X_1a = make_basis(grid_X.T, 'c')
grid_Yhat  = np.dot(grid_X_1a, w)
plt.figure(6)
plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.show()
Y_hat = np.dot(X_1a, w)
print("1c res:", residual(Y, Y_hat))

#1d
print("1D")
X_1a = make_basis(X, 'd')
test = np.dot(X_1a.T, X_1a)
w = find_weights(X_1a, Y)
grid_X_1a = make_basis(grid_X.T, 'd')
grid_Yhat  = np.dot(grid_X_1a, w)
plt.figure(7)
plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.show()
Y_hat = np.dot(X_1a, w)
print("1d res:", residual(Y, Y_hat))

# Part 2

X = np.vstack((np.ones(sunspot_counts[years<last_year].shape), sunspot_counts[years<last_year])).T
print("X shape", X.shape)
grid_sunspots = np.linspace(0, max(sunspot_counts), 200)
grid_X = np.vstack((np.ones(grid_sunspots.shape), grid_sunspots))
Y = republican_counts.copy()
Y = Y[years<last_year]

#2a
print("2A")
X_1a = make_basis(X, 'a', False)
test = np.dot(X_1a.T, X_1a)
w = find_weights(X_1a, Y)
print("grid_X shape", grid_X.shape)
grid_X_1a = make_basis(grid_X.T, 'a', False)
grid_Yhat  = np.dot(grid_X_1a, w)
plt.figure(8)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o', grid_sunspots, grid_Yhat, '-')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()
Y_hat = np.dot(X_1a, w)
print("Y_hat", Y_hat)
print("2a res:", residual(Y, Y_hat))

#2c
print("2C")
X_1a = make_basis(X, 'c', False)
test = np.dot(X_1a.T, X_1a)
w = find_weights(X_1a, Y)
grid_X_1a = make_basis(grid_X.T, 'c', False)
grid_Yhat  = np.dot(grid_X_1a, w)
plt.figure(9)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o', grid_sunspots, grid_Yhat, '-')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()
Y_hat = np.dot(X_1a, w)
print("Y_hat", Y_hat)
print("2c res:", residual(Y, Y_hat))

#2d
print("2D")
X_1a = make_basis(X, 'd', False)
test = np.dot(X_1a.T, X_1a)
w = find_weights(X_1a, Y)
grid_X_1a = make_basis(grid_X.T, 'd', False)
grid_Yhat  = np.dot(grid_X_1a, w)
plt.figure(10)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o', grid_sunspots, grid_Yhat, '-')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()
Y_hat = np.dot(X_1a, w)
print("Y_hat", Y_hat)
print("2d res:", residual(Y, Y_hat))