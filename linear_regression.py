from numpy import *
import math
import matplotlib.pyplot as plt

def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (b + (m*x))) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    
    N = len(points)
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]

        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))

    new_b = b_current - ((learningRate * 2) * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m

    cost_h = []
    for i in range(num_iterations):
        b, m= step_gradient(b, m, array(points), learning_rate)

        cost_h.append(compute_error_for_line_given_points(b, m, points))
        if(i % 100 == 0):
            #cost_h.append(compute_error_for_line_given_points(b, m, points))
            print("After {0} iterations b = {1}, m = {2}, error = {3}".format(i, b, m, compute_error_for_line_given_points(b, m, points)))
    return b, m, cost_h

def run():
    points = genfromtxt("data.csv", delimiter=",")
    initial_b = 0
    initial_m = 0
    learning_rate = 0.00065
    num_iterations = 30000

    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m,compute_error_for_line_given_points(initial_b, initial_m, points)))
    print("Running...")
    b, m, cost_h = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)

    x_points = []
    y_points = []

    for i in range(0, len(points)):
        x_points.append(points.item(i, 0))
        y_points.append(b + (m * points.item(i, 0)))
    
    plt.figure(1)
    plt.ylabel('Y')
    plt.xlabel('X')

    plt.plot(x_points, y_points, color='r')
    plt.scatter(points[:,0], points[:,1])

    plt.figure(2) 
    plt.ylabel('Total cost')
    plt.xlabel('Iteration')
    plt.plot(cost_h)

    plt.show()

if __name__ == '__main__':
    run()