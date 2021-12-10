import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# %matplotlib inline if Jupyter

# Loading the data


def load_data():

    data = np.genfromtxt('./kangaroo.csv', delimiter=',')

    x = data[:, 0]
    y = data[:, 1]

    train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(
        x, y, test_size=0.33, random_state=42)

    return train_set_x, test_set_x, train_set_y, test_set_y


def initialize_with_zeros():
    """
    This function initializes parameters theta and b as 0.

    Returns:
    theta -- initialized scalar parameter
    b -- initialized scalar (corresponds to the bias)
    """

    theta = 0
    b = 0

    assert(isinstance(theta, int))
    assert(isinstance(b, int))

    return theta, b


def propagate(theta, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    theta -- parameter, a scalar
    b -- bias, a scalar
    X -- features vector of size (number of examples, )
    Y -- results vector (number of examples, )

    Return:
    cost -- cost function for linear regression
    dt -- gradient of the loss with respect to theta, thus same shape as theta
    db -- gradient of the loss with respect to b, thus same shape as b

    Tips:
    - Write your code step by step for the propagation.
    - Use np.dot() to avoid for-loops in favor of code vectorization
    """

    m = X.shape[0]

    H = X.dot(theta) + b                       # compute activation
    cost = 1 / (2*m) * np.sum((H - Y) ** 2)    # compute cost

    dt = 1 / m * X.dot((H - Y).T)
    db = 1 / m * np.sum(H - Y)

    assert(dt.dtype == float)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dt": dt,
             "db": db}

    return grads, cost


def optimize(theta, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes theta and b by running a gradient descent algorithm

    Arguments:
    theta -- parameter, a scalar
    b -- bias, a scalar
    X -- features vector of shape (number of examples, )
    Y -- results vector of shape (number of examples, )
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights theta and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for theta and b.
    """

    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(theta, b, X, Y)

        # Retrieve derivatives from grads
        dt = grads["dt"]
        db = grads["db"]

        theta = theta - learning_rate * dt
        b = b - learning_rate * db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))


    params = {"theta": theta,
              "b": b}

    grads = {"dt": dt,
             "db": db}

    return params, grads, costs


def predict(theta, b, X):
    """
    Predict using learned linear regression parameters (theta, b)

    Arguments:
    theta -- parameter, a scalar
    b -- bias, a scalar
    X -- features vector of size (number of examples, )

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions for the examples in X
    """

    # Compute vector "Y_prediction" predicting the width of a kangoroo nasal
    ### START CODE HERE ### (≈ 1 line of code)
    Y_prediction = theta * X + b
    ### END CODE HERE ###

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the linear regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (m_train, )
    Y_train -- training values represented by a numpy array (vector) of shape (m_train, )
    X_test -- test set represented by a numpy array of shape (m_test, )
    Y_test -- test values represented by a numpy array (vector) of shape (m_test, )
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """

    # initialize parameters with zeros (≈ 1 line of code)
    theta, b = initialize_with_zeros()

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(theta, b, train_set_x, train_set_y, num_iterations, learning_rate, False)

    # Retrieve parameters w and b from dictionary "parameters"
    theta = parameters["theta"]
    b = parameters["b"]

    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(theta, b, test_set_x)
    Y_prediction_train = predict(theta, b, train_set_x)

    # Print train/test Errors
    print("Train RMSE: {} ".format(np.sqrt(np.mean((Y_prediction_train - Y_train) ** 2))))
    print("Test RMSE: {} ".format(np.sqrt(np.mean((Y_prediction_test - Y_test) ** 2))))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "theta": theta,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d

train_set_x, test_set_x, train_set_y, test_set_y = load_data()

m_train = int((len(train_set_x) + len(train_set_y)) / 2)
m_test = int((len(test_set_x) + len(test_set_y)) / 2)


# Standardization
mean = np.concatenate([train_set_x, test_set_x]).mean()
std = np.concatenate([train_set_x, test_set_x]).std()

train_set_x = (train_set_x - mean) / std
test_set_x = (test_set_x - mean) / std


d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=500, learning_rate=0.05, print_cost=True)


# Training set
plt.figure(figsize=(4, 3))
plt.title("Training set")

plt.scatter(train_set_x, train_set_y)
x = np.array([min(train_set_x), max(train_set_x)])
theta = d["theta"]
b = d["b"]
y = theta * x + b
plt.plot(x, y)
plt.axis("tight")
plt.xlabel("Length")
plt.ylabel("Width")
plt.tight_layout()


# Test set
plt.figure(figsize=(4, 3))
plt.title("Test set")

plt.scatter(test_set_x, test_set_y)
x = np.array([min(test_set_x), max(test_set_x)])
theta = d["theta"]
b = d["b"]
y = theta * x + b
plt.plot(x, y)
plt.axis("tight")
plt.xlabel("Length")
plt.ylabel("Width")
plt.tight_layout()
