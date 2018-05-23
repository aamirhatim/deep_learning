import autograd.numpy as np
from autograd import value_and_grad
from autograd import grad as compute_grad
from autograd.misc.flatten import flatten_func

def gradient_descent(g,alpha_choice,max_its,w):
    # compute the gradient function of our input function - note this is a function too
    # that - when evaluated - returns both the gradient and function evaluations (remember
    # as discussed in Chapter 3 we always ge the function evaluation 'for free' when we use
    # an Automatic Differntiator to evaluate the gradient)
    g_flat, unflatten, w = flatten_func(g, w)
    grad = value_and_grad(g_flat)

    # run the gradient descent loop
    weight_history = []      # container for weight history
    weight_history.append(unflatten(w))
    cost_history = []        # container for corresponding cost function history
    cost_history.append(g_flat(w))
    for k in range(max_its):
        # check if diminishing steplength rule used

        alpha = alpha_choice

        # evaluate the gradient, store current weights and cost function value
        cost_eval,grad_eval = grad(w)

        grad_eval.shape = np.shape(w)
        # w,un= flatten(w)

        # take gradient descent step
        w = w - alpha*grad_eval

        weight_history.append(unflatten(w))
        # cost_history.append(cost_eval)
        cost_history.append(g_flat(w))

    # collect final weights
    # weight_history.append(w)
    # compute final cost function value via g itself (since we aren't computing
    # the gradient at the final step we don't get the final cost function value
    # via the Automatic Differentiatoor)

    return weight_history,cost_history

def normalized_gradient_descent(g, alpha, max_its, w, beta):
    # flatten the input function, create gradient based on flat function
    g_flat, unflatten, w = flatten_func(g, w)
    grad = compute_grad(g_flat)

    # Record histories
    weight_hist = []
    weight_hist.append(unflatten(w))
    cost_hist = []

    # run the gradient descent loop
    z = np.zeros((np.shape(w)))         # momentum term

    for k in range(max_its):
        # evaluate the gradient, compute its length
        grad_eval = grad(w)
        grad_eval.shape = np.shape(w)
        grad_norm = np.linalg.norm(grad_eval)

        # check that magnitude of gradient is not too small, if yes pick a random direction to move
        if grad_norm == 0:
            # pick random direction and normalize to have unit legnth
            grad_eval = 10**-6*np.sign(2*np.random.rand(len(w)) - 1)
            grad_norm = np.linalg.norm(grad_eval)

        grad_eval /= grad_norm

        # take descent step with momentum
        z = beta*z + grad_eval
        w = w - alpha*z

        # Record and update histories
        weight_hist.append(unflatten(w))
        cost_hist.append(g_flat(w))

    return weight_hist, cost_hist
