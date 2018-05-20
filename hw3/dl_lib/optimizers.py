import autograd.numpy as np
from autograd import value_and_grad
from autograd.misc.flatten import flatten_func

def gradient_descent(g, alpha_choice, max_its, w):
    # Find flatten/unflatten procedures for weight matrix
    g_flat, unflatten, w = flatten_func(g, w)

    gradient = value_and_grad(g_flat)

    # run the gradient descent loop
    weight_history = []      # container for weight history
    cost_history = []        # container for corresponding cost function history
    alpha = 0
    for k in range(1,max_its+1):
        # check if diminishing steplength rule used
        if alpha_choice == 'diminishing':
            alpha = 1/float(k)
        else:
            alpha = alpha_choice

        # evaluate the gradient, store current weights and cost function value
        cost_eval,grad_eval = gradient(w)
        grad_eval.shape = np.shape(w)

        weight_history.append(unflatten(w))
        cost_history.append(cost_eval)

        # take gradient descent step
        w = w - alpha*grad_eval

    # collect final weights
    weight_history.append(unflatten(w))
    # compute final cost function value via g itself (since we aren't computing
    # the gradient at the final step we don't get the final cost function value
    # via the Automatic Differentiatoor)
    cost_history.append(g(unflatten(w)))

    return weight_history,cost_history
