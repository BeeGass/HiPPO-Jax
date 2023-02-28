def pick_optimizer_fn(starting_learning_rate, name="adam"):
    # refer to https://optax.readthedocs.io/en/latest/api.html#optimizer-schedules
    optim = None

    if name == "sgd":
        optim = optax.sgd(starting_learning_rate)
    elif name == "adam":
        optim = optax.adam(
            starting_learning_rate,
        )
    elif name == "adagrad":
        optim = optax.adagrad(starting_learning_rate)
    elif name == "rmsprop":
        optim = optax.rmsprop(starting_learning_rate)
    else:
        raise ValueError("optimizer name not recognized")

    return optim


def pick_scheduler_fn(
    start_learning_rate, steps, decay_rate, init_value, end_val, name
):
    # refer to https://optax.readthedocs.io/en/latest/api.html#schedules
    scheduler = None

    if name == "constant":
        scheduler = optax.constant_schedule(init_value)

    elif name == "exp_decay":
        scheduler = optax.exponential_decay(
            init_value=start_learning_rate, transition_steps=1000, decay_rate=0.99
        )
    elif name == "linear":
        scheduler = optax.linear_schedule(init_value=init_value, end_value=end_val)

    else:
        raise ValueError("scheduler name not recognized")

    return scheduler
