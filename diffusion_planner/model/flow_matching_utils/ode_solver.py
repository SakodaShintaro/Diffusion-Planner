import torch


def euler_integration(func, x, num_steps):
    """
    Numerical integration using Euler's method
    x_t+1 = x_t + f(x_t, t) * dt
    """
    B, P, _ = x.shape
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.ones(B).to(x.device) * (i * dt)
        v = func(x, t)
        v = v.reshape(B, P, -1, 4)
        x = x.reshape(B, P, -1, 4)
        x[:, :, 1:] += v[:, :, 1:] * dt
        x = x.reshape(B, P, -1)

    return x


def heun_integration(func, x, num_steps):
    """
    Numerical integration using Heun's method (Improved Euler)
    k1 = f(x_t, t)
    k2 = f(x_t + k1 * dt, t + dt)
    x_t+1 = x_t + (k1 + k2) * dt / 2
    """
    B, P, _ = x.shape
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t_curr = torch.ones(B).to(x.device) * ((i + 0) * dt)
        t_next = torch.ones(B).to(x.device) * ((i + 1) * dt)

        # Step 1: k1 = f(x_t, t)
        k1 = func(x, t_curr)
        k1 = k1.reshape(B, P, -1, 4)
        x_reshaped = x.reshape(B, P, -1, 4)

        # Prediction step: x_pred = x_t + k1 * dt
        x_pred = x.clone().reshape(B, P, -1, 4)
        x_pred[:, :, 1:] = x_reshaped[:, :, 1:] + k1[:, :, 1:] * dt
        x_pred = x_pred.reshape(B, P, -1)

        # Step 2: k2 = f(x_pred, t + dt)
        k2 = func(x_pred, t_next)
        k2 = k2.reshape(B, P, -1, 4)

        # Update step: x_t+1 = x_t + (k1 + k2) * dt / 2
        x = x.reshape(B, P, -1, 4)
        x[:, :, 1:] += (k1[:, :, 1:] + k2[:, :, 1:]) * dt / 2
        x = x.reshape(B, P, -1)

    return x


def rk4_integration(func, x, num_steps):
    """
    Numerical integration using 4th-order Runge-Kutta method
    k1 = f(x_t, t)
    k2 = f(x_t + k1 * dt/2, t + dt/2)
    k3 = f(x_t + k2 * dt/2, t + dt/2)
    k4 = f(x_t + k3 * dt, t + dt)
    x_t+1 = x_t + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
    """
    B, P, _ = x.shape
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t_curr = torch.ones(B).to(x.device) * ((i + 0) * dt)
        t_mid = torch.ones(B).to(x.device) * ((i + 0.5) * dt)
        t_next = torch.ones(B).to(x.device) * ((i + 1) * dt)

        x_reshaped = x.reshape(B, P, -1, 4)

        # Step 1: k1 = f(x_t, t)
        k1 = func(x, t_curr).reshape(B, P, -1, 4)

        # Step 2: k2 = f(x_t + k1 * dt/2, t + dt/2)
        x_k2 = x.clone().reshape(B, P, -1, 4)
        x_k2[:, :, 1:] = x_reshaped[:, :, 1:] + k1[:, :, 1:] * dt / 2
        x_k2 = x_k2.reshape(B, P, -1)
        k2 = func(x_k2, t_mid).reshape(B, P, -1, 4)

        # Step 3: k3 = f(x_t + k2 * dt/2, t + dt/2)
        x_k3 = x.clone().reshape(B, P, -1, 4)
        x_k3[:, :, 1:] = x_reshaped[:, :, 1:] + k2[:, :, 1:] * dt / 2
        x_k3 = x_k3.reshape(B, P, -1)
        k3 = func(x_k3, t_mid).reshape(B, P, -1, 4)

        # Step 4: k4 = f(x_t + k3 * dt, t + dt)
        x_k4 = x.clone().reshape(B, P, -1, 4)
        x_k4[:, :, 1:] = x_reshaped[:, :, 1:] + k3[:, :, 1:] * dt
        x_k4 = x_k4.reshape(B, P, -1)
        k4 = func(x_k4, t_next).reshape(B, P, -1, 4)

        # Update step: x_t+1 = x_t + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
        x = x.reshape(B, P, -1, 4)
        x[:, :, 1:] += (k1[:, :, 1:] + 2 * k2[:, :, 1:] + 2 * k3[:, :, 1:] + k4[:, :, 1:]) * dt / 6
        x = x.reshape(B, P, -1)

    return x
