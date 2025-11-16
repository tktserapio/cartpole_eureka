def base_reward_code():
    return """
def compute_reward(x, x_dot, theta, theta_dot, action):
    angle_penalty = -theta**2
    position_penalty = -0.1 * x**2
    vel_penalty = -0.01 * (x_dot**2 + theta_dot**2)
    alive_bonus = 1.0

    total = alive_bonus + angle_penalty + position_penalty + vel_penalty

    components = {
        "alive_bonus": alive_bonus,
        "angle_penalty": angle_penalty,
        "position_penalty": position_penalty,
        "vel_penalty": vel_penalty
    }
    return float(total), components
"""

def build_reward_fn_from_code(code_str):
    local_vars = {}
    exec(code_str, {}, local_vars)
    return local_vars["compute_reward"]
