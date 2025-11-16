import statistics

def summarize_reward_components(model, env, steps=1000):
    logs = {}
    obs, info = env.reset()
    done = False

    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        comps = info.get("reward_components", {})

        for k, v in comps.items():
            logs.setdefault(k, []).append(float(v))

        if done:
            obs, info = env.reset()
            done = False

    out = []
    for k, vals in logs.items():
        out.append(f"{k}: mean={statistics.mean(vals):.3f}, min={min(vals):.3f}, max={max(vals):.3f}")
    return "\n".join(out)
