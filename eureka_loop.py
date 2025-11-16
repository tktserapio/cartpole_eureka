from training import train_model, evaluate
from reflection import summarize_reward_components
from llm_interface import generate_reward_code
from reward_code import base_reward_code
import re

def extract_code_from_response(response_text):
    """Extract Python code from LLM response that may contain markdown code blocks."""
    # Try different patterns in order of preference
    patterns = [
        r'```python\n(.*?)\n```',  # ```python ... ```
        r'```python(.*?)```',      # ```python...``` (no newlines)
        r'```\n(.*?)\n```',        # ``` ... ``` (generic)
        r'```(.*?)```',            # ```...``` (generic, no newlines)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if code:  # Make sure we extracted something
                return code
    
    # If no code blocks found, return the entire response
    # (in case LLM returned raw code without markdown)
    return response_text.strip()

def eureka_search(iterations=2, batch_size=2, train_steps=20_000):
    best_code = None
    best_score = -float("inf")

    prompt = (
        "Write a Python function compute_reward(x, x_dot, theta, theta_dot, action) "
        "that returns (reward, components). Improve CartPole balance time."
    )

    # seed with the base reward
    prompt += "\n\nHere is the current baseline:\n" + base_reward_code()

    for it in range(iterations):
        candidates = []
        for _ in range(batch_size):
            raw_response = generate_reward_code(prompt)
            code = extract_code_from_response(raw_response)
            candidates.append(code)

        # evaluate all candidates
        evaluated = []
        for i, code in enumerate(candidates):
            try:
                model, env = train_model(code, total_timesteps=train_steps)
                score = evaluate(model, env)
                evaluated.append((score, code, model, env))
                print(f"  Candidate {i}: Score = {score:.1f}")
            except Exception as e:
                print(f"  Candidate {i}: FAILED - {type(e).__name__}: {e}")
                evaluated.append((-1e9, code, None, None))

        # choose best
        evaluated.sort(key=lambda x: x[0], reverse=True)
        best_iter_score, best_iter_code, best_model, best_env = evaluated[0]

        if best_iter_score > best_score:
            best_score = best_iter_score
            best_code = best_iter_code

        # build reflection text
        if best_model and best_env:
            reflection = summarize_reward_components(best_model, best_env)
        else:
            reflection = "Model failed. Reward produced unstable behavior."

        prompt += f"""

Iteration {it} results:
Fitness: {best_iter_score}
Reward component summary:
{reflection}

Please improve the reward function.
"""

        print(f"[Iter {it}] Score={best_iter_score:.1f}")

    return best_code, best_score
