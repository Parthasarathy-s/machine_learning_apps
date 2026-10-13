import numpy as np

x = np.array([5, 1, 0])           # features
w = np.array([0.4, 1.5, -2.0])    # weights
b = -1.0                          # bias

z = np.dot(x, w) + b              # 2.5 >> z = w₁x₁ + w₂x₂ + ... + b
p = 1 / (1 + np.exp(-z))          # 0.924 >> p = 1 / (1 + e⁻ᶻ)
print(z, p)


"""
Step 1: Convert the email into numerical features
x1 = 5     # Number of links
x2 = 1     # Contains "free" (yes)
x3 = 0     # Sender is known (no)

Step 2: Learned weights from training
w1 = 0.4    # More links increase spam score
w2 = 1.5    # "Free" increases spam score
w3 = -2.0   # Known sender decreases spam score
b  = -1.0   # Bias (baseline score)

Step 3: Calculate the weighted sum
z = w1*x1 + w2*x2 + w3*x3 + b
z = (0.4*5) + (1.5*1) + (-2.0*0) + (-1.0)
z = 2.0 + 1.5 + 0 - 1.0
z = 2.5

Step 4: Apply the sigmoid function
p = 1 / (1 + e**(-z))
p = 1 / (1 + e**(-2.5))
p ≈ 0.924

Step 5: Classify using a 0.5 threshold
Since 0.924 > 0.5, predict SPAM!

Note: The weights are illustrative. A real model
learns them from training data.
"""
