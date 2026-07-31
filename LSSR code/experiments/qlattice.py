import feyn
import numpy as np
import pandas as pd

# 1. Generate data and run the QLattice
np.random.seed(42)
X = np.random.normal(size=100)
y = np.exp(X)
data = pd.DataFrame(X, columns=['X'])
data['y'] = y

ql = feyn.QLattice(random_seed=666)
models = ql.auto_run(data, 'y', n_epochs=2)
best_model = models[0]

# 2. Convert to SymPy for readable equation
# signif controls decimal precision; include_weights=False removes coefficient display
sympy_model = best_model.sympify(signif=3, include_weights=False)

# 3. View the equation
print(sympy_model.as_expr())
