from utils import *
import os

"""sizes = {}
for tick in [s[:-7] for s in os.listdir("./data/ETFs")]:
    X, y, max_window = get_batch(tick)
    sizes[tick] = max_window"""

# scaler = RobustScaler(); scaler = scaler.fit(X)
X_raw, y_raw = get_batch("aok")
X, y = create_windows(X_raw, y_raw)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

model = LSTM_model(X_train, y_train)
fit_evaluate_LSTM(X_train, y_train, X_test, y_test, model, "aok")
model.save(f'models/aok')