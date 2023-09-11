from utils import *

fund_ticker = "xlu"

X, y = get_batch(fund_ticker)
X, y = create_windows(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

model = LSTM_model(X_train, y_train)
fit_evaluate_LSTM(X_train, y_train, X_test, y_test, model, "LSTM")