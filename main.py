from utils import *

fund_ticker = "xlu"
days = 100
start_date = (datetime.now() - timedelta(days=5) - timedelta(days=days)).strftime('%Y-%m-%d')
end_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')

X, y = get_batch(fund_ticker=fund_ticker)
X, y = create_windows(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
breakpoint()
model = LSTM_model(X_train, y_train)
fit_evaluate_LSTM(X_train, y_train, X_test, y_test, model, "LSTM")
breakpoint()