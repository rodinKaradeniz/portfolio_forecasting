from flask import Flask, render_template
import data_handler
import prediction_handler

app = Flask(__name__)

@app.route("/")
def home():
    # TODO: initialize DB

    # return render_template("home.html")
    return 'Welcome to the Portfolio Forecasting API!'

@app.route('/getStockSummary/<ticker>', methods=['GET'])
def get_stock_summary(ticker):
    """Returns stock information summary.
    """
    stock_info = data_handler.get_stock_summary(ticker)

    # TODO: jsonify stock_info - create response
    return

@app.route('/getStockHistory/<ticker>', methods=['GET'])
def get_stock_history(ticker):
    """Returns past stock trend.
    """
    stock_history = data_handler.get_stock_history(ticker)

    # TODO: jsonify stock_history - create response
    return

@app.route('/getPredictions/<ticker>', methods=['GET'])
def get_predictions(ticker):
    """Returns the predicted stock price.
    """
    # Save past stock data to db, for prediction algorithms to access.
    data_handler.get_stock_history(ticker, save_to_db=True)

    price_prediction = prediction_handler.get_price_prediction(ticker)
    sentiment_prediction = prediction_handler.get_sentiment_prediction(ticker)

    # TODO: jsonify predictions - create response
    return

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)