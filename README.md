# Stock Price Prediction with PyTorch

This project uses recurrent neural networks to predict Amazon (AMZN) stock closing prices using historical price data. Two neural network architectures, Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU), were implemented and compared using PyTorch.

## Project Overview

The objective of this project is to use historical Amazon stock prices to predict the next trading day's closing price.

The models use the previous 20 trading days of closing prices as input to predict the closing price of the following trading day.

Historical Amazon stock data from 2015 through 2024 was obtained using the `yfinance` Python library.

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- yfinance
- Google Colab

## Data

Amazon (AMZN) historical stock data was downloaded directly using `yfinance`.

The original dataset includes:

- Open price
- High price
- Low price
- Closing price
- Trading volume

Only the daily closing price was used as the input feature for the prediction models.

## Data Preprocessing

The closing-price data was normalized to a range of -1 to 1 using `MinMaxScaler`.

A sliding-window approach was then used to create sequences. Each input sequence contains the previous 20 trading days of closing prices, while the target is the closing price of the following trading day.

The dataset was divided chronologically into:

- 80% training data
- 20% testing data

## Models

### LSTM

The Long Short-Term Memory model contains:

- 2 LSTM layers
- 32 hidden units
- 1 output layer
- Mean Squared Error loss
- Adam optimizer
- 100 training epochs

### GRU

The Gated Recurrent Unit model uses the same general configuration:

- 2 GRU layers
- 32 hidden units
- 1 output layer
- Mean Squared Error loss
- Adam optimizer
- 100 training epochs

## Results

| Model | MSE | RMSE |
|------|------:|------:|
| LSTM | 25.53 | $5.05 |
| GRU | 20.66 | $4.54 |

The GRU achieved lower MSE and RMSE values than the LSTM and therefore produced slightly better predictions on the test dataset.

Both models generally followed the direction of Amazon's actual stock price, although their predictions were smoother during sudden price movements.

## Running the Project

The project can be run using Google Colab or a local Python environment.

Required Python libraries include:

```python
numpy
pandas
matplotlib
torch
scikit-learn
yfinance
```

When using Google Colab, `yfinance` can be installed with:

```python
!pip install yfinance
```

The notebook automatically downloads the required Amazon stock-price data, so no separate dataset download is required.

## Evaluation

Model performance was evaluated on unseen test data using:

**Mean Squared Error (MSE)** — measures the average squared difference between predicted and actual values.

**Root Mean Squared Error (RMSE)** — expresses prediction error on approximately the same scale as the original stock prices.

The GRU achieved an RMSE of $4.54 compared with $5.05 for the LSTM.

## Limitations

The models use only historical closing prices. They do not incorporate other factors that may influence stock prices, such as trading volume, company earnings, financial news, macroeconomic conditions, or market sentiment.

Therefore, this project demonstrates the application of recurrent neural networks to financial time-series data rather than providing a real-world trading strategy.

## Conclusion

Both LSTM and GRU neural networks were successfully applied to Amazon stock-price prediction. The GRU performed slightly better on the test dataset, achieving lower MSE and RMSE values.

The project demonstrates the complete machine-learning workflow from data collection and preprocessing through model training, prediction, visualization, and performance evaluation.
