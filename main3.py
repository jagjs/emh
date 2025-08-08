import yfinance as yf
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import mplcursors as mpl
import pandas as pd
from keras.src.callbacks import EarlyStopping
from keras.src.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import random
import os
import requests
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pytz
from sklearn.metrics import mean_absolute_error
import optuna


# Load FinBERT
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
modell = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")


# Convert class probabilities to a score between -1 and 1
def sentiment_score_from_probs(probs):
    labels = ['positive', 'neutral', 'negative']
    score = probs[0][0] * 1 + probs[0][1] * 0 + probs[0][2] * -1
    return float(score)  # continuous value between -1 and 1

# Main sentiment function
def get_finbert_sentiment_score(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = modell(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
    return sentiment_score_from_probs(probs)




tickers = ["AAPL"]
directional_accuracies = {}



api_key = 'this is where the api goes'





#Key-word bank for technology words.
tech_keywords = [
    "technology", "tech stocks", "Nasdaq", "semiconductor", "AI", "chip industry",
    "Silicon Valley", "big tech", "cloud computing", "machine learning"
]

oil_keywords = [
    "fuel", "crude oil", "oil", "natural gas", "refining",
    "offshore drilling", "LNG", "petrochemicals", "oil reserves", "renewable energy",
    "biofuels", "net zero", "emissions", "fossil fuels", "carbon neutrality"

]

#getting the top 100 headlines for the ticker bounded by the timeperiod set
def fetch_top_headlines(ticker, date_start, date_finish):
    url = 'https://newsapi.org/v2/everything'

    # Build query string: ticker-specific + tech keywords
    keyword_str = " OR ".join(tech_keywords)
    query = f'{ticker} stock OR {ticker} shares OR {ticker} company OR {keyword_str}'

    params = {
        'q': query,
        'from': date_start,
        'to': date_finish,
        'language': 'en',
        'sortBy': 'relevancy',
        'pageSize': 100,
        'apiKey': api_key
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data.get("status") != "ok":
        print(f"Error fetching news for {ticker}: {data}")
        return []

    return data.get("articles", [])








for ticker in tickers:

    directional_accuracies[ticker] = []
    tf.keras.backend.clear_session()

    all_articles = []

    date_start = "2025-06-24"
    date_finish = "2025-07-23"

    articles = fetch_top_headlines(ticker, date_start,date_finish)
    for article in articles:
        title = article.get("title", "")
        try:
            score = get_finbert_sentiment_score(title)
        except Exception as e:
            print(f"Error processing title: {title!r}, error: {e}")
            continue
        score = get_finbert_sentiment_score(title)
        all_articles.append({
            "ticker": ticker,
            "title": article.get("title"),
            "publishedAt": article.get("publishedAt"),
            "source": article.get("source", {}).get("name"),
            "url": article.get("url"),
            "sentiment_score": score
        })


    df_articles = pd.DataFrame(all_articles)

        # Retrieve Apple stock data (AAPL)
    stock = yf.Ticker(ticker)

    # Get historical market data for the last 1 day with hourly intervals
    stock_data = stock.history(start=date_start, end=date_finish, interval="15m")


    # 2. Get the stock's timezone from yfinance
    market_tz = stock_data.index.tz

    # 3. Convert article timestamps to datetime, localize to UTC, and convert to market timezone
    df_articles['publishedAt'] = pd.to_datetime(df_articles['publishedAt'], utc=True)
    df_articles['publishedAt'] = df_articles['publishedAt'].dt.tz_convert(market_tz)

    # 3. Round to 30-min interval
    df_articles['rounded_time'] = df_articles['publishedAt'].dt.ceil('15T')

    # 4. Group by rounded time and compute average sentiment
    sentiment_by_time = df_articles.groupby('rounded_time')['sentiment_score'].mean()



    # Align sentiment index with stock_data index by reindexing and forward fill
    sentiment_by_time = sentiment_by_time.reindex(stock_data.index, method='ffill')

    # Add sentiment as a new column in stock_data
    stock_data['sentiment_score'] = sentiment_by_time

    stock_data['sentiment_score'].fillna(0, inplace=True)






    for run in range(3):

        tf.keras.backend.clear_session()















        # Initialize lists
        time_step = []




        # Create multivariate series with mid price and sentiment
        stock_data['Mid_Price'] = (stock_data['Open'].values + stock_data['Close'].values)/2
        features = stock_data[['Mid_Price', 'sentiment_score']].values  # shape: (n_samples, 2)


        time_step = stock_data.index.tolist()




        # Convert lists to numpy arrays
        time = np.array(time_step)
        series = np.array(stock_data['Mid_Price'])








        # Define the split time
        split_ratio = 0.8
        split_time = int(len(time) * split_ratio)
        val_time = int(len(time) * 0.2)

        scaler = MinMaxScaler()
        price_scaler = MinMaxScaler()

        # Get the train set
        time_train = time[:split_time]
        x_train = features[:split_time]

        # Get the validation set
        time_valid = time[split_time:]
        x_valid = features[split_time:]



        #Prevents data leakage by adjusting scalar to training data

        scaler.fit(x_train)

        price_train = stock_data['Mid_Price'].values[:split_time].reshape(-1, 1)
        price_scaler.fit(price_train)



        x_train = scaler.transform(x_train)
        x_valid = scaler.transform(x_valid)












        #Process data so it can be input into model
        def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
            """Generates dataset windows

            Args:
              series (array of float) - contains the values of the time series
              window_size (int) - the number of time steps to include in the feature
              batch_size (int) - the batch size
              shuffle_buffer(int) - buffer size to use for the shuffle method

            Returns:
              dataset (TF Dataset) - TF Dataset containing time windows
            """

            # Generate a TF Dataset from the series values
            dataset = tf.data.Dataset.from_tensor_slices(series)

            # Window the data but only take those with the specified size
            dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)

            # Flatten the windows by putting its elements in a single batch
            dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

            # Create tuples with features and labels
            dataset = dataset.map(lambda window: (window[:-1], window[-1, 0]))

            # Shuffle the windows
            dataset = dataset.shuffle(shuffle_buffer)

            # Create batches of windows
            dataset = dataset.batch(batch_size)

            # Optimize the dataset for training
            dataset = dataset.cache().prefetch(1)

            return dataset

        def model_forecast(model, series, window_size, batch_size):
            """Uses an input model to generate predictions on data windows

            Args:
              model (TF Keras Model) - model that accepts data windows
              series (array of float) - contains the values of the time series
              window_size (int) - the number of time steps to include in the window
              batch_size (int) - the batch size

            Returns:
              forecast (numpy array) - array containing predictions
            """



            # Generate a TF Dataset from the series values
            dataset = tf.data.Dataset.from_tensor_slices(series)

            # Window the data but only take those with the specified size
            dataset = dataset.window(window_size, shift=1, drop_remainder=True)

            # Flatten the windows by putting its elements in a single batch
            dataset = dataset.flat_map(lambda w: w.batch(window_size))

            # Create batches of windows
            dataset = dataset.batch(batch_size).prefetch(1)

            # Get predictions on the entire dataset
            forecast = model.predict(dataset, verbose=0)

            return forecast



        def evaluate_directional_accuracy(y_actual, y_pred, verbose=True):
            """
            Computes the directional accuracy between actual and predicted price sequences.

            Parameters:
                y_actual (array-like): True values (e.g., aligned_x_valid)
                y_pred (array-like): Predicted values (e.g., rescaled_results)
                verbose (bool): If True, prints the accuracy. Defaults to True.

            Returns:
                float: Directional accuracy (0.0 - 1.0)
            """
            # Convert to Series
            y_actual = pd.Series(y_actual).reset_index(drop=True)
            y_pred = pd.Series(y_pred).reset_index(drop=True)

            # Create DataFrame
            df = pd.DataFrame({
                'Actual': y_actual,
                'Predicted': y_pred
            })

            # Compute directions (+1 for increase, -1 for decrease, 0 for no change)
            df['Actual_Dir'] = df['Actual'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            df['Pred_Dir'] = df['Predicted'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

            # Drop first row (diff produces NaN)
            df = df.dropna()

            # Compare directions
            df['Correct_Direction'] = (df['Actual_Dir'] == df['Pred_Dir']).astype(int)

            # Accuracy score
            accuracy = df['Correct_Direction'].mean()

            if verbose:
                print(f"Directional Accuracy: {accuracy:.2%}")


            return accuracy









        # Parameters
        window_size = int(len(time) * 0.15) #so that we have a set number of points before a value we look at to learn patterns from etc.
        batch_size = 32 #updating of weights/parameters every 32 sets of windows
        shuffle_buffer_size = split_time #shuffles around the windowed sets to add randomness but for efficiency, it doesnt shuffle all of the windowed sets


        # Generate the dataset windows
        dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)





        # Initialize the optimizer
        optimizer = tf.keras.optimizers.SGD(momentum=0.9) # uses stoichastic gradient descent during backwards propogation to minimise loss by altering/optimising the weightage of parameters








        # Define EarlyStopping and ModelCheckpoint callbacks
        early_stopping = EarlyStopping(monitor='val_mae', mode='min', patience=10, restore_best_weights=True)
        checkpoint = ModelCheckpoint('best_model.h5', monitor='val_mae', save_best_only=True, mode='min', verbose=1) #saves the model with the weights whenever there is an improvement in val_mae (decreases), but it doesnt resave if val_mae stays the same







        val_dataset = windowed_dataset(x_valid, window_size, batch_size, shuffle_buffer_size)
        results = []







        def evaluate_true_random_walk(val_dataset, noise_scale=1.0):
            y_true_rw = []
            y_pred_rw = []

            # Estimate standard deviation of price changes from training set
            price_changes = np.diff(series[:split_time])
            std_dev = np.std(price_changes)

            for x_batch, y_batch in val_dataset:
                last_prices = x_batch[:, -1, 0].numpy()  # last price in each window

                # Add random noise to simulate random walk
                noise = np.random.normal(loc=0.0, scale=std_dev * noise_scale, size=last_prices.shape)
                random_walk_preds = last_prices + noise

                y_true_rw.extend(y_batch.numpy().flatten()) #Preparation for the format that directional_accuracy() function expects
                y_pred_rw.extend(random_walk_preds.flatten())

            # Store for later use

            random_walk_y_true_series = pd.Series(y_true_rw)
            random_walk_y_pred_series = pd.Series(y_pred_rw)

            val_mae = mean_absolute_error(y_true_rw, y_pred_rw)

            results.append({
                'model_name': 'Model_1_TrueRandomWalk',
                'mean_val_mae': val_mae,
                'val_mae_list': val_mae,
                'learning_rate': 0,
                'filters': 0,
                'kernel_size': 0,
                'lstm_units_1': 0,
                'lstm_units_2': 0
            })


            return val_mae, random_walk_y_true_series, random_walk_y_pred_series

        val_maee, random_walk_y_true_series, random_walk_y_pred_series = evaluate_true_random_walk(val_dataset, noise_scale=1.0)










        def build_model_2(hp):
            tf.keras.backend.clear_session()
            model = tf.keras.models.Sequential([
                tf.keras.Input(shape=(hp['window_size'], 2)),
                tf.keras.layers.Conv1D(filters=hp['filters'], kernel_size=hp['kernel_size'], activation='relu'),
                tf.keras.layers.MaxPooling1D(pool_size=2),
                tf.keras.layers.LSTM(hp['lstm_units_1'], return_sequences=True),
                tf.keras.layers.LSTM(hp['lstm_units_2']),
                tf.keras.layers.Dense(1),
                tf.keras.layers.Lambda(lambda x: x * 100.0)
            ])
            return model


        def build_model_3(hp):
          tf.keras.backend.clear_session()
          model = tf.keras.models.Sequential([
             tf.keras.Input(shape=(hp['window_size'], 2)),
             tf.keras.layers.LSTM(hp['lstm_units_1']),
             tf.keras.layers.Dense(1),
             tf.keras.layers.Lambda(lambda x: x * 100.0)
             ])
          return model

        def build_model_4(hp):
          tf.keras.backend.clear_session()
          model = tf.keras.models.Sequential([
             tf.keras.Input(shape=(hp['window_size'], 2)),
             tf.keras.layers.LSTM(hp['lstm_units_1'], return_sequences=True),
             tf.keras.layers.LSTM(hp['lstm_units_2']),
             tf.keras.layers.Dense(1),
             tf.keras.layers.Lambda(lambda x: x * 100.0)
            ])
          return model




        model_builders = {
            "Model_2_StackedLSTMWithCNN": build_model_2,
            "Model_3_SingleLSTM": build_model_3,
            "Model_4_StackedLSTM": build_model_4,


        }

        def filter_outliers_iqr(val_mae_list):
            q1 = np.percentile(val_mae_list, 25)
            q3 = np.percentile(val_mae_list, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return [x for x in val_mae_list if lower_bound <= x <= upper_bound]


        def rmse(y_true, y_pred):
            return np.sqrt(np.mean((y_true - y_pred) ** 2))




        def objective(trial, model_fn, model_name, dataset, x_train):
            tf.keras.backend.clear_session()

            # Suggest hyperparameters
            learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=False)
            filters = trial.suggest_categorical("filters", [16, 32, 64])
            kernel_size = trial.suggest_categorical("kernel_size", [2, 3, 5])
            lstm_units_1 = trial.suggest_categorical("lstm_units_1", [16, 32, 64])
            lstm_units_2 = trial.suggest_categorical("lstm_units_2", [16, 32, 64])
            window_size = int(len(time) * 0.15)
            batch_size = 32

            hp = {
                'learning_rate': learning_rate,
                'filters': filters,
                'kernel_size': kernel_size,
                'lstm_units_1': lstm_units_1,
                'lstm_units_2': lstm_units_2,
                'window_size': window_size,
                'batch_size': batch_size
            }

            val_mae_list = []
            num_repeats = 3

            for repeat in range(num_repeats):
                tf.keras.backend.clear_session()

                print(f"Repeat {repeat+1}: lr={learning_rate:.6f}, filters={filters}, kernel={kernel_size}, LSTM1={lstm_units_1}, LSTM2={lstm_units_2}")

                model = model_fn(hp)
                optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
                model.compile(loss=tf.keras.losses.Huber(),
                optimizer=optimizer,
                metrics=["mae"])

                dataset = dataset.repeat()
                steps_per_epoch = (len(x_train)) // batch_size

                val_dataset = windowed_dataset(x_valid, window_size, batch_size, shuffle_buffer_size)
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=10, mode="min", restore_best_weights=True)
                checkpoint = tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_mae", mode="min", verbose=1)
                # Train the model
                history = model.fit(dataset,epochs=100, steps_per_epoch=steps_per_epoch, validation_data=val_dataset,  callbacks=[early_stopping, checkpoint])
                best_model = tf.keras.models.load_model('best_model.h5')

                best_val_mae = best_model.evaluate(val_dataset, verbose=0)[1]  # index 1 = mae
                val_mae_list.append(best_val_mae)
                tf.keras.backend.clear_session()

                filtered = filter_outliers_iqr(val_mae_list)
                mean_val_mae = np.mean(filtered)
                print(f"✅ mean val_mae: {mean_val_mae:.5f}")
                results.append({
                    'model_name': model_name,
                    'mean_val_mae': mean_val_mae,
                    'val_mae_list': val_mae_list,
                    'learning_rate': learning_rate,
                    'filters': filters,
                    'kernel_size': kernel_size,
                    'lstm_units_1': lstm_units_1,
                    'lstm_units_2': lstm_units_2
                    })
            return mean_val_mae























        for model_name, model_fn in model_builders.items():
            tf.keras.backend.clear_session()
            print(f"\n🔍 Starting Optuna search for: {model_name}")
            print(f"\n This is run: {run}")
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: objective(trial, model_fn, model_name, dataset, x_train), n_trials=20)










        #Report best config
        best_run = min(results, key=lambda x: x['mean_val_mae'])

        print("\n Best hyperparameters based on mean val_mae:")
        print(f"filters: {best_run['model_name']}")
        print(f"mean val_mae: {best_run['mean_val_mae']:.5f}")
        print(f"learning_rate: {best_run['learning_rate']:.6f}")
        print(f"filters: {best_run['filters']}")
        print(f"kernel_size: {best_run['kernel_size']}")
        print(f"lstm_units_1: {best_run['lstm_units_1']}")
        print(f"lstm_units_2: {best_run['lstm_units_2']}")
        print(f"val_mae values: {best_run['val_mae_list']}")

        print(results)





















        if best_run['model_name'] == 'Model_1_TrueRandomWalk':
            accuracy = evaluate_directional_accuracy(random_walk_y_true_series, random_walk_y_pred_series, verbose=True)
            directional_accuracies[ticker].append(accuracy)






        else:
            # Reduce the original series
            forecast_series = series[split_time-window_size: -1]


            hp_best = {
                'window_size': int(len(time) * 0.15),  # keep same window size as before
                'filters': best_run['filters'],
                'kernel_size': best_run['kernel_size'],
                'lstm_units_1': best_run['lstm_units_1'],
                'lstm_units_2': best_run['lstm_units_2']
            }

            model_builder = model_builders[best_run['model_name']]
            best_model = model_builder(hp_best)

            optimizer = tf.keras.optimizers.SGD(learning_rate=best_run['learning_rate'], momentum=0.9)
            best_model.compile(loss=tf.keras.losses.Huber(),
                optimizer=optimizer,
                metrics=["mae"])


            # Prepare training and validation datasets
            train_dataset = windowed_dataset(x_train, hp_best['window_size'], batch_size, shuffle_buffer_size)
            train_dataset = train_dataset.repeat()
            val_dataset = windowed_dataset(x_valid, hp_best['window_size'], batch_size, shuffle_buffer_size)

            # Retrain model on training data with early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=10, mode='min', restore_best_weights=True)

            steps_per_epoch = (len(x_train)) // batch_size

            history = best_model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=100,
                steps_per_epoch=steps_per_epoch,
                callbacks=[early_stopping]
            )



            scaled_series = scaler.transform(features)
            forecast_series = scaled_series[split_time - window_size : -1]


            # Use helper function to generate predictions
            forecast_valid = model_forecast(best_model, forecast_series, window_size, batch_size)

            # Drop single dimensional axis
            results_valid = forecast_valid.squeeze()
            rescaled_results = price_scaler.inverse_transform(results_valid.reshape(-1, 1)).squeeze()

            print(type(rescaled_results))
            print(rescaled_results.shape)
            print(rescaled_results.ndim)

            aligned_x_valid = series[split_time : split_time + len(rescaled_results)]

            print(rescaled_results)
            error = rmse(aligned_x_valid, rescaled_results)
            print(f"RMSE: {error}") #RMSE IS THE WRONG METRIC SINCE WE AREN'T TRYING TO PREDICT PRICE, BUT RATHER DIRECTION.
            print(f"filters: {best_run['model_name']}")
            print(f"Train start: {x_train[0]}, end: {x_train[-1]}")
            print(f"Valid start: {x_valid[0]}, end: {x_valid[-1]}")
            print(f"Train shape: {x_train.shape}, Valid shape: {x_valid.shape}")
            print("First 5 time values:", time[:5])
            print("Last 5 time values:", time[-5:])










            accuracy = evaluate_directional_accuracy(aligned_x_valid, rescaled_results, verbose=True)
            directional_accuracies[ticker].append(accuracy)






print("Directional Accuracy Summary:")
for ticker, acc_list in directional_accuracies.items():
    print(f"\n{ticker}:")
    for i, acc in enumerate(acc_list, 1):
        print(f"  Run {i}: {acc:.4%}")

print(all_articles)
print(stock_data)
print(results)