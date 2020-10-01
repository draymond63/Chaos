import json
import pandas as pd
from tqdm import tqdm

def float_range(start, stop, step):
  while start < stop:
    yield float(start)
    start += step

class Predictor():
    def __init__(self, Map='parabolic', Rs_file=None):
        self.Rs = {}
        # Choosing the logistic map
        if Map == 'parabolic':
            self.logistic_map = self._parabolic_map
        else:
            raise NotImplementedError('Maps other than parabolic are not implemented currently')
        # Load the model if requested
        if Rs_file:
            with open(Rs_file, 'r') as f:
                self.Rs = json.load(f)

    def save(self, Rs_file):
        assert hasattr(self, 'Rs'), "Model must be trained or loaded before saving"
        with open(Rs_file, 'w') as f:
                json.dump(self.Rs, f)

    def test(self, data):
        correct = 0
        for i, answer in enumerate(data):
            guess = round(self.predict(i))
            if guess == answer:
                correct += 1
        return correct / len(data)

    def predict(self, time_steps=1, Rs=None) -> float:
        assert hasattr(self, 'Rs'), "Model must be trained or loaded before predicting"
        # Give the option to provide a dictionary
        if not Rs:
            Rs = self.Rs
        # Gather all the predictions
        predictions = []
        for r in Rs:
            x = Rs[r] # Grab value that we left off at
            for _ in range(time_steps):
                x = self.logistic_map(r, x)
            # Grab the final prediction for that branch
            predictions.append(x)
        # Find the average result and return it
        avg_pred = sum(predictions)/len(predictions)
        return 1 if avg_pred > self.threshold else 0


    def train(self, r: iter, data: list, window=None, r_step=1e-4, x_step=.01, threshold='data'):
        assert len(r) == 2, f"'r' range requires a beginning and end (length 2), not {r}"
        self.r_step = r_step
        self.x_step = x_step
        # Set the threshold
        if threshold == 'data':
            assert 1 in data and 0 in data, "To use data for threshold, it must be comprised of 1s and 0s"
            self.threshold = 1 - sum(data) / len(data) # ! Get percentage of 1s in data
            print(self.threshold)
        else:
            assert isinstance(threshold, float), f"Threshold must be 'data' or a float, not {threshold}"
            self.threshold = threshold

        if window:  return self.rolling_train(r, data, window, r_step, x_step, threshold)
        else:       return self.absolute_train(r, data, r_step, x_step, threshold)

    # Train an r range for each window and test it's prediction ability for the next data point    
    def rolling_train(self, r: iter, data: list, window=10, r_step=1e-4, x_step=.05, threshold=0.5):
        R_groups = []
        success = 0

        for i in tqdm([*range(len(data) - window - 1)]):
            # Get the data we need to train on
            w_end = i + window
            w_data = data[i:w_end]
            # Train R values and store it
            Rs = self.absolute_train(r, w_data, r_step, x_step, threshold, ret=True, perfection=False)
            R_groups.append(Rs)
            # Predict the next piece of data
            guess = self.predict(Rs=Rs)
            if guess == data[w_end + 1]:
                success += 1
            # Write the current success rate
            # tqdm.write(f'{success / (i + 1) * 100}') 
        
        self.Rs = R_groups
        print('Prediction Rate:', success / len(data) * 100)
        return success / len(data) * 100



    def absolute_train(self, r: iter, data: list, r_step=1e-7, x_step=.01, threshold=0.5, ret=False, perfection=True):
        Rs = {}
        # Iterate through possible r-values
        best = None
        best_r_val = 0
        for current_r in float_range(*r, r_step):
            # Try different inital x values
            for init_x in float_range(0, 1, x_step):
                r_is_good, fin_x = self.test_r_value(current_r, data, init_x, ret_x=True)
                # If there is a switch in the range
                if r_is_good == 1:
                    Rs[current_r] = fin_x
                # Save the best r
                if r_is_good > best_r_val:
                    best_r_val = r_is_good
                    best = {current_r: init_x}

        if perfection:
            assert len(Rs), "No r-values found"
        if len(Rs):
            return Rs if ret else self.Rs.update(Rs)
        else:
            tqdm.write(f'Using best r at {best_r_val * 100} %,\tdata = {data}')
            return best


    def test_r_value(self, r: float, data: list, x=0.5, ret_x=False) -> float:
        assert isinstance(data[0], (bool, int)), f"Data given must be a list of 1/0s or True/Falses, not {type(data[0])}"
        # Iterate through the logistic map, comparing it to the data
        for index, answer in enumerate(data):
            x = self.logistic_map(r, x)
            # Is below threshold and answer = 1 or above and answer = 0, it's wrong
            if (x > self.threshold) ^ answer:
                progress = index/len(data) # Return how far it got (as a decimal)
                return (progress, x) if ret_x else progress
        return (1, x) if ret_x else 1

    def _parabolic_map(self, r: float, x: float) -> float:
        return r * x * (1 - x)



def get_weather_AUS(location = 'Albury'):
    df = pd.read_csv('weatherAUS.csv')
    df = df.filter(['Date', 'Location', 'RainTomorrow'], axis=1)
    # Look at one location
    # print(df['Location'].unique())
    df = df[df['Location'] == location]
    # Sort and drop the dates
    df.sort_values('Date', inplace=True)
    
    data = df['RainTomorrow'].apply(lambda x: 1 if x == 'Yes' else 0)
    return list(data)


### Threshold: Data
# WINDOW: 5       6
#-------------------------------------
# 1:    81.0    __._
# 7:    59.8    __._
# 14:   57.8    __._

### Threshold: 0.5
# WINDOW: 5       6
#-------------------------------------
# 1:    __._    __._
# 7:    __._    __._
# 14:   __._    __._


if __name__ == "__main__":
    data = get_weather_AUS()
    p = Predictor()

    days_ahead = (1, 7, 14)
    windows = (6, 8, 10)
    thresholds = ('data', 0.5) # ! SEE WHETHER IT WENT LEFT OR RIGHT, NOT ABOVE OR BELOW THRESHOLD

    results = []

    for t in thresholds:
        if t == 0.5:
            window = (5, *windows)
        for s in days_ahead:
            for w in windows:
                test = data[::s]
                # Shave off the later ones
                if s == 1:
                    test = test[:300]
                print('STARTING (t, s, w, l):', t, s, w, len(test))
                r = p.train([3, 4], test, window=w, threshold=t)

                results.append({
                    'threshold': t,
                    'days-ahead': s,
                    'window': w,
                    'length': len(test),
                    'success': r,
                })

    with open('results.json', 'r') as f:
        json.dump(results, f)


