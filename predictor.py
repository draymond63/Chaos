import json
import pandas as pd
from tqdm import tqdm

def float_range(start, stop, step):
  while start < stop:
    yield round(float(start), 5)
    start += step

class Predictor():
    def __init__(self, Map='parabolic', Rs_file=None, threshold=0.5):
        self.threshold = threshold
        self.Rs = {}
        self.problems = []
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
            predictions.append(1 if x > self.threshold else 0)
        # Find the average result and return it
        avg_pred = sum(predictions)/len(predictions)
        return avg_pred # ! ROUND THIS

    # * Callable train function that chooses window if requested
    def train(self, r: iter, data: list, window=None, r_step=1e-4, x_step=.01, recursion=0):
        assert len(r) == 2, f"'r' range requires a beginning and end (length 2), not {r}"
        if window:  return self.rolling_train(r, data, window, r_step, x_step, recursion)
        else:       return self.absolute_train(r, data, r_step, x_step, True, False, recursion, True)

    # * Train an r range for each window and test it's prediction ability for the next data point    
    def rolling_train(self, r: iter, data: list, window=5, r_step=1e-4, x_step=.05, recursion=0):
        R_groups = []
        success = 0
        total = 0

        for i in tqdm([*range(len(data) - window - 1)]):
            # Get the data we need to train on
            w_end = i + window
            w_data = data[i:w_end]
            # Train R values and store it
            Rs = self.absolute_train(r, w_data, r_step, x_step, ret=True, perfection=False, recursion=recursion, use_tqdm=False)
            if len(Rs) != 1:
                R_groups.append(Rs)
                # Predict the next piece of data
                guess = self.predict(Rs=Rs)
                if guess == data[w_end + 1]:
                    success += 1
                # Write the current success rate
                total += 1
                tqdm.write(f'{round(success / total * 100, 2)}\t# R\'s: {len(Rs)}') 
        
        self.Rs = R_groups
        print('Prediction Rate:', success / len(data) * 100)
        return success / len(data) * 100

    # * Train an r range using all the data given
    def absolute_train(self, r: iter, data: list, r_step=1e-7, x_step=.01, ret=False, perfection=True, recursion=0, use_tqdm=True):
        # Optionally have the tqdm bar
        iteration = tqdm([*float_range(*r, r_step)]) if use_tqdm else float_range(*r, r_step)
        # Iterate through possible r-values
        Rs = {}
        best = None
        best_r_val = 0
        for current_r in iteration:
            # Try different inital x values
            for init_x in float_range(0 + x_step, 1 - x_step, x_step):
                r_is_good, fin_x = self.test_r_value(current_r, data, init_x, ret_x=True)
                # If there is a switch in the range
                if r_is_good == 1:
                    Rs[current_r] = fin_x
                # Save the best r
                if r_is_good > best_r_val:
                    best_r_val = r_is_good
                    best = {current_r: init_x}

        # Try a finer gradient for r if requested
        if recursion and len(Rs) == 0:
            Rs = self.absolute_train(r, data, r_step/10, x_step, True, perfection, recursion - 1, use_tqdm)
        if perfection:
            assert len(Rs), "No r-values found"
        if len(Rs):
            return Rs if ret else self.Rs.update(Rs)
        else:
            # tqdm.write(f'Using best r at {round(best_r_val * 100)} %,\tdata = {data}')
            self.problems.append(list(data))
            return best


    def test_r_value(self, r: float, data: list, x=0.5, ret_x=False) -> float:
        assert isinstance(data[0], (bool, int)), f"Data given must be a list of 1/0s or True/Falses, not {type(data[0])}"
        # Iterate through the logistic map, comparing it to the data
        for index, answer in enumerate(data):
            x = self.logistic_map(r, x)
            # See if its above or below the threshold
            if (x >= self.threshold) ^ answer or abs(x - self.threshold) < 0.05:
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


### Threshold = data, window = 5, days-ahead = 1: acurracy of 81 %


if __name__ == "__main__":
    # data = get_weather_AUS()
    # test = data[::7]
    # p = Predictor()
    # r = p.train([0, 4], test, window=6, r_step=1e-2, x_step=1e-3)
    # p.save('model.json')
    # print(r)

    length = 10
    used = []
    p = Predictor()

    import itertools
    for num_ones in tqdm(range(length + 1)):
        data = [1] * num_ones
        while len(data) < length:
            data.append(0)

        for d in itertools.permutations(data):
            if d not in used:
                # tqdm.write(str(d))
                used.append(d)
                p.absolute_train([3.5, 4], d, 1e-2, 1e-2, perfection=False, use_tqdm=False)
    
    print(len(p.problems))
    with open(f'problems-{length}.json', 'w') as f:
        json.dump(p.problems, f)
