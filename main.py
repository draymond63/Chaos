import numpy as np
import pandas as pd
from tqdm import tqdm


class Predictor():
	def __init__(self, series: pd.Series) -> None:
		self.mapp = self._formatData(series)

	def _formatData(self, series: pd.Series) -> pd.Series:
		# Group data into buckets
		# ?Perform distance calculations or group buckets logarithmically?
		# Make sure the data is clean
		s = series.dropna().reset_index(drop=True)
		return np.round(s)

	def getPoint(self, i, length):
		assert i < len(self.mapp) - length, \
			f"Index out of range: {i} must be less than {len(self.mapp) - length}"
		return self.mapp[i:i + length]

	def getPointCoord(self, point, coord):
		assert point < len(self.mapp) - coord, \
			f"Point out of range: {point} must be less than {len(self.mapp) - coord}"
		return self.mapp[point + coord]

	def predict(self, data: list) -> float:
		data = self._formatData(data)
		# Starting indices of all the possible vecs that could match the data
		guesses = [*range(len(self.mapp) - len(data) - 1)]
		# Compare each coordinate of data with a vector in the data
		for coord_idx, coord in enumerate(data):
			for vec_idx, guess in enumerate(guesses):
				# If this vector no longer matches the data, remove it
				if coord != self.getPointCoord(guess, coord_idx):
					guesses.pop(vec_idx) # ? This might be weird because we are enumerating through the data we are editing
		# get the last point as the guess for what is happening next in the data
		answers = [self.getPointCoord(v, len(data)) for v in guesses]
		if len(answers):
			return np.median(answers)


def test_model(vec_len=50):
	# Gather data
	df = pd.read_csv('weatherAUS.csv')['Rainfall']
	split = round(len(df) * 0.95)
	train, test = df.head(split), df.tail(len(df) - split)
	pred = Predictor(train)

	correct = 0
	wrong_idxs = []
	for vec_idx in tqdm(range(len(test) - vec_len)):
		current = df.loc[vec_idx:vec_idx + vec_len - 1]
		guess = pred.predict(current)
		answer = df.loc[vec_idx + vec_len]
		if guess == answer:
			correct += 1
		else:
			wrong_idxs.append(vec_idx)
	print(wrong_idxs)
	print(correct/(len(test) - vec_len))

if __name__ == '__main__':
	test_model()

