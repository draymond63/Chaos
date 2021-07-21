import numpy as np
import pandas as pd
from tqdm import tqdm

class Predictor():
	def __init__(self, series: pd.Series) -> None:
		self.mapp = self._formatData(series)

	def _formatData(self, series: pd.Series) -> list:
		# Group data into buckets
		# ? Perform distance calculations or group buckets logarithmically ?
		# Make sure the data is clean
		s = series.dropna().reset_index(drop=True)
		return np.round(s.to_numpy().tolist())

	def getVec(self, i, length):
		assert i < len(self.mapp) - length, \
			f"Index out of range: {i} must be less than {len(self.mapp) - length}"
		return self.mapp[i:i + length]

	def getVecCoord(self, point, coord):
		assert point < len(self.mapp) - coord, \
			f"Point out of range: {point} must be less than {len(self.mapp) - coord}"
		return self.mapp[point + coord]

	def predict(self, data: list) -> float:
		data = np.round(data)
		# Starting indices of all the possible vecs that could match the data
		guesses = []
		for i in range(len(self.mapp) - len(data)):
			if np.array_equal(data, self.getVec(i, len(data))): # np.allclose
				# Get the last point as the guess for what is happening next
				guess = self.getVecCoord(i, len(data))
				guesses.append(guess)
		if len(guesses):
			return np.median(guesses)


def test_model(test_len=100, vec_len=50):
	# Gather data
	df = pd.read_csv('weatherAUS.csv')['Rainfall']
	# Vec length is subtracted from the test length
	train = df.head(len(df) - test_len - vec_len)
	test = df.tail(test_len + vec_len).dropna().tolist()
	pred = Predictor(train)

	correct = 0
	incorrect = 0
	for vec_idx in tqdm(range(test_len)):
		current = test[vec_idx:vec_idx + vec_len]
		if np.any(current):
			guess = pred.predict(current)
			answer = test[vec_idx + vec_len]
			if guess == answer:
				correct += 1
			else:
				incorrect += 1
	print(correct/(correct + incorrect))

if __name__ == '__main__':
	test_model(500)
