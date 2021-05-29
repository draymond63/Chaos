import numpy as np
import pandas as pd
from tqdm import tqdm


class Predictor():
	def __init__(self, series: pd.Series, dim=3) -> None:
		self.data = series
		self.dim = dim
		self.length = len(series) - dim

	def getRow(self, i):
		assert i < self.length, f"getRow index out of range: {i} must be less than {self.length}"
		return self.data[i:i + self.dim]


	def convertToMap(series: pd.Series, dim=3):
		vecs = np.ndarray((len(series) - dim, dim))
		for i in tqdm(range(len(series) - dim)):
			vecs[i] = series.loc[i : i+dim-1]
		return vecs



if __name__ == '__main__':
	df = pd.read_csv('weatherAUS.csv')
	df = df.filter(['Date', 'Location', 'Rainfall', 'RainToday', 'RainTomorrow'])
	df = df.head(50)
	vecs = convertToMap(df['Rainfall'])
	print(vecs[0:5])

