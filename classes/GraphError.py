import numpy as np


class GraphSquareError:
	def __init__(self, fig, ax, line):
		# figura relacionada al grafico
		self.__fig = fig
		self.__ax = ax
		# linea del error
		self.__line = line
		# datos de los errores
		self.__data = []

	def add_data(self, data_add):
		self.__data.append(data_add)

	# funcion que actualiza la grafica del error
	def update_graph(self, epochs):
		line_points = np.arange(epochs)
		self.__ax.clear()
		self.__ax.plot(line_points, self.__data)