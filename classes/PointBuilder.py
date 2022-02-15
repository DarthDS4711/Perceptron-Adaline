import numpy as np


class PointBuilder:
    def __init__(self, fig, ax, plot, another, line, fig_test):
        # figuras del canvas
        self.fig = fig
        self.ax = ax
        self.plot = plot
        self.another = another
        self.fig_test = fig_test
        # figuras del barrido
        self.fig_x = self.ax.scatter([], [], color='darkred', marker='.')
        self.fig_y = self.ax.scatter([], [], color='darkcyan', marker='.')
        # conexión del evento para detección de clicks
        self.cid = self.fig.figure.canvas.mpl_connect('button_press_event', self)
        self.dataPlot = []
        self.dataPlot2 = []
        # datos de el barrido
        self.dataPlot3 = []
        self.dataPlot4 = []
        self.class_data = -1 
        # linea que representa la fontera de decisión
        self.__line = line

    # función que nos actualizará el estado del evento de clicks
    def update_state_event(self, state):
        if state:
            self.fig.figure.canvas.mpl_connect('button_press_event', self)
        else:
            self.fig.canvas.mpl_disconnect(self.cid)

    def __call__(self, event):
        # si la figura no contiene un evento
        if event.inaxes!=self.ax.axes: 
            return
        # impresión de las pocisiones marcadas por el mouse
        print(f'Position X: {event.xdata}')
        print(f'Position Y: {event.ydata}')
        # si el contador es par se pone de un color diferente que si no lo es
        match self.class_data:
            case 0:
                self.dataPlot.append((event.xdata, event.ydata))
                self.plot.set_offsets(self.dataPlot)
            case 1:
                self.dataPlot2.append((event.xdata, event.ydata))
                self.another.set_offsets(self.dataPlot2)
        # actualización de la figura
        self.fig.canvas.draw()



    # asignar de nueva cuenta, los datos iniciales (valores de entrenamiento) del gráfico
    def set_data_again(self):
        self.plot = self.ax.scatter([], [], color='red', marker='x')
        self.another = self.ax.scatter([], [], color='blue', marker='o')
        self.__line, = self.ax.plot(0, 0, 'b-')
        self.fig_test = self.ax.scatter([], [], color='black', marker='8')
        self.ax.set_xlim([-5, 5])
        self.ax.set_ylim([-5, 5])
        self.ax.set_title('Perceptron simple')
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # función que nos actualiza los datos que fueron evaluados, para su ubicación en una clase
    def set_new_points(self, x1, x2, class_data):
        match class_data:
            case 0:
                self.dataPlot3.append((x1, x2))
                self.fig_x.set_offsets(self.dataPlot3)
            case 1:
                self.dataPlot4.append((x1, x2))
                self.fig_y.set_offsets(self.dataPlot4)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    

    # función que nos actualiza la línea de la frontera de decisión
    def update_line(self, weight1, weigth2, theta):
        line_points = np.linspace(-5, 5)
        self.__line.set_xdata(line_points)
        # ecuación de la recta tipo y = mx +b 
        weights_data = (-weight1 * line_points + theta) / weigth2
        self.__line.set_ydata(weights_data)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    # función que dibuja en el plano la superficie de decisión
    def draw_desition_superface(self, perceptron):
        n_points = 50
        n_points_y = 12
        feature_x = np.linspace(-5, 5, n_points)
        feature_y = np.linspace(-4.7, 4.7, n_points_y)
        for index in range(0, n_points_y):
            y = feature_y[index]
            for subIndex in range(0, n_points):
                x = feature_x[subIndex]
                class_predicted = perceptron.return_value_of_z_out_of_train(x, y)
                self.set_new_points(x, y, class_predicted)
        

    def change_class(self, class_data):
        self.class_data = class_data
        print(self.class_data)
    

    def get_data_class_one(self):
        return self.dataPlot
    

    def get_data_class_two(self):
        return self.dataPlot2
