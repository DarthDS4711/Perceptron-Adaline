from tkinter import Entry, Tk, Frame, Button, Label, messagebox, Text, Radiobutton
import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from classes.Perceptron import Perceptron
from classes.PointBuilder import PointBuilder
from validators.validator import *


class Window:
    def __init__(self):
        self.__window = Tk()
        self.__window.geometry('850x620')
        self.__window.wm_title('Perceptron Simple')
        self.__window.minsize(width=850, height=620)
        # evitar el cambio de tamaño por el usuario
        self.__window.resizable(False, False)
        self.__frame = Frame(self.__window,  bg='gray22', bd=3)
        self.__frame.grid(row=0, column=0, columnspan=4)
        fig, ax = plt.subplots()
        plot = ax.scatter([], [], color='red', marker='o')
        another = ax.scatter([], [], color='blue', marker='o')
        fig_test = ax.scatter([], [], color='black', marker='8')
        line, = ax.plot(0, 0, 'b-')
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_title('Perceptron simple')
        self.__pointsBuilder = PointBuilder(fig, ax, plot, another, line, fig_test)
        self.__perceptron = Perceptron()
        # agregar el gráfico a la ventana
        self.__canvas = FigureCanvasTkAgg(fig, master=self.__frame)
        self.__canvas.get_tk_widget().grid(row=0, column=0, columnspan=4)

        # botones principales del programa
        self.__btn1 = Button(self.__window, text='Train', width=15, bg='green', fg='white',
                             command=self.train)
        self.__btn2 = Radiobutton(self.__window, text='Red rose', command=self.class_flower_rose, value=1)

        self.__btn3 = Radiobutton(self.__window, text='Blue Rose',command=self.class_flower_white, value=2)

        self.__btn4 = Button(self.__window, padx=5, pady=5, text='Inicialize weigths',
                             command=self.inicialize_random, fg='white', bg='orange')
        self.__btn5 = Button(self.__window, width=15, text='Test', command=self.evaluate_points, 
                            state=tkinter.DISABLED, bg='blue', fg='white')

        self.__btn6 = Button(self.__window, text='Quit', bg='red', fg='white',
                             command=self.quit, width=15)
        # botones relacionados a los entrys
        self.__btn_epochs = Button(self.__window, text='Asign',
                                   command=self.get_epochs, width=15)
        self.__btn_learning_rate = Button(self.__window, text='Asign',
                                          command=self.get_learning_rate, width=15)

        # inicialización de entrys
        self.__entry_epochs = Entry(self.__window)
        self.__entry_learning_rate = Entry(self.__window)
        self.__text_theta = Text(
            self.__window, height=1, width=24, state=tkinter.DISABLED)
        self.__text_epochs = Text(
            self.__window, height=1, width=24, state=tkinter.DISABLED)
        self.__text_train = Text(
            self.__window, height=1, width=24, state=tkinter.DISABLED)

        # inicialización de labels (relacionados a los entrys)
        Label(self.__window, text='N° Epochs: ').grid(row=3, column=0)
        Label(self.__window, text='Learning Rate: ').grid(row=4, column=0)
        # inicialización de label relacionado a entrys de información
        Label(self.__window, text='Final Value theta: ').grid(row=3, column=3)
        Label(self.__window, text='Final number epochs: ').grid(row=4, column=3)
        Label(self.__window, text='Train: ').grid(row=5, column=3)

        # sección para inicializar los botones
        self.set_buttons()
        self.set_entrys()

        self.__window.mainloop()

    def set_entrys(self):
        # Entrys relacionados a ingresar información al programa
        self.__entry_epochs.grid(row=3, column=1)
        self.__entry_learning_rate.grid(row=4, column=1)
        # entrys relacionados con desplegar información
        self.__text_theta.grid(row=3, column=4)
        self.__text_epochs.grid(row=4, column=4)
        self.__text_train.grid(row=5, column=4)

    def set_buttons(self):
        # Botones principales del programa
        self.__btn1.grid(row=1, column=0)
        self.__btn2.grid(row=1, column=1)
        self.__btn3.grid(row=1, column=2)
        self.__btn4.grid(row=1, column=3)
        self.__btn5.grid(row=1, column=4)
        self.__btn6.grid(row=0, column=4)
        # botones relacionados a los entrys
        self.__btn_epochs.grid(row=3, column=2)
        self.__btn_learning_rate .grid(row=4, column=2)

    
    # boton para finalizar la execución de la aplicación
    def quit(self):
        self.__window.quit()

    # función que actualiza el estado de las cajas de texto
    def update_text_boxes(self, state):
        if state:
            self.__text_theta['state'] = tkinter.NORMAL
            self.__text_epochs['state'] = tkinter.NORMAL
            self.__text_train['state'] = tkinter.NORMAL
        else:
            self.__text_theta['state'] = tkinter.DISABLED
            self.__text_epochs['state'] = tkinter.DISABLED
            self.__text_train['state'] = tkinter.DISABLED

    # función que actualiza el estado de los botones relacionados a los inputs
    def update_buttons_entrys(self, state):
        if state:
            self.__btn_epochs['state'] = tkinter.DISABLED
            self.__btn_learning_rate['state'] = tkinter.DISABLED
        else:
            self.__btn_epochs['state'] = tkinter.NORMAL
            self.__btn_learning_rate['state'] = tkinter.NORMAL


   # función para mostrar la información por pantalla
    def show_info(self):
       self.update_text_boxes(True)
       self.__text_theta.insert('1.0', str(self.__perceptron.get_theta()))
       self.__text_epochs.insert('1.0', str(self.__perceptron.get_number_of_epochs()))
       if self.__perceptron.get_status_perceptron():
            self.__text_train.insert('1.0', "OK")
       else:
           self.__text_train.insert('1.0', "Error")
       self.update_text_boxes(False)

    # función que nos valida si existe información previa para entrenar
    def __validate_data_to_train(self):
        number_data_class_one = len(self.__pointsBuilder.get_data_class_one())
        number_data_class_two = len(self.__pointsBuilder.get_data_class_two())
        if number_data_class_one == 0 or number_data_class_two == 0:
            return False
        else:
            return True


    # función que nos valida que ya se haya registrado el factor de aprendizaje y el número de epocas
    def __validate_data_epochs_and_flearning(self):
        epochs = self.__perceptron.get_epochs()
        f_learning = self.__perceptron.get_factor_learning()
        if epochs == 0 or f_learning == 0:
            return False
        else:
            return True

    # metodo que comienza a entrenar con los datos actuales
    def train(self):
        if self.__validate_data_to_train():
            if self.__validate_data_epochs_and_flearning():
                self.block_main_buttons()
                self.__pointsBuilder.update_state_event(False)
                self.update_buttons_entrys(True)
                self.__perceptron.set_inputs_outpus(self.__pointsBuilder.get_data_class_one(),
                    self.__pointsBuilder.get_data_class_two())
            
                self.__perceptron.train_adaline(self.__pointsBuilder)
                self.show_info()
                self.__pointsBuilder.change_class(-1)
                self.__btn5['state'] = tkinter.NORMAL
                self.__pointsBuilder.update_state_event(True)
            else:
                 messagebox.showinfo(
                    message="Factor de aprendizaje o n épocas no asignado", title="Error")     
        else:
            messagebox.showinfo(
                message="No existen datos en una o las dos clases", title="Error")

    # Cambio en el tipo de flor a mapear
    def class_flower_rose(self):
        self.__pointsBuilder.change_class(0)

    def class_flower_white(self):
        self.__pointsBuilder.change_class(1)

    # inicialización de manera aleatoria de los pesos del perceptron
    def inicialize_random(self):
        self.__perceptron.inicialize_weigths()
        weitgh1 = self.__perceptron.get_weigth1()
        weitgh2 = self.__perceptron.get_weigth2()
        weigth0 = self.__perceptron.get_theta()
        self.__pointsBuilder.update_line(weitgh1, weitgh2, weigth0)

    # evaluación de los puntos obtenidos posteriores al entrenamiento
    def evaluate_points(self):
        self.__pointsBuilder.draw_desition_superface(self.__perceptron)

    # función para obtener el learning rate
    def get_learning_rate(self):
        learning_rate = self.__entry_learning_rate.get()
        self.__entry_learning_rate.delete("0", "end")
        if not validate_float(learning_rate):
            messagebox.showinfo(
                message="Factor de aprendizaje incorrecto", title="Error")
        else:
            learning_rate = float(learning_rate)
            self.__perceptron.set_factor_learning(learning_rate)
            messagebox.showinfo(
                message="Factor de aprendizaje agregado correctamente", title="Éxito")

    # función relacionada a la obtención del número de epocas
    def get_epochs(self):
        n_epochs = self.__entry_epochs.get()
        self.__entry_epochs.delete("0", "end")
        if not validate_integer(n_epochs):
            messagebox.showinfo(
                message="Número de epocas incorrecto", title="Error")
        else:
            n_epochs = int(n_epochs)
            self.__perceptron.set_epochs(n_epochs)
            messagebox.showinfo(
                message="Número de epocas agregado correctamente", title="Éxito")

    def block_main_buttons(self):
        self.__btn1['state'] = tkinter.DISABLED
        self.__btn2['state'] = tkinter.DISABLED
        self.__btn3['state'] = tkinter.DISABLED
        self.__btn4['state'] = tkinter.DISABLED
