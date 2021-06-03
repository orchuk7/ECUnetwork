import tkinter as tk
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
import tensorflow as tf
import numpy as np
import statistics

model = tf.keras.models.load_model('model')

window = tk.Tk()
frame = tk.Frame(master=window, width=600, height=300, 	bg="black")
frame.pack()

def clicked():
    data = np.array([[float(txt_1.get()),
                      float(txt_2.get()),
                      float(txt_3.get()),
                      float(txt_4.get()),
                      float(txt_5.get()),
                      float(txt_6.get()),
                      float(txt_7.get()),
                      float(txt_8.get())]])
    mean = statistics.mean(data[0])
    std = statistics.stdev(data[0])
    data -= mean
    data /= std
    pred = model.predict(data)
    res = "Fuel Inj Amt: {}".format(pred[0][0] * 10)  
    label_9.configure(text=res) 


label_main = tk.Label(
    text="Enter all values",
    fg="white",
    bg="black",
    font=('Arial', 18)
)

label_main.place(x=210, y=0)

label_1 = tk.Label(
    text="Accel Pedal",
    fg="white",
    bg="black",
    font=('Arial', 14)
)
label_1.place(x=5, y=40)

label_2 = tk.Label(
    text="Coolant Temp",
    fg="white",
    bg="black",
    font=('Arial', 14)
)
label_2.place(x=5+140, y=40)

label_3 = tk.Label(
    text="Engine Lambda",
    fg="white",
    bg="black",
    font=('Arial', 14)
)
label_3.place(x=5+140*2, y=40)

label_4 = tk.Label(
    text="Engine Speed",
    fg="white",
    bg="black",
    font=('Arial', 14)
)
label_4.place(x=5+146*3, y=40)

label_5 = tk.Label(
    text="Man Air Pr",
    fg="white",
    bg="black",
    font=('Arial', 14)
)
label_5.place(x=5, y=130)

label_6 = tk.Label(
    text="Mass Air Flow",
    fg="white",
    bg="black",
    font=('Arial', 14)
)
label_6.place(x=5+140, y=130)

label_7 = tk.Label(
    text="Rel Thrott Pos",
    fg="white",
    bg="black",
    font=('Arial', 14)
)
label_7.place(x=5+143*2, y=130)

label_8 = tk.Label(
    text="Spark Advance",
    fg="white",
    bg="black",
    font=('Arial', 14)
)
label_8.place(x=5+146*3, y=130)

label_9 = tk.Label(
    text="Fuel Inj Amt: ",
    fg="white",
    bg="black",
    font=('Arial', 14)
)
label_9.place(x=150, y=230)


txt_1 = tk.Entry(window,width=10)
txt_1.place(x=10, y=70)

txt_2 = tk.Entry(window,width=10)
txt_2.place(x=160, y=70)

txt_3 = tk.Entry(window,width=10)
txt_3.place(x=10+150*2, y=70)

txt_4 = tk.Entry(window,width=10)
txt_4.place(x=10+150*3, y=70)

txt_5 = tk.Entry(window,width=10)
txt_5.place(x=10, y=160)

txt_6 = tk.Entry(window,width=10)
txt_6.place(x=160, y=160)

txt_7 = tk.Entry(window,width=10)
txt_7.place(x=10+150*2, y=160)

txt_8 = tk.Entry(window,width=10)
txt_8.place(x=10+150*3, y=160)

btn = tk.Button(window, text="Calculate", command=clicked)
btn.place(x=10, y=230)

window.mainloop()
