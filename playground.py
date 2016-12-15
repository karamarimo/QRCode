# from tkinter import *

# root = Tk()
# myImage = PhotoImage(file='./qr.png')
# label = Label(image=myImage)
# label.pack()

# root.mainloop()

def f(x):
    return x ** 2

a = f(100)

print(a)

class Vector(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # def __add__(self, other):
    #   if type(other) = Vector:
    #       