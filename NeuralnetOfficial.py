import random
import math

RATE = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
num_input = 49
num_hidden = 30
num_output = 3
iterations = 1000


class Matrix:
    def get(self, r, c):
        return self.matrix[r*self.cols + c]
    def setelem(self, r, c, x):
        self.matrix[r*self.cols + c] = x
    def __init__(self, rows, cols, x = None):
        self.matrix = []
        self.rows = rows
        self.cols = cols
        if(x == None):
            for i in range(0, rows*cols):
                self.matrix.append(R())
        else:
            if(x.__class__.__name__ == 'list'):
                if(len(x) != self.rows * self.cols):
                    raise ValueError("Different value in rows and cols")
                self.matrix = x
            else:
                for i in range(0, rows*cols):
                    self.matrix.append(x)
    def __str__(self):
        s = ""
        for i in range(0, self.rows):
            for j in range(0, self.cols):
                s += str(self.get(i, j)) + " "
            s += "\n"
        return s
    def __add__(self, m):
        if(self.rows != m.rows or self.cols != m.cols):
            raise ValueError("Can't do sum of different matrix")
        new = Matrix(self.rows, self.cols)
        for i in range(0, self.rows*self.cols):
            new.matrix[i] = self.matrix[i] + m.matrix[i]
        return new
    def __sub__(self, m):
        if(self.rows != m.rows or self.cols != m.cols):
            raise ValueError("Can't do diff of different matrix")
        new = Matrix(self.rows, self.cols)
        for i in range(0, self.rows*self.cols):
            new.matrix[i] = self.matrix[i] - m.matrix[i]
        return new
    def __mul__(self, m):
        if(self.rows != m.rows or self.cols != m.cols):
            raise ValueError("Can't do mul of different matrix")
        new = Matrix(self.rows, self.cols)
        for i in range(0, self.rows*self.cols):
            new.matrix[i] = self.matrix[i] * m.matrix[i]
        return new
    def __neg__(self):
        new = Matrix(self.rows, self.cols)
        for i in range(0, self.rows*self.cols):
            new.matrix[i] = -self.matrix[i]
        return new
    def __matmul__(self, m):
        if(self.cols != m.rows):
            raise ValueError("Can't do matmul of different matrix")
        new = Matrix(self.rows, m.cols)
        for i in range(0, self.rows):
            for j in range(0, m.cols):
                somma = 0
                for k in range(0, self.cols):
                    somma += self.get(i, k)*m.get(k, j)
                new.setelem(i, j, somma)
        return new
    def __invert__(self): # ~ trasposta
        new = Matrix(self.cols, self.rows)
        for i in range(0, self.rows):
            for j in range(0, self.cols):
                new.setelem(j, i, self.get(i, j))
        return new
    def __pow__(self, c):
        new = Matrix(self.rows, self.cols)
        for i in range(0, self.rows):
            for j in range(0, self.cols):
                new.setelem(i, j, c*self.get(i, j))
        return new
    def activation_funtion(self):
        new = Matrix(self.rows, self.cols)
        for i in range(0, self.rows):
            for j in range(0, self.cols):
                value = activation_funtion(self.get(i, j))
                new.setelem(i, j, value)
        return new
    def get_row(self, i):
        ret = Matrix(1, self.cols)
        for j in range(0, self.cols):
            ret.setelem(0, j, self.get(i, j))
        return ret
    def get_col(self, j):
        ret = Matrix(self.rows, 1)
        for i in range(0, self.rows):
            ret.setelem(i, 0, self.get(i, j))
        return ret

def build_matrix(s):
    l = s.split("\n")
    rows = len(l)
    cols = 0
    new = []
    for i in l:
        l2 = i.strip().split(" ")
        cols = len(l2)
        for j in l2:
            new.append(float(j))
    return Matrix(rows, cols, new)


def activation_funtion(x):
    return 1/(1 + math.exp(-x))
    # return x

class Level:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
    def __str__(self):
        return str(self.weight) + "NEW\n" + str(self.bias)
    def output(self, input):
        ret = (self.weight @ input) + self.bias
        return ret.activation_funtion()
    def update(self, d, input, rate):
        dw = (d @ (~input)) ** rate
        self.weight = self.weight + dw
        db = d ** rate
        self.bias = self.bias + db
    def learning_out(self, target, input, output, rate):
        uni = Matrix(output.rows, 1, 1)
        d = output * ((uni - output) * (target - output))
        self.update(d, input, rate)
        return d
    def learning_hidden(self, weight, do, input, output, rate):
        uni = Matrix(output.rows, 1, 1)
        dh = output * ((uni - output) * ((~weight) @ do))
        self.update(dh, input, rate)
        return dh

def print_list(l):
    for i in l:
        if(i.__class__.__name__ == 'Level'):
            print(i.weight)
        else:
            print(i)

def reverse(l):
    new = []
    for i in range(len(l)-1, -1, -1):
        new.append(l[i])
    return new

class Network:
    def __init__(self, input=0, hidden=[], output=0):
        self.levels = []
        hidden.append(output)
        actual_input = input
        for h in hidden:
            m = Matrix(h, actual_input)
            b = Matrix(h, 1)
            l = Level(m, b)
            self.levels.append(l)
            actual_input = h
    def __str__(self):
        s = ""
        for l in self.levels:
            s += str(l) + "NEW\n"
        return s.strip("NEW\n")
    def set(self, levels):
        self.levels = levels
        return self
    def back_propagation(self, input, target, rate):
        levels = reverse(self.levels)
        outputs = self.feed_forward(input)
        outputs = reverse(outputs)
        actual_d = levels[0].learning_out(target, outputs[1], outputs[0], rate)
        for i in range(1, len(levels)):
            actual_d = levels[i].learning_hidden(levels[i - 1].weight, actual_d, outputs[i + 1], outputs[i], rate)
    def feed_forward(self, input):
        actual_input = input
        list_output = [input]
        for l in self.levels:
            actual_input = l.output(actual_input)
            list_output.append(actual_input)
        return list_output
    def learning(self, inputs, targets, iterations, rates):
        if(len(inputs) != len(targets)):
            raise ValueError("Number of inputs and targets are different")
        n = int(iterations/len(rates))
        for rate in rates:
            for k in range(0, n):
                for i in range(0, len(inputs)):
                    self.back_propagation(inputs[i], targets[i], rate)
    def save(self, name):
        file = open(name, "w")
        file.write(str(self))
        file.close()

def load_net(name):
    file = open(name, "r")
    text = file.read()
    file.close()
    l = text.split("NEW")
    mh = build_matrix(l[0].strip())
    bh = build_matrix(l[1].strip())
    mo = build_matrix(l[2].strip())
    bo = build_matrix(l[3].strip())
    l1 = Level(mh, bh)
    l2 = Level(mo, bo)
    return Network().set([l1, l2])

def test(l, n):
    for path in l:
        i = take_input(path)
        print(n.feed_forward(i)[-1])

def R():
    return random.uniform(0.0, 0.1)
    # return random.randint(0, 9)

def reduce(m):
    l = []
    for i in range(0, m.rows, int(m.rows/7)):
        for j in range(0, m.cols, int(m.cols/7)):
            somma = 0
            for k in range(0, 4):
                for h in range(0, 4):
                    somma += m.get(i + k, j + h)
            l.append(somma/16)
    return Matrix(49, 1, l)

def take_input(nome_file):
    file = open(nome_file, "rb")
    byte = file.read()
    file.close()
    l = []
    d = {0.0 : 1.0, 1.0 : 0.0}
    for i in byte[11:]:
        l.append(d[float(i)])
    m = Matrix(28, 28, l)
    return reduce(m)

def create_inputs():
    ret = []
    ret.append(take_input("./0/Immagine1.pgm"))
    ret.append(take_input("./0/Immagine2.pgm"))
    ret.append(take_input("./0/Immagine3.pgm"))
    ret.append(take_input("./0/Immagine4.pgm"))
    ret.append(take_input("./1/Immagine1.pgm"))
    ret.append(take_input("./1/Immagine2.pgm"))
    ret.append(take_input("./1/Immagine3.pgm"))
    ret.append(take_input("./1/Immagine4.pgm"))
    ret.append(take_input("./2/Immagine1.pgm"))
    ret.append(take_input("./2/Immagine2.pgm"))
    ret.append(take_input("./2/Immagine3.pgm"))
    ret.append(take_input("./2/Immagine4.pgm"))
    ret.append(take_input("./7/Immagine1.pgm"))
    ret.append(take_input("./7/Immagine2.pgm"))
    return ret
def create_targets(output):
    ret = []
    ret.append( Matrix(output, 1, [1, 0, 0]) )
    ret.append( Matrix(output, 1, [1, 0, 0]) )
    ret.append( Matrix(output, 1, [1, 0, 0]) )
    ret.append( Matrix(output, 1, [1, 0, 0]) )
    ret.append( Matrix(output, 1, [0, 1, 0]) )
    ret.append( Matrix(output, 1, [0, 1, 0]) )
    ret.append( Matrix(output, 1, [0, 1, 0]) )
    ret.append( Matrix(output, 1, [0, 1, 0]) )
    ret.append( Matrix(output, 1, [0, 0, 1]) )
    ret.append( Matrix(output, 1, [0, 0, 1]) )
    ret.append( Matrix(output, 1, [0, 0, 1]) )
    ret.append( Matrix(output, 1, [0, 0, 1]) )
    ret.append( Matrix(output, 1, [0, 0, 0]) )
    ret.append( Matrix(output, 1, [0, 0, 0]) )
    return ret


# LEARNING SCRIPT
# n = Network(num_input, [num_hidden], num_output)
# n.learning(create_inputs(), create_targets(num_output), iterations, RATE)
# n.save("new_net.txt")
# test(["./0/Immagine5.pgm", "./1/Immagine5.pgm", "./2/Immagine5.pgm", "./7/Immagine2.pgm"], n)

# ONLY OUTPUT SCRIPT
n2 = load_net("net.txt")
test(["./0/Immagine5.pgm", "./1/Immagine5.pgm", "./2/Immagine5.pgm", "./7/Immagine2.pgm"], n2)
