from matplotlib import pyplot as plt
import numpy as np

class Render():
    def __init__(self,print_list,plot_labels,line_styles):
        plt.close('all')
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        self.states = []
        self.times = []
        self.print_list = print_list
        self.plot_labels = plot_labels
        self.line_styles = line_styles
        
    def add(self, time, state):
        self.states.append(state)
        self.times.append(time)
        
    def clear(self,):
        self.states = []
        self.times = []
        
    def plot(self):
        plt.cla()
        self.states = np.array(self.states)
        self.times = np.array(self.times)
        for print_state, label, line_style in zip(self.print_list, self.plot_labels, self.line_styles):
            plt.plot(self.times, self.states[:,print_state], label = label, ls=line_style, lw=1)
        plt.legend()
        plt.grid(True)
        plt.draw()
        plt.pause(1)
        self.clear()
        
print_list = [1]
plot_labels = ['mae','pai']
line_styles = ['-','--']

a = Render(print_list,plot_labels,line_styles)

for i in range(100):
    state = np.ones(4)*i
    a.add(i,state)
    
a.plot()