#!/usr/bin/env python3
import tkinter
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.backends.backend_tkagg as tkagg
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import pathlib

# =======================
# Model & Analysis Functions
# =======================

def MLKF_1dof(m1, l1, k1, f1):
    """Return mass, damping, stiffness & force matrices for a 1DOF system."""
    M = np.array([[m1]])
    L = np.array([[l1]])
    K = np.array([[k1]])
    F = np.array([f1])
    return M, L, K, F

def MLKF_2dof(m1, l1, k1, f1, m2, l2, k2, f2):
    """Return mass, damping, stiffness & force matrices for a 2DOF system."""
    M = np.array([[m1, 0],
                  [0, m2]])
    L = np.array([[l1 + l2, -l2],
                  [-l2,      l2]])
    K = np.array([[k1 + k2, -k2],
                  [-k2,      k2]])
    F = np.array([f1, f2])
    return M, L, K, F

def MLKF_3dof(m1, l1, k1, f1, m2, l2, k2, f2, m3, l3, k3, f3):
    """
    Return mass, damping, stiffness & force matrices for a 3DOF system.
    m1 is the main mass, m2 the 1st absorber (below m₁) and m3 the 2nd absorber (above m₁).
    """
    M = np.array([
        [m1, 0, 0],
        [0, m2, 0],
        [0, 0, m3]
    ])
    L = np.array([
        [l1 + l2 + l3, -l2,       -l3],
        [-l2,           l2,         0],
        [-l3,           0,         l3]
    ])
    K = np.array([
        [k1 + k2 + k3, -k2,       -k3],
        [-k2,           k2,         0],
        [-k3,           0,         k3]
    ])
    F = np.array([f1, f2, f3])
    return M, L, K, F

def freq_response(w_list, M, L, K, F):
    """Return the complex frequency response of the system."""
    responses = []
    for w in w_list:
        A = -w*w * M + 1j*w * L + K
        responses.append(np.linalg.solve(A, F))
    return np.array(responses)

def time_response(t_list, M, L, K, F):
    """Return the time response of the system to a step input."""
    mm = M.diagonal()
    def slope(t, y):
        n = len(mm)
        x = y[:n]
        v = y[n:]
        a = (F - L.dot(v) - K.dot(x)) / mm
        return np.concatenate((v, a))
    sol = scipy.integrate.solve_ivp(
        fun=slope,
        t_span=(t_list[0], t_list[-1]),
        y0=np.zeros(2*len(mm)),
        t_eval=t_list,
        method='Radau'
    )
    return sol.y[:len(mm), :].T

def last_nonzero(arr, axis, invalid_val=-1):
    """Return index of last nonzero element of an array along the specified axis."""
    mask = (arr != 0)
    reversed_mask = np.flip(mask, axis=axis)
    idx = arr.shape[axis] - reversed_mask.argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), idx, invalid_val)

def plot_response(fig, hz, sec, M, L, K, F, show_phase=None):
    """Plot frequency and time domain responses."""
    # Frequency-domain response
    f_response = freq_response(hz * 2 * np.pi, M, L, K, F)
    f_amplitude = np.abs(f_response)
    # Time-domain response
    t_response = time_response(sec, M, L, K, F)
    
    # Legends for frequency plot
    f_legends = []
    for i in range(f_amplitude.shape[1]):
        idx = np.argmax(f_amplitude[:, i])
        f_legends.append(f"m{i+1} peak {f_amplitude[idx, i]:.4g} m at {hz[idx]:.4g} Hz")
    
    # Legends for time plot (settling time within 2%)
    equilib = np.abs(freq_response([0], M, L, K, F))[0]
    toobig = np.abs(100 * (t_response - equilib) / equilib) >= 2
    lastbig = last_nonzero(toobig, axis=0, invalid_val=len(sec)-1)
    t_legends = []
    for i in range(t_response.shape[1]):
        t_legends.append(f"m{i+1} settled beyond {sec[lastbig[i]]:.4g} sec")
    
    fig.clear()
    if show_phase is not None:
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
        ax3 = fig.add_subplot(3, 1, 3)
    else:
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
    
    ax1.set_title('Frequency Domain Response (Amplitude)')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Amplitude (m)')
    lines = ax1.plot(hz, f_amplitude)
    ax1.legend(lines, f_legends)
    
    if show_phase is not None:
        if show_phase == 0:
            ax2.set_title('Frequency Domain Response (Phase)')
            phase_response = np.angle(f_response, deg=True)
        else:
            ref_index = show_phase - 1
            ax2.set_title(f'Phase Relative to m{show_phase}')
            phase_response = np.angle(f_response / f_response[:, [ref_index]], deg=True)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Phase (°)')
        lines_phase = ax2.plot(hz, phase_response)
        ax2.legend(lines_phase, [f"m{i+1}" for i in range(f_response.shape[1])])
    
    ax_final = ax3 if show_phase is not None else ax2
    ax_final.set_title('Time Domain Response (Step Force)')
    ax_final.set_xlabel('Time (sec)')
    ax_final.set_ylabel('Displacement (m)')
    lines_time = ax_final.plot(sec, t_response)
    ax_final.legend(lines_time, t_legends)
    
    fig.tight_layout()

# =======================
# GUI and Helper Classes
# =======================

class FloatEntry(tkinter.Entry):
    """A specialized Entry widget that only accepts floats."""
    def __init__(self, master, width, on_changed=lambda: None):
        self._on_changed = on_changed
        self._text = tkinter.StringVar()
        self._text.trace_add("write", self._on_text_change)
        super().__init__(master, width=width, textvariable=self._text)
        self._validate()
    
    def _on_text_change(self, *args):
        self._validate()
        self._on_changed()
    
    def _validate(self):
        try:
            self._value = float(self.get())
            self.config({"background": "White"})
        except ValueError:
            self._value = None
            self.config({"background": "Pink"})
    
    @property
    def value(self):
        return self._value

class Params:
    """A collection of label/entry pairs for input parameters."""
    def __init__(self, on_changed=lambda: None):
        self._labels1 = {}
        self._entries = {}
        self._labels2 = {}
        self._on_changed = on_changed
    
    def create_param(self, master, label, key, row, column):
        text1, text2 = label.split(':')
        label1 = tkinter.Label(master, text=text1)
        entry = FloatEntry(master, width=10, on_changed=self._on_changed)
        label2 = tkinter.Label(master, text=text2)
        label1.grid(row=row, column=column, padx=5, pady=3, sticky='se')
        entry.grid(row=row, column=column+1, padx=5, pady=3, sticky='sew')
        label2.grid(row=row, column=column+2, padx=5, pady=3, sticky='sw')
        self._labels1[key] = label1
        self._entries[key] = entry
        self._labels2[key] = label2
    
    def state(self, keys):
        key_list = keys.split(',')
        return tkinter.NORMAL if all(self._entries[k].value is not None for k in key_list) else tkinter.DISABLED
    
    def set_state(self, key, state):
        self._labels1[key]['state'] = state
        self._entries[key]['state'] = state
        self._labels2[key]['state'] = state
    
    def __getitem__(self, key):
        return self._entries[key].value

class GUI:
    """Graphical User Interface for the Vibration Absorber system."""
    def __init__(self):
        self.params = Params(on_changed=self.enable_disable)
        self.root = tkinter.Tk()
        self.root.title("Vibration Absorber - 3DOF System")
        
        # Create the matplotlib figure and canvas with increased rowspan.
        self.fig = plt.Figure(figsize=(5, 4), dpi=100)
        self.canvas = tkagg.FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=1, column=4, rowspan=102, padx=5, pady=3, sticky='nsew')
        self.canvas.mpl_connect("resize_event", lambda event: self.fig.tight_layout(pad=2.5))
        
        # Toolbar frame
        toolbar_frame = tkinter.Frame(self.root)
        toolbar_frame.grid(row=0, column=4, sticky='nsew')
        toolbar = tkagg.NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        
        # Parameter groups for main mass and absorbers
        self.keys_1dof = "m1,k1,l1,f1"
        self.labels_1dof = "m₁ =:kg,k₁ =:N/m,λ₁ =:Ns/m,f₁ =:N"
        self.keys_abs1 = "m2,k2,l2"
        self.labels_abs1 = "m₂ =:kg,k₂ =:N/m,λ₂ =:Ns/m"
        self.keys_abs2 = "m3,k3,l3"
        self.labels_abs2 = "m₃ =:kg,k₃ =:N/m,λ₃ =:Ns/m"
        
        # Create parameters for main mass and absorbers on the left
        self.create_params(self.labels_1dof, self.keys_1dof, first_row=1)
        
        # Checkbutton and parameters for 1st absorber (below m₁)
        self.absorber1_enabled = tkinter.IntVar()
        self.check_abs1 = tkinter.Checkbutton(
            self.root, text="Add 1st Absorber (below)",
            variable=self.absorber1_enabled, command=self.enable_disable)
        self.check_abs1.grid(row=50, column=0, columnspan=3, padx=5, pady=3, sticky='nsw')
        self.create_params(self.labels_abs1, self.keys_abs1, first_row=51)
        
        # Checkbutton and parameters for 2nd absorber (above m₁)
        self.absorber2_enabled = tkinter.IntVar()
        self.check_abs2 = tkinter.Checkbutton(
            self.root, text="Add 2nd Absorber (above)",
            variable=self.absorber2_enabled, command=self.enable_disable)
        self.check_abs2.grid(row=60, column=0, columnspan=3, padx=5, pady=3, sticky='nsw')
        self.create_params(self.labels_abs2, self.keys_abs2, first_row=61)
        
        # Checkbutton for showing phase response
        self.enabled_phase = tkinter.IntVar()
        check_phase = tkinter.Checkbutton(
            self.root, text="Show phase", variable=self.enabled_phase)
        check_phase.grid(row=99, column=0, columnspan=3, padx=5, pady=3, sticky='nsw')
        
        # Plot button
        self.button_plot = tkinter.Button(self.root, text="Plot", command=self.plot)
        self.button_plot.grid(row=100, column=0, columnspan=3, padx=5, pady=3, sticky='nsew')
        
        # Diagram image label (only one label is used).
        prog_dir = pathlib.Path(__file__).parent
        self.diagram_1dof = tkinter.PhotoImage(file=prog_dir / "a1_diagram_1dof.png")
        self.diagram_2dof = tkinter.PhotoImage(file=prog_dir / "a1_diagram_2dof.png")   
        self.diagram_3dof = tkinter.PhotoImage(file=prog_dir / "a1_diagram_3dof.png")
        self.label_diagram = tkinter.Label(self.root, image=self.diagram_1dof)
        self.label_diagram.grid(row=101, column=0, columnspan=3, padx=5, pady=3, sticky='nsew')
        
        # Quit button
        button_quit = tkinter.Button(self.root, text="Quit", command=self.root.destroy)
        button_quit.grid(row=104, column=0, columnspan=3, padx=5, pady=3, sticky='nsew')
        
        # Configure grid weights for proper expansion
        self.root.rowconfigure(101, weight=1)
        self.root.columnconfigure(4, weight=1)
        
        self.enable_disable()
        self.root.mainloop()
    
    def create_params(self, labels, keys, first_row):
        for row, (label, key) in enumerate(zip(labels.split(','), keys.split(',')), start=first_row):
            self.params.create_param(self.root, label, key, row, 0)
    
    def enable_disable(self, *args):
        # Enforce: if absorber2 is checked, absorber1 must be checked.
        if self.absorber2_enabled.get() and not self.absorber1_enabled.get():
            self.absorber1_enabled.set(1)
        
        # Disable absorber2 checkbox if absorber1 is not enabled.
        if not self.absorber1_enabled.get():
            self.absorber2_enabled.set(0)
            self.check_abs2.config(state="disabled")
        else:
            self.check_abs2.config(state="normal")
        
        # Determine system mode and update diagram.
        a1 = bool(self.absorber1_enabled.get())
        a2 = bool(self.absorber2_enabled.get())
        
        if not a1 and not a2:
            state = self.params.state(self.keys_1dof)
            self.label_diagram.configure(image=self.diagram_1dof)
        elif a1 and not a2:
            state = self.params.state(f"{self.keys_1dof},{self.keys_abs1}")
            self.label_diagram.configure(image=self.diagram_2dof)
        else:
            # When absorber2 is ticked (with or without absorber1), show the 3DOF diagram.
            if a1:
                state = self.params.state(f"{self.keys_1dof},{self.keys_abs1},{self.keys_abs2}")
            else:
                state = self.params.state(f"{self.keys_1dof},{self.keys_abs2}")
            self.label_diagram.configure(image=self.diagram_3dof)
        
        self.button_plot['state'] = state
        for key in self.keys_abs1.split(','):
            self.params.set_state(key, tkinter.NORMAL if a1 else tkinter.DISABLED)
        for key in self.keys_abs2.split(','):
            self.params.set_state(key, tkinter.NORMAL if a2 else tkinter.DISABLED)
    
    def plot(self):
        a1 = bool(self.absorber1_enabled.get())
        a2 = bool(self.absorber2_enabled.get())
        
        if not a1 and not a2:
            args_keys = self.keys_1dof
            MKLF = MLKF_1dof
            kwargs = {}
        elif a1 and not a2:
            args_keys = f"{self.keys_1dof},{self.keys_abs1}"
            MKLF = MLKF_2dof
            kwargs = {"f2": 0}
        else:
            if a1:
                args_keys = f"{self.keys_1dof},{self.keys_abs1},{self.keys_abs2}"
                MKLF = MLKF_3dof
                kwargs = {"f2": 0, "f3": 0}
            else:
                args_keys = f"{self.keys_1dof},{self.keys_abs2}"
                MKLF = MLKF_2dof
                kwargs = {"f2": 0}
        
        kwargs.update({arg: self.params[arg] for arg in args_keys.split(',')})
        M, L, K, F = MKLF(**kwargs)
        
        hz = np.linspace(0, 5, 10001)
        sec = np.linspace(0, 30, 10001)
        if self.enabled_phase.get() == 0:
            phase = None
        elif (not a1 and not a2) or (a1 ^ a2):
            phase = 0
        else:
            phase = 1
        
        plot_response(self.fig, hz, sec, M, L, K, F, phase)
        self.canvas.draw()

if __name__ == "__main__":
    GUI()
