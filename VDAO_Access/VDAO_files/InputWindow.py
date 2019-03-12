import sys
if sys.version_info[0] <= 2: # add tkinker depending on the 
    import Tkinter as tk
    import ttk
else:
    import tkinter as tk
    import tkinter.ttk as ttk

class InputWindow:
        
    def btnOK_Click(self):
        self.Enter_keyDown()

    def btnCancel_Click(self):
        self.eventClose(frameNumber=None)
        self.root.destroy()

    def Enter_keyDown(self, event=None):
        try:
            value = int(self.entryVar.get())
            if value < self.minValue or value > self.maxValue:
                self.lblFrame.config(foreground="red")
            else:
                self.lblFrame.config(foreground="black")
                self.eventClose(value)
                self.root.destroy()
        except ValueError:
            self.lblFrame.config(foreground="red")

    def Escape_keyDown(self,event=None):
        self.root.destroy()

    def __init__(self, parent, eventClose, title="", minValue=0, maxValue=0):
        self.root = parent
        self.eventClose = eventClose
        self.minValue = minValue
        self.maxValue = maxValue
        # Panel with label and entry box
        pnlEntry = tk.PanedWindow(self.root)
        pnlEntry.pack(fill=tk.BOTH, expand=True)
         # Label
        lblInstructions = tk.Label(pnlEntry, text='Enter with the frame number between %d and %d:\n' % (minValue, maxValue))
        lblInstructions.pack(fill=tk.BOTH, expand=True)
        # Label
        self.lblFrame = tk.Label(pnlEntry, text='Frame: ')
        self.lblFrame.pack(side=tk.LEFT)
        # Entry
        self.entryVar = tk.StringVar()
        entry = tk.Entry(pnlEntry, textvariable=self.entryVar)
        entry.pack(side=tk.LEFT, anchor=tk.E, fill=tk.BOTH, expand=True)
        # Panel with buttons
        pnlButtons = tk.PanedWindow(self.root)
        pnlButtons.pack(fill=tk.BOTH, expand=True)
        # Button OK
        btnOk = tk.Button(pnlButtons, text="OK", command=self.btnOK_Click)
        btnOk.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Button Cancel
        btnCancel = tk.Button(pnlButtons, text="Cancel", command=self.btnCancel_Click)
        btnCancel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        # Bind key events
        entry.bind('<Return>', self.Enter_keyDown)
        entry.bind('<Escape>', self.Escape_keyDown)
        # Focus on the entry widget
        entry.focus()

    def Center(self, referenceWindow):
        # Gets the requested values of the height and widht.
        windowWidth = referenceWindow.winfo_reqwidth()
        windowHeight = referenceWindow.winfo_reqheight()
        # Gets both half the screen width/height and window width/height
        positionRight = int(referenceWindow.winfo_screenwidth()/2 - windowWidth/2)
        positionDown = int(referenceWindow.winfo_screenheight()/2 - windowHeight/2)
        # Positions the window in the center of the page.
        self.root.geometry("+{}+{}".format(positionRight, positionDown))
    
# root = tk.Tk()
# inputWindow = InputWindow(parent=root, eventClose=None, title="Entry with a frame number", minValue=0, maxValue=5)
# inputWindow.Center(root)
# root.mainloop()