import sys
if sys.version_info[0] < 2: # add tkinker depending on the 
    import Tkinter as tk
else:
    import tkinter as tk

class CheckBar:
    
    def __init__(self, parent=None, picks=[], maxCol=5, commands=None, root=None, initialState='normal'):
        self.DescriptionElementAll = None
        self.CommandElementAll = None
        self.vars = []
        self.checkbuttons = []
        self.parent = parent
        self.maxCol = maxCol
        self.allPicks = picks
        if commands == None:
            self.commands = [None for i in picks]
        else:
            self.commands = commands
        self.maxWidth = len(CheckBar.getMaxWord(self.allPicks))
        self.AddElements()
        self.initialState = initialState
        if self.initialState == 'disable':
            self.DisableAllElements()

    def Add(self, picks=[], anchor=tk.W, commands=None):
        # r = int(len(self.allPicks)/self.maxCol)
        # c = len(self.allPicks)%self.maxCol
        self.RemoveAllElements()
        self.allPicks = self.allPicks + picks
        if commands != None:
            self.commands = self.commands + commands
        self.maxWidth = len(CheckBar.getMaxWord(self.allPicks))
        self.AddElements()
    
    def AddElements(self):
        r = c = 0
        var = tk.IntVar()
        for idx in range(len(self.allPicks)):
            var = tk.IntVar()
            chk = tk.Checkbutton(self.parent, text=' %s'%self.allPicks[idx], variable=var, onvalue=1, 
                                offvalue=0, pady=2, width=self.maxWidth+2, anchor=tk.W,
                                command=self.commands[idx])
            self.checkbuttons.append(chk)
            chk.grid(row=r,column=c)
            c = c + 1
            if c >= self.maxCol:
                c = 0
                r = r + 1
            self.vars.append(var)

    def AddElementAll(self, description, command=None):
        self.RemoveAllElements(removeCommands=False) # dont remove all commands. they'll be used
        self.DescriptionElementAll = description
        self.CommandElementAll = command
        self.commands = [self.AllIsSelected] + [c for c in self.commands]
        self.allPicks = [description]+ self.allPicks
        self.maxWidth = len(CheckBar.getMaxWord(self.allPicks))
        self.AddElements()
        if self.initialState == 'disable': #Apply initial state
            self.DisableAllElements()

    def RemoveAllElements(self, removeCommands = False):
        for widget in self.parent.winfo_children():
            widget.destroy()
        self.vars = []
        self.checkbuttons = []
        self.DescriptionElementAll = None
        if removeCommands:
            self.commands = []

    def DisableAllElements(self):
        for widget in self.parent.winfo_children():
            widget.configure(state='disable')
    
    def EnableAllElements(self):
        for widget in self.parent.winfo_children():
            widget.configure(state='normal')

    def GetStates(self):
        myDict = {}
        for idx in range(len(self.vars)):
            myDict[self.allPicks[idx]] = self.vars[idx].get()
        # return map((lambda var: var.get()), self.vars)
        return myDict

    def GetOnlySelected(self):
        ret = []
        for idx in range(len(self.vars)):
            if self.vars[idx].get() == 1:
                ret.append(self.allPicks[idx])
        if self.DescriptionElementAll in ret:
            ret.remove(self.DescriptionElementAll)
        return ret

    def GetOnlySelectedAndEnabled(self):
        ret = []
        for idx in range(len(self.vars)):
            if self.vars[idx].get() == 1 and self.checkbuttons[idx].cget('state') != 'disabled':
                ret.append(self.allPicks[idx])
        if self.DescriptionElementAll in ret:
            ret.remove(self.DescriptionElementAll)
        return ret

    def AllIsSelected(self):
        states = self.GetStates()
        if states[next(iter(states))] == 1:
            # Uncheck or check all items
            for chkButton in self.checkbuttons:
                chkButton.select()
        else:
            # Uncheck or check all items
            for chkButton in self.checkbuttons:
                chkButton.deselect()
        if self.CommandElementAll != None:
            self.CommandElementAll()
    
    @staticmethod
    def getMaxWord(words):
        maxWord = words[0]
        for w in words:
            if len(w) > len(maxWord):
                maxWord = w
        return maxWord