import tkinter as tk
import tkinter.font as tkFont
import tkinter.ttk as ttk
from PIL import ImageTk, Image

class MyListBox:
    """use a ttk.TreeView as a multicolumn ListBox"""
    def __init__(self, parent, headers, itemSelectedCallBack):
        self.parent = parent
        self.headers = headers
        # Setup Widget
        # self.tree = None
        # container.pack(fill='both', expand=True)
        # create a treeview with dual scrollbars
        self.tree = ttk.Treeview(columns=headers, show="headings")
        vsb = ttk.Scrollbar(orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(column=0, row=0, sticky='nsew', in_=self.parent)
        vsb.grid(column=1, row=0, sticky='ns', in_=self.parent)
        hsb.grid(column=0, row=1, sticky='ew', in_=self.parent)
        self.parent.grid_columnconfigure(0, weight=1)
        self.parent.grid_rowconfigure(0, weight=1)
        self.tree.bind('<<TreeviewSelect>>', self.ListItemSelected)
        self.itemSelectedCallBack = itemSelectedCallBack
        ## Build tree
        # Add headers
        for col in headers:
            self.tree.heading(col, text=col.title(),command=lambda c=col: sortby(self.tree, c, 0))
            # adjust the column's width to the header string
            self.tree.column(col, width=tkFont.Font().measure(col.title()), anchor=tk.CENTER)
        
    def AddItems(self, items):
        self.items = items
        self.tree.delete(*self.tree.get_children())
        for item in items:
            self.tree.insert('', 'end', values=item)
            # adjust column's width if necessary to fit each value
            for ix, val in enumerate(item):
                col_w = tkFont.Font().measure(val)
                if self.tree.column(self.headers[ix],width=None)<col_w:
                    self.tree.column(self.headers[ix], width=col_w)
    
    def GetTotalItems(self):
        if not hasattr(self, 'items'):
            return 0
        return len(self.items)

    def ListItemSelected(self, a):
        curItem = self.tree.focus()
        selectedItem = self.tree.item(curItem)['values']
        self.itemSelectedCallBack(selectedItem)

def isnumeric(s):
    """test if a string is numeric"""
    for c in s:
        if c in "1234567890-.":
            numeric = True
        else:
            return False
    return numeric
    
def change_numeric(data):
    """if the data to be sorted is numeric change to float"""
    new_data = []
    if isnumeric(data[0][0]):
        # change child to a float
        for child, col in data:
            new_data.append((float(child), col))
        return new_data
    return data

def sortby(tree, col, descending):
    """sort tree contents when a column header is clicked on"""
    # grab values to sort
    data = [(tree.set(child, col), child) \
        for child in tree.get_children('')]
    # if the data to be sorted is numeric change to float
    data =  change_numeric(data)
    # now sort the data in place
    data.sort(reverse=descending)
    for ix, item in enumerate(data):
        tree.move(item[1], '', ix)
    # switch the heading so it will sort in the opposite direction
    tree.heading(col, command=lambda col=col: sortby(tree, col, \
        int(not descending)))



