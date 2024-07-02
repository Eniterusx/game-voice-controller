import time

import pygetwindow as gw
import pyautogui as pg

from tkinter import *
from tkinter import ttk

from PIL import ImageTk, Image

available_keys = ["<None>", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                  'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                  'Y', 'Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'] # Can add more keys if needed

available_commands = ["<None>", "Up", "Down", "Left", "Right", "Stop", "Go", "Jump"] # Placeholder


class SelectWindowMenu:
    """A dropdown menu for selecting the window that will receive the inputs."""
    def __init__(self, master, status):
        self.master = master
        self.options_dict = self._get_windows()
        self.variable = StringVar(master)
        self.variable.set(status)
        self.img = self._get_icon("resources/dropdown.png")
        self.menu = self._configure_menu(master)
        self._update_windows()
        self._bind_dropdown_click()

    def _get_icon(self, path):
        img = Image.open(path).convert("RGBA")
        img = img.resize((13, 15))
        img = ImageTk.PhotoImage(img)
        return img

    def _configure_menu(self, master):
        menu = OptionMenu(master, self.variable, *self.options_dict.keys(), command=self._select_window)
        menu.config(indicatoron=0, image=self.img, compound=RIGHT, width=300, height=20)
        menu.pack(pady=10)
        return menu

    def _get_windows(self):
        """Get a list of windows and return a dictionary of window titles to window objects."""
        windows = gw.getWindowsWithTitle("")
        # remove empty entries
        windows = [window for window in windows if window.title != ""]
        # create a mapping of window titles to window objects
        windows = {window.title: window for window in windows}
        # strip non-ascii characters and limit to 36 characters, also enumerate the titles
        windows = {f"{i+1}. {''.join([char for char in title if ord(char) <= 0xFFFF])}"[:36]: window for i, (title, window) in enumerate(windows.items())}
        return windows
    
    def _update_windows(self):
        """Refresh the windows list"""
        self.options_dict = self._get_windows()
        menu = self.menu["menu"]
        menu.delete(0, "end")
        for option in self.options_dict.keys():
            menu.add_command(label=option, command=lambda value=option: self.variable.set(value))

    def _bind_dropdown_click(self):
        """Bind the dropdown click event to update the window list."""
        self.menu.bind('<Button-1>', lambda event: self._update_windows())

    def _select_window(self, _):
        global selected_window
        selected_window = self.options_dict[self.variable.get()]


class RebindsList:
    """A list of key rebinds."""
    def __init__(self, master, status):
        self.master = master
        self.variable = StringVar(master)
        self.variable.set(status)
        self.label = self._create_label(master)
        self.divider = self._create_divider(master)
        self.rebind_manager = self._create_rebind_manager(master)

        self.canvas = Canvas(master)
        self.scrollbar = Scrollbar(master)
        self.inner_frame = Frame(self.canvas)
        self._config_canvas()

        self.inner_frame.bind("<Configure>", self.on_frame_configure)
        self.rebinds = []
        self.add_rebind(3)

    def on_frame_configure(self, event=None):
        """Reset the scroll region to encompass the inner frame."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _create_label(self, master):
        label = Label(master, textvariable=self.variable)
        label.config(width=240, height=2, font=("Arial", 12, "bold"))
        label.pack()
        return label
    
    def _config_canvas(self):
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="top", fill="both", expand=True)

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        # unbind the default scroll event
        self.canvas.unbind("<MouseWheel>")

        # To actually center the inner frame, we need to wait until the canvas is fully rendered
        # This is a bit of a hack, but it works
        self.canvas.after(100, self._adjust_canvas_layout)

    def _adjust_canvas_layout(self):
        self.canvas.update_idletasks()
        self.inner_frame.update_idletasks()

        canvas_width = self.canvas.winfo_width()
        frame_width = self.inner_frame.winfo_reqwidth()
        center_x = (canvas_width - frame_width) // 2
        self.canvas.create_window(center_x, 0, window=self.inner_frame, anchor="nw")

    def _create_divider(self, master):
        divider = ttk.Separator(master, orient='horizontal')
        # Assuming master's width is known or can be obtained
        divider.pack(fill='x', pady=(0, 10))
        return divider
    
    def _create_rebind_manager(self, master):
        frame = Frame(master)
        frame.config(width=240, height=20)

        add_button = Button(frame, text="Add rebind", command=self.add_rebind, width=20, pady=5)
        add_button.pack(side=LEFT, padx=20)

        remove_button = Button(frame, text="Remove rebind", command=self.remove_rebind, width=20, pady=5)
        remove_button.pack(side=LEFT, padx=20)

        frame.pack(pady=(0, 10))
        return frame
    
    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling."""
        first, _ = self.canvas.yview()
        unit = 1 if event.delta < 0 else -1

        # Prevent scrolling past the top of the list
        if first == 0.0 and unit < 0:
            return
        
        self.canvas.yview_scroll(unit, "units")

    def add_rebind(self, n=1):
        for _ in range(n):
            rebind = KeyRebind(master=self.inner_frame)
            self.rebinds.append(rebind)

    def remove_rebind(self):
        if self.rebinds:
            rebind = self.rebinds.pop()
            rebind.frame.destroy()


class KeyRebind:
    """A key rebind entry."""
    def __init__(self, master):
        self.master = master
        self.frame = self._create_frame(master)

    def _create_frame(self, master):
        frame = Frame(master)
        frame.config(width=240, height=20, pady=5)

        command_label = Label(frame, text="Command:")
        command_label.pack(side=LEFT)

        command_dropdown = RebindDropdown(frame, available_commands[0], *available_commands)
        command_dropdown.menu.pack(side=LEFT)

        key_label = Label(frame, text="Key:")
        key_label.pack(side=LEFT, padx=(15, 0))

        key_dropdown = RebindDropdown(frame, available_keys[0], *available_keys)
        key_dropdown.menu.pack(side=LEFT)

        frame.pack()

        return frame

class RebindDropdown:
    """A dropdown menu for selecting rebind options."""
    def __init__(self, master, status, *options):
        self.master = master
        self.variable = StringVar(master)
        self.variable.set(status)
        self.img = self._get_icon("resources/dropdown.png")
        self.options = options
        self.menu = self._configure_menu(master)

    def _get_icon(self, path):
        img = Image.open(path).convert("RGBA")
        img = img.resize((13, 15))
        img = ImageTk.PhotoImage(img)
        return img

    def _configure_menu(self, master):
        menu = OptionMenu(master, self.variable, *self.options, command=self._save_rebind)
        menu.config(indicatoron=0, image=self.img, compound=RIGHT, width=70, height=20)
        return menu
    
    def _save_rebind(self, _):
        #TODO: handle the rebinds
        pass

root = Tk()
root.title("Voice commands to keybinds")
root.geometry("400x500")
root.resizable(False, False)

icon = Image.open("resources/blahaj.png")
icon = ImageTk.PhotoImage(icon)
root.iconphoto(False, icon)

window = SelectWindowMenu(master=root, status="Select the game window")
key_rebinds = RebindsList(master=root, status="Configure rebinds")

root.mainloop()
