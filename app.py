import sounddevice as sd
import threading
import time
import warnings

import tkinter as tk
from tkinter import *
from tkinter import ttk

from PIL import ImageTk, Image
from vad import VADModel

import platform
import os

import pyautogui as pg
if platform.system() == "Windows":
    import pygetwindow as gw
elif platform.system() == "Linux":
    import pywinctl as gw
else:
    raise NotImplementedError("This platform is not supported.")

available_keys = ["<None>", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                  'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                  'Y', 'Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
                  'LMB', 'RMB'] # Can add more keys if needed

available_commands = ["Left", "Right", "Up", "Down", "Go", "Stop", "On",
                      "Off", "Yes", "No", "Zero", "One", "Two", "Three",
                      "Four", "Five", "Six", "Seven", "Eight", "Nine"] # Can add more commands if needed

class KeyBindManager:
    def __init__(self, window):
        self.keybinds = []
        self.window = window
        self.model = VADModel(self._execute_command)
    
    def _add_keybind(self, key, command):
        if not key or not command or key == "<None>":
            return
        self.keybinds.append((key, command))
        # print(f"Keybind added: {key} -> {command}")

    def _remove_keybind(self, key, command):
        if not key or not command or key == "<None>":
            return
        if (key, command) in self.keybinds:
            self.keybinds.remove((key, command))
            # print(f"Keybind removed: {key} -> {command}")

    def _execute_command(self, command):
        print(f"Voice command detected: {command}")
        for key, cmd in self.keybinds:
            if cmd.lower() == command.lower():
                self._execute_keybind(key, command)

    def _execute_keybind(self, key, command):
        # print(f"Executing keybind: {key} -> {command}")
        if key == "<None>":
            return
        # send a key press event to the selected window
        try:
            # print(self.window.title)
            win = gw.getWindowsWithTitle(self.window.title)[0]
            win.activate()
            counter = 10
            while not win.isActive and counter > 0:
                time.sleep(0.01)
                counter -= 1
            if not win.isActive:
                warnings.warn("WARNING\nCould not activate the window")
            if win.isActive:
                if key == "LMB":
                    pg.mouseDown(button='left')
                    time.sleep(0.05)
                    pg.mouseUp(button='left')
                elif key == "RMB":
                    pg.mouseDown(button='right')
                    time.sleep(0.05)
                    pg.mouseUp(button='right')
                else:
                    pg.keyDown(key.lower())
                    time.sleep(0.05)
                    pg.keyUp(key.lower())
                print(f"Keybind executed: {key} -> {command}")
        except IndexError:
            print(f"Window titled: '{self.window} not found.")
        except Exception as e:
            print(f"Error executing keybind for {key}: {e}")

    def vad_listener(self):
        global start_stop
        with sd.InputStream(channels=1,
                            samplerate=self.model.sample_rate,
                            dtype='int16',
                            device=self.model.device_id,
                            blocksize=self.model.chunk_length,
                            callback=self.model.audio_callback):
            print("Listening for voice commands...")
            while start_stop.is_running:
                sd.sleep(1000)
        print("Voice command listener stopped.")


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
        windows = self._get_all_window_names("")
        # remove empty entries
        windows = [window for window in windows if window.title != ""]
        # create a mapping of window titles to window objects
        windows = {window.title: window for window in windows}
        # strip non-ascii characters and limit to 36 characters, also enumerate the titles
        windows = {f"{i+1}. {''.join([char for char in title if ord(char) <= 0xFFFF])}"[:36]: window for i, (title, window) in enumerate(windows.items())}
        return windows
    
    def _get_all_window_names(self, name):
        if platform.system() == "Windows":
            return gw.getWindowsWithTitle(name)
        elif platform.system() == "Linux":
            return gw.getAllWindows()
    
    def _update_windows(self):
        """Refresh the windows list"""
        self.options_dict = self._get_windows()
        menu = self.menu["menu"]
        menu.delete(0, "end")
        for option in self.options_dict.keys():
            menu.add_command(label=option, command=tk._setit(self.variable, option, self._select_window))

    def _bind_dropdown_click(self):
        """Bind the dropdown click event to update the window list."""
        self.menu.bind('<Button-1>', lambda event: self._update_windows())

    def _select_window(self, _):
        if self.variable.get() not in self.options_dict:
            return
        global selected_window
        selected_window = self.options_dict[self.variable.get()]
        # print(f"Selected window: {selected_window.title}")
        global key_bind_manager
        key_bind_manager.window = selected_window
        self._on_selection_change()
    
    def _on_selection_change(self):
        global start_stop
        if self.variable.get() not in self.options_dict:
            start_stop.button.config(state=DISABLED)
        else:
            start_stop.button.config(state=NORMAL)

class SelectDeviceMenu:
    def __init__(self, master, status):
        self.master = master
        self.options_dict = self._get_devices()
        self.variable = StringVar(master)
        self.variable.set(status)
        self.img = self._get_icon("resources/dropdown.png")
        self.menu = self._configure_menu(master)
        self._update_devices()

    def _get_icon(self, path):
        img = Image.open(path).convert("RGBA")
        img = img.resize((13, 15))
        img = ImageTk.PhotoImage(img)
        return img

    def _configure_menu(self, master):
        menu = OptionMenu(master, self.variable, *self.options_dict.keys(), command=self._select_device)
        menu.config(indicatoron=0, image=self.img, compound=RIGHT, width=300, height=20)
        menu.pack(pady=10)
        return menu
    
    def _get_devices(self): 
        devices = sd.query_devices()
        input_devices = {
            f"{device['name']} ({device['hostapi']})": device
            for device in devices
            if device['max_input_channels'] > 0 and device['default_samplerate'] > 0
        }
        # add a "Default" device, give it index of None, put it at the top of the list
        input_devices = {"Default device": {"index": None, "name": "Default"}, **input_devices}
        return input_devices
    
    def _update_devices(self):
        self.options_dict = self._get_devices()
        menu = self.menu["menu"]
        menu.delete(0, "end")
        for option in self.options_dict.keys():
            menu.add_command(label=option, command=tk._setit(self.variable, option, self._select_device))

    def _select_device(self, _):
        if self.variable.get() not in self.options_dict:
            return
        device = self.options_dict[self.variable.get()]
        global key_bind_manager
        key_bind_manager.model.device_id = device['index']

class StartStopButton:
    def __init__(self, master, status):
        self.master = master
        self.variable = StringVar(master)
        self.variable.set(status)
        self.button = Button(master, textvariable=self.variable, command=self._toggle, width=30, state=DISABLED)
        self.button.pack(pady=10)
        self.is_running = False

    def _toggle(self):
        if self.is_running:
            self.variable.set("Start")
            self.is_running = False
        else:
            self.variable.set("Stop")
            self.is_running = True
            global selected_window
            global key_bind_manager
            threading.Thread(target=key_bind_manager.vad_listener, daemon=True).start()


class RebindsListGUI:
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
        self.rebind_guis = []
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
            rebind = KeyRebindGUI(master=self.inner_frame)
            self.rebind_guis.append(rebind)

    def remove_rebind(self):
        if self.rebind_guis:
            self.rebind_guis[-1]._destroy()
            self.rebind_guis.pop()


class KeyRebindGUI:
    """A key rebind entry."""
    def __init__(self, master):
        self.master = master
        self.key = None
        self.command = None
        self.frame = self._create_frame(master)

    def _create_frame(self, master):
        frame = Frame(master)
        frame.config(width=240, height=20, pady=5)

        command_label = Label(frame, text="Command:")
        command_label.pack(side=LEFT)

        command_dropdown = RebindDropdown(frame, available_commands[0], self._set_command, *available_commands)
        command_dropdown.menu.pack(side=LEFT)

        key_label = Label(frame, text="Key:")
        key_label.pack(side=LEFT, padx=(15, 0))

        key_dropdown = RebindDropdown(frame, available_keys[0], self._set_key, *available_keys)
        key_dropdown.menu.pack(side=LEFT)

        frame.pack()

        return frame
    
    def _destroy(self):
        global key_bind_manager
        key_bind_manager._remove_keybind(self.key, self.command)
        self.frame.destroy()
    
    def _set_key(self, key):
        self._set_rebind(key, self.command)
        self.key = key

    def _set_command(self, command):
        self._set_rebind(self.key, command)
        self.command = command
    
    def _set_rebind(self, key, command):
        global key_bind_manager
        key_bind_manager._remove_keybind(self.key, self.command)
        key_bind_manager._add_keybind(key, command)

class RebindDropdown:
    """A dropdown menu for selecting rebind options."""
    def __init__(self, master, status, callback, *options):
        self.master = master
        self.variable = StringVar(master)
        self.variable.set(status)
        self.img = self._get_icon("resources/dropdown.png")
        self.options = options
        self.callback = callback
        self.menu = self._configure_menu(master)

    def _get_icon(self, path):
        img = Image.open(path).convert("RGBA")
        img = img.resize((13, 15))
        img = ImageTk.PhotoImage(img)
        return img

    def _configure_menu(self, master):
        menu = OptionMenu(master, self.variable, *self.options, command=self._save_rebind)
        menu.config(indicatoron=0, image=self.img, compound=RIGHT, width=70, height=20)
        self._save_rebind(None)
        return menu
    
    def _save_rebind(self, _):
        self.callback(self.variable.get())

selected_window = None
key_bind_manager = KeyBindManager(None)

root = Tk()
root.title("Voice to keybinds")
root.geometry("400x550")
root.resizable(False, False)

if os.path.exists("resources/blahaj.png"):
    icon = Image.open("resources/blahaj.png")
    icon = ImageTk.PhotoImage(icon)
    root.iconphoto(False, icon)

window = SelectWindowMenu(master=root, status="Select the game window")
device_id = SelectDeviceMenu(master=root, status="Select the input device")
start_stop = StartStopButton(master=root, status="Start")
key_rebinds = RebindsListGUI(master=root, status="Configure rebinds")

root.mainloop()


