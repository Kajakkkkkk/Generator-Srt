import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
from transcriber import transcribe_video_to_srt
import threading


def start_gui():
    def browse_video():
        file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if file_path:
            video_entry.delete(0, tk.END)
            video_entry.insert(0, file_path)

    def browse_output():
        file_path = filedialog.asksaveasfilename(defaultextension=".srt", filetypes=[("SRT files", "*.srt")])
        if file_path:
            output_entry.delete(0, tk.END)
            output_entry.insert(0, file_path)

    def run_transcription():
        video_path = video_entry.get()
        output_path = output_entry.get()
        model_size = model_var.get()

        if not video_path or not output_path:
            messagebox.showerror("Błąd", "Musisz wybrać plik wideo i lokalizację zapisu.")
            return

        threading.Thread(target=start_transcription, args=(video_path, output_path, model_size)).start()

    def start_transcription(video_path, output_path, model_size):
        progress["value"] = 0
        progress.update()

        success = transcribe_video_to_srt(video_path, output_path, model_size, update_progress)

        if success:
            messagebox.showinfo("Sukces", "Napisy zostały wygenerowane!")
        else:
            messagebox.showerror("Błąd", "Wystąpił problem podczas generowania napisów.")

    def update_progress(progress_value):
        progress["value"] = progress_value
        progress.update()

    root = tk.Tk()
    root.title("Whisper Subtitle Generator")
    root.geometry("800x600")
    root.attributes('-toolwindow', True)

    def toggle_fullscreen(event=None):
        root.attributes("-fullscreen", not root.attributes("-fullscreen"))

    root.bind("<F11>", toggle_fullscreen)

    FONT = ("Segoe UI", 10)
    BTN_STYLE = {"font": ("Segoe UI", 10, "bold"), "bg": "#4CAF50", "fg": "white", "padx": 10, "pady": 5}
    LABEL_STYLE = {"font": FONT, "bg": "#f2f2f2"}

    tk.Label(root, text="Plik wideo:", **LABEL_STYLE).grid(row=0, column=0, sticky="e", padx=10, pady=10)
    video_entry = tk.Entry(root, width=50, font=FONT)
    video_entry.grid(row=0, column=1, columnspan=2, padx=10)
    tk.Button(root, text="Wybierz...", command=browse_video, bg="#2196F3", fg="white", padx=8, pady=4).grid(row=0,
                                                                                                            column=3)

    tk.Label(root, text="Zapisz jako:", **LABEL_STYLE).grid(row=1, column=0, sticky="e", padx=10, pady=10)
    output_entry = tk.Entry(root, width=50, font=FONT)
    output_entry.grid(row=1, column=1, columnspan=2, padx=10)
    tk.Button(root, text="Zapisz jako...", command=browse_output, bg="#607D8B", fg="white", padx=8, pady=4).grid(row=1,
                                                                                                                 column=3)

    tk.Label(root, text="Model Whispera:", **LABEL_STYLE).grid(row=2, column=0, sticky="e", padx=10, pady=10)
    model_var = tk.StringVar(value="base")
    model_menu = tk.OptionMenu(root, model_var, "tiny", "base", "small", "medium", "large")
    model_menu.config(font=FONT)
    model_menu.grid(row=2, column=1, columnspan=2, sticky="w", padx=10)

    progress = Progressbar(root, orient="horizontal", length=300, mode="determinate")
    progress.grid(row=4, column=1, columnspan=2, pady=10)

    run_button = tk.Button(root, text="🎮 Start transkrypcji", command=run_transcription, **BTN_STYLE)
    run_button.grid(row=3, column=1, columnspan=2, pady=20)

    root.mainloop()
