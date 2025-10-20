import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
import threading
import queue
import os

from transcriber import process_video, prefetch_model

def start_gui():
    root = tk.Tk()
    root.title("Whisper SRT (PL) – Faster-Whisper")
    root.geometry("960x660")
    root.configure(bg="#f2f2f2")

    ui_q = queue.Queue()
    cancel_flag = [False]

    def post_progress(val, eta_text=None):
        ui_q.put(("progress", int(max(0, min(100, val))), eta_text))

    def post_done(success, files):
        ui_q.put(("done", success, files))

    def post_info(text):
        ui_q.put(("info", text))

    def pump():
        try:
            while True:
                item = ui_q.get_nowait()
                kind = item[0]
                if kind == "progress":
                    percent, eta_text = item[1], item[2]
                    progress["value"] = percent
                    eta_var.set(eta_text or "")
                elif kind == "done":
                    run_btn.config(state="normal")
                    cancel_btn.config(state="disabled")
                    progress["value"] = 100 if item[1] else 0
                    eta_var.set("")
                    if item[1]:
                        msg = "Wygenerowano pliki:\n\n" + "\n".join(item[2])
                        messagebox.showinfo("Sukces", msg)
                    else:
                        messagebox.showerror("Przerwano / Błąd", "Przetwarzanie nie zostało ukończone.")
                elif kind == "info":
                    status_var.set(str(item[1]))
        except queue.Empty:
            pass
        root.after(50, pump)

    def browse_video():
        p = filedialog.askopenfilename(
            filetypes=[("Wideo/Audio", "*.mp4;*.mkv;*.mov;*.m4a;*.wav;*.flac"), ("Wszystkie", "*.*")]
        )
        if p:
            video_var.set(p)
            base = os.path.splitext(p)[0]
            out_var.set(base)

    def browse_output_base():
        p = filedialog.asksaveasfilename(defaultextension=".srt", filetypes=[("SRT", "*.srt")])
        if p:
            if p.endswith(".srt"):
                p = p[:-4]
            out_var.set(p)

    def on_prefetch_model():
        model_size = model_var.get()
        def worker():
            try:
                post_info("Pobieranie modelu Whisper…")
                path = prefetch_model(model_size)
                post_info(f"Model Whisper gotowy: {path}")
                messagebox.showinfo("Model", f"Model pobrany/gotowy:\n{path}")
            except Exception as e:
                post_info("Błąd pobierania modelu")
                messagebox.showerror("Model", f"Nie udało się pobrać: {e}")
        threading.Thread(target=worker, daemon=True).start()

    def on_run():
        video_path = video_var.get().strip()
        output_base = out_var.get().strip()
        model_size = model_var.get()
        device = device_var.get()
        compute_type = ctype_var.get()
        workers = int(workers_var.get())
        vad_on = bool(vad_var.get())
        min_silence = int(minsil_var.get())
        beam = int(beam_var.get())
        merge_chars = int(merge_chars_var.get())
        merge_gap = float(merge_gap_var.get())

        enable_presplit = bool(presplit_var.get())
        presplit_minutes = int(presplit_min_var.get())

        # NOWE: soft/hard limit
        soft_max = int(soft_max_var.get())
        hard_max = int(hard_max_var.get())

        if not video_path or not output_base:
            messagebox.showerror("Błąd", "Wybierz plik wideo i bazową nazwę wyjściową.")
            return

        run_btn.config(state="disabled")
        cancel_btn.config(state="normal")
        progress["value"] = 0
        eta_var.set("")
        status_var.set("")
        cancel_flag[0] = False

        def worker():
            ok, files = process_video(
                video_path=video_path,
                output_base=output_base,
                model_size=model_size,
                compute_type=compute_type,         # 'float16' polecany na 4090
                device=device,                     # 'cuda' / 'cpu'
                num_workers=workers,
                vad_filter=vad_on,
                min_silence_ms=min_silence,
                beam_size=beam,
                progress_cb=post_progress,
                cancel_flag=cancel_flag,
                prefetch=True,
                merge_max_chars=merge_chars,
                merge_max_gap=merge_gap,
                enable_presplit=enable_presplit,
                presplit_minutes=presplit_minutes,
                # NOWE: miękki/twardy limit na wpis
                soft_max_chars=soft_max,
                hard_max_chars=hard_max,
            )
            post_done(ok, files)

        threading.Thread(target=worker, daemon=True).start()

    def on_cancel():
        cancel_flag[0] = True
        cancel_btn.config(state="disabled")

    # ---------------- UI ----------------
    FONT = ("Segoe UI", 10)

    # Wiersz 0
    tk.Label(root, text="Plik wideo:", bg="#f2f2f2", font=FONT).grid(row=0, column=0, sticky="e", padx=10, pady=10)
    video_var = tk.StringVar()
    tk.Entry(root, textvariable=video_var, width=62, font=FONT).grid(row=0, column=1, columnspan=5, padx=10)
    tk.Button(root, text="Wybierz...", command=browse_video, bg="#2196F3", fg="white", padx=10, pady=6).grid(row=0, column=6, padx=5)

    # Wiersz 1
    tk.Label(root, text="Baza nazwy wyjściowej:", bg="#f2f2f2", font=FONT).grid(row=1, column=0, sticky="e", padx=10, pady=10)
    out_var = tk.StringVar()
    tk.Entry(root, textvariable=out_var, width=62, font=FONT).grid(row=1, column=1, columnspan=5, padx=10)
    tk.Button(root, text="Zapisz jako...", command=browse_output_base, bg="#607D8B", fg="white", padx=10, pady=6).grid(row=1, column=6, padx=5)

    # Wiersz 2 – model / device / compute
    tk.Label(root, text="Model Whisper:", bg="#f2f2f2", font=FONT).grid(row=2, column=0, sticky="e", padx=10, pady=10)
    model_var = tk.StringVar(value="large-v3")
    tk.OptionMenu(root, model_var, "large-v3").grid(row=2, column=1, sticky="w")

    tk.Button(root, text="Pobierz model", command=on_prefetch_model, bg="#8E24AA", fg="white", padx=12, pady=6).grid(row=2, column=2, padx=10)

    tk.Label(root, text="Urządzenie:", bg="#f2f2f2", font=FONT).grid(row=2, column=3, sticky="e")
    device_var = tk.StringVar(value="cuda")
    tk.OptionMenu(root, device_var, "cuda", "cpu").grid(row=2, column=4, sticky="w")

    tk.Label(root, text="Compute type:", bg="#f2f2f2", font=FONT).grid(row=2, column=5, sticky="e")
    ctype_var = tk.StringVar(value="float16")
    tk.OptionMenu(root, ctype_var, "float16", "float32", "int8_float16").grid(row=2, column=6, sticky="w")

    # Wiersz 3 – workers/VAD/silence/beam
    tk.Label(root, text="Workers:", bg="#f2f2f2", font=FONT).grid(row=3, column=0, sticky="e")
    workers_var = tk.StringVar(value="6")
    tk.Entry(root, textvariable=workers_var, width=6, font=FONT).grid(row=3, column=1, sticky="w")

    tk.Label(root, text="VAD:", bg="#f2f2f2", font=FONT).grid(row=3, column=2, sticky="e")
    vad_var = tk.IntVar(value=1)
    tk.Checkbutton(root, variable=vad_var, bg="#f2f2f2").grid(row=3, column=3, sticky="w")

    tk.Label(root, text="Min silence (ms):", bg="#f2f2f2", font=FONT).grid(row=3, column=4, sticky="e")
    minsil_var = tk.StringVar(value="600")
    tk.Entry(root, textvariable=minsil_var, width=8, font=FONT).grid(row=3, column=5, sticky="w")

    tk.Label(root, text="Beam size:", bg="#f2f2f2", font=FONT).grid(row=4, column=0, sticky="e")
    beam_var = tk.StringVar(value="5")
    tk.Entry(root, textvariable=beam_var, width=6, font=FONT).grid(row=4, column=1, sticky="w")

    # Wiersz 4 – scalanie
    tk.Label(root, text="Łączenie (max znaki):", bg="#f2f2f2", font=FONT).grid(row=4, column=2, sticky="e")
    merge_chars_var = tk.StringVar(value="140")
    tk.Entry(root, textvariable=merge_chars_var, width=8, font=FONT).grid(row=4, column=3, sticky="w")

    tk.Label(root, text="Łączenie (max przerwa s):", bg="#f2f2f2", font=FONT).grid(row=4, column=4, sticky="e")
    merge_gap_var = tk.StringVar(value="0.8")
    tk.Entry(root, textvariable=merge_gap_var, width=8, font=FONT).grid(row=4, column=5, sticky="w")

    # Wiersz 5 – pre-split
    tk.Label(root, text="Pre-split:", bg="#f2f2f2", font=FONT).grid(row=5, column=0, sticky="e")
    presplit_var = tk.IntVar(value=1)
    tk.Checkbutton(root, variable=presplit_var, bg="#f2f2f2").grid(row=5, column=1, sticky="w")

    tk.Label(root, text="Co ile minut:", bg="#f2f2f2", font=FONT).grid(row=5, column=2, sticky="e")
    presplit_min_var = tk.StringVar(value="9")
    tk.Entry(root, textvariable=presplit_min_var, width=6, font=FONT).grid(row=5, column=3, sticky="w")

    # Wiersz 6 – soft/hard limity napisów
    tk.Label(root, text="Max znaki (soft):", bg="#f2f2f2", font=FONT).grid(row=6, column=2, sticky="e")
    soft_max_var = tk.StringVar(value="80")
    tk.Entry(root, textvariable=soft_max_var, width=6, font=FONT).grid(row=6, column=3, sticky="w")

    tk.Label(root, text="Max znaki (hard):", bg="#f2f2f2", font=FONT).grid(row=6, column=4, sticky="e")
    hard_max_var = tk.StringVar(value="90")
    tk.Entry(root, textvariable=hard_max_var, width=6, font=FONT).grid(row=6, column=5, sticky="w")

    # Pasek postępu + status
    progress = Progressbar(root, orient="horizontal", length=680, mode="determinate", maximum=100)
    progress.grid(row=7, column=1, columnspan=5, pady=20)

    eta_var = tk.StringVar(value="")
    tk.Label(root, textvariable=eta_var, bg="#f2f2f2", font=("Segoe UI", 9, "italic")).grid(row=8, column=1, columnspan=5)

    status_var = tk.StringVar(value="")
    tk.Label(root, textvariable=status_var, bg="#f2f2f2", font=("Segoe UI", 9)).grid(row=9, column=1, columnspan=5)

    # Przyciski Start/Cancel
    run_btn = tk.Button(root, text="Start (PL → SRT)", command=on_run, bg="#4CAF50", fg="white", padx=16, pady=10, font=("Segoe UI", 11, "bold"))
    run_btn.grid(row=10, column=2, pady=10)

    cancel_btn = tk.Button(root, text="⏹ Cancel", command=on_cancel, bg="#E53935", fg="white", padx=16, pady=10, font=("Segoe UI", 11, "bold"))
    cancel_btn.grid(row=10, column=3, pady=10)
    cancel_btn.config(state="disabled")

    root.after(50, pump)
    root.mainloop()

if __name__ == "__main__":
    start_gui()
