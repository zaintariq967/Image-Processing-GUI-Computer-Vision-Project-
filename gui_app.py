import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------- Global Variables ----------
original_img = None
processed_img = None

# ---------- Helper Functions ----------
def open_image():
    global original_img, processed_img

    file = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not file:
        return
    
    original_img = Image.open(file)
    processed_img = original_img.copy()
    
    display_original(original_img)
    display_output(processed_img)


def display_original(img):
    tk_img = ImageTk.PhotoImage(img.resize((300, 300)))
    original_panel.config(image=tk_img)
    original_panel.image = tk_img


def display_output(img):
    tk_img = ImageTk.PhotoImage(img.resize((300, 300)))
    output_panel.config(image=tk_img)
    output_panel.image = tk_img


def to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def to_pil(img):
    if len(img.shape) == 2:
        return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# ---------- Image Processing Ops ----------
def darken():
    global processed_img
    if processed_img is None: return
    cv_img = to_cv(processed_img)
    cv_img = cv2.subtract(cv_img, np.full(cv_img.shape, 50, dtype=np.uint8))
    processed_img = to_pil(cv_img)
    display_output(processed_img)


def lighten():
    global processed_img
    if processed_img is None: return
    cv_img = to_cv(processed_img)
    cv_img = cv2.add(cv_img, np.full(cv_img.shape, 50, dtype=np.uint8))
    processed_img = to_pil(cv_img)
    display_output(processed_img)


def contrast():
    global processed_img
    if processed_img is None: return
    cv_img = to_cv(processed_img)
    cv_img = cv2.convertScaleAbs(cv_img, alpha=1.5, beta=0)
    processed_img = to_pil(cv_img)
    display_output(processed_img)


def negative():
    global processed_img
    if processed_img is None: return
    cv_img = to_cv(processed_img)
    cv_img = 255 - cv_img
    processed_img = to_pil(cv_img)
    display_output(processed_img)


def histogram():
    if processed_img is None: return
    cv_img = to_cv(processed_img)
    plt.figure()
    for i, c in enumerate(['b', 'g', 'r']):
        plt.hist(cv_img[:, :, i].ravel(), bins=256, color=c)
    plt.title("Histogram")
    plt.show()


def hist_equalization():
    global processed_img
    if processed_img is None: return
    cv_img = to_cv(processed_img)
    ycrcb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    cv_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    processed_img = to_pil(cv_img)
    display_output(processed_img)


def ahe():
    global processed_img
    if processed_img is None: return
    cv_img = to_cv(processed_img)
    lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=40.0)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    cv_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    processed_img = to_pil(cv_img)
    display_output(processed_img)


def clahe_fun():
    global processed_img
    if processed_img is None: return
    cv_img = to_cv(processed_img)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    cv_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    processed_img = to_pil(cv_img)
    display_output(processed_img)


def pixel_info():
    if original_img is None: return
    w, h = original_img.size
    info_label.config(text=f"Width = {w}, Height = {h}")


# ---------- GUI Layout ----------
root = tk.Tk()
root.title("Image Processing GUI")
root.geometry("800x600")
root.config(bg="white")

title_label = tk.Label(root, text="Image Processing GUI", fg="red", bg="white", font=("Arial", 20))
title_label.pack(pady=10)

frame = tk.Frame(root, bg="white")
frame.pack()

original_panel = tk.Label(frame, bg="lightgray")
original_panel.grid(row=0, column=0, padx=20)

output_panel = tk.Label(frame, bg="lightgray")
output_panel.grid(row=0, column=1, padx=20)

btn_frame = tk.Frame(root, bg="white")
btn_frame.pack(pady=20)

# Button Grid
buttons = [
    ("Upload Image", open_image),
    ("Display", lambda: display_output(processed_img)),
    ("Pixel Info", pixel_info),
    ("Darken", darken),
    ("Lighten", lighten),
    ("Contrast", contrast),
    ("Negative", negative),
    ("Histogram", histogram),
    ("Hist Eq", hist_equalization),
    ("AHE", ahe),
    ("CLAHE", clahe_fun),
]

r = c = 0
for text, cmd in buttons:
    tk.Button(btn_frame, text=text, width=15, command=cmd).grid(row=r, column=c, padx=5, pady=5)
    c += 1
    if c == 3:
        c = 0
        r += 1

info_label = tk.Label(root, text="", bg="white", fg="blue", font=("Arial", 14))
info_label.pack(pady=5)

root.mainloop()
