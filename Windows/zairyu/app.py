import tkinter as tk
from tkinter import filedialog, messagebox
import os, csv
from model_runner import OCRModel
from extract_fields import extract_fields

ocr = OCRModel("deepseek-ai/deepseek-vl2-tiny")

def save_to_csv(data_list, path):
    keys = ["学籍番号"] + [k for k in data_list[0].keys() if k != "学籍番号"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in data_list:
            writer.writerow(row)

def process_single():
    path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
    if not path: return
    result = ocr.predict(path)
    fields = extract_fields(result)
    fields["学籍番号"] = os.path.splitext(os.path.basename(path))[0]
    save_path = filedialog.asksaveasfilename(defaultextension=".csv")
    if save_path:
        save_to_csv([fields], save_path)
        messagebox.showinfo("完成しました", "CSV ファイル保存しました")

def process_batch():
    folder = filedialog.askdirectory()
    if not folder: return
    results = []
    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            result = ocr.predict(os.path.join(folder, file))
            fields = extract_fields(result)
            fields["学籍番号"] = os.path.splitext(file)[0]
            results.append(fields)            
    save_path = filedialog.asksaveasfilename(defaultextension=".csv")
    if save_path:
        save_to_csv(results, save_path)
        messagebox.showinfo("完成", f"{len(results)} のファイル識別完了，CSV 保存しました")

root = tk.Tk()
root.title("在留カード データ認識ツール")
root.geometry("400x250")
tk.Label(root, text="認識モードを選びください", font=("Arial", 16)).pack(pady=10)
tk.Button(root, text="単一の画像の認識", command=process_single, width=30).pack(pady=10)
tk.Button(root, text="一括でフォルダーを識別する", command=process_batch, width=30).pack(pady=10)
root.mainloop()
