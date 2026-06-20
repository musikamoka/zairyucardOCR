import tkinter as tk
from tkinter import filedialog, messagebox
import os, csv
from model_runner import OCRModel
from extract_fields import extract_fields

# 默认还没初始化模型
ocr = None  
device_choice = None  # 用来保存 GPU/CPU 选择

def init_ocr():#初始化
    global ocr 
    use_cpu = (device_choice.get() == "cpu")
    ocr = OCRModel("deepseek-ai/deepseek-vl2-tiny", use_cpu=use_cpu)

ocr = None  # 占位，点击按钮时再初始化
def save_to_csv(data_list, path):
    keys = ["学籍番号"] + [k for k in data_list[0].keys() if k != "学籍番号"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in data_list:
            writer.writerow(row)

#保存单张
def process_single():
    if ocr is None:    
        init_ocr()
    path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
    if not path: return
    result = ocr.predict(path)#原始结果
    fields = result if isinstance(result, dict) else extract_fields(result)#处理后的结果
    fields["学籍番号"] = os.path.splitext(os.path.basename(path))[0]
    
    answer = messagebox.askyesno("確認", "1 枚認識しました。CSV に保存しますか？")#询问保存
    if answer: #yes
        save_path = filedialog.asksaveasfilename(defaultextension=".csv")
        if save_path:
            save_to_csv([fields], save_path)
            messagebox.showinfo("完成しました", "CSV ファイル保存しました")
    else:  # 不保存，只显示
        # 控制台打印完整结果
        print("=== OCR 原始結果 ===")
        print(result)
        messagebox.showinfo("OCR 原始結果", str(result))

#保存批量
def process_batch():
    if ocr is None:    
        init_ocr()
    folder = filedialog.askdirectory()
    if not folder: return
    results = []
    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            result = ocr.predict(os.path.join(folder, file))
            fields = result if isinstance(result, dict) else extract_fields(result)#处理后的结果
            fields["学籍番号"] = os.path.splitext(file)[0]
            results.append(fields)            
            
    answer = messagebox.askyesno("確認", f"{len(results)} 枚認識しました。CSV に保存しますか？")#询问保存
    if answer: #yes
        save_path = filedialog.asksaveasfilename(defaultextension=".csv")
        if save_path:
            save_to_csv(results, save_path)
            messagebox.showinfo("完成", f"{len(results)} のファイル識別完了，CSV 保存しました")
    else:  # 不保存，只显示
        # 控制台打印完整结果
        for r in results:
            print(r)
        messagebox.showinfo("OCR 原始結果", f"{len(results)} 枚の原始結果はコンソールに出力しました")

root = tk.Tk()
root.title("在留カード データ認識ツール")
root.geometry("400x250")
tk.Label(root, text="認識モードを選びください", font=("Arial", 16)).pack(pady=10)
# 🆕 CPU/GPU 选择
frame = tk.Frame(root)
frame.pack(pady=5)
tk.Label(frame, text="デバイス:").pack(side=tk.LEFT)
device_choice = tk.StringVar(value="gpu")  # 默认 GPU
tk.Radiobutton(frame, text="GPU", variable=device_choice, value="gpu").pack(side=tk.LEFT)
tk.Radiobutton(frame, text="CPU", variable=device_choice, value="cpu").pack(side=tk.LEFT)
tk.Button(root, text="１枚の在留カードを認識する", command=process_single, width=30).pack(pady=10)
tk.Button(root, text="複数の在留カードを認識する", command=process_batch, width=30).pack(pady=10)
root.mainloop()
