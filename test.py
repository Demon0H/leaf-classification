import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
from tkinter import filedialog, messagebox

# 加载模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 加载训练好的模型权重
model = SimpleCNN()
model.load_state_dict(torch.load('leaf_classification_model.pth', map_location=torch.device('cpu')))
model.eval()

# 图片预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 转为灰度图
    transforms.Resize((28, 28)),  # 缩放到 28x28
    transforms.ToTensor(),  # 转为张量
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])

# GUI 界面
class LeafClassificationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("枫叶与柳叶分类器")
        self.geometry("600x500")

        # 图片变量
        self.image = None
        self.tk_image = None
        self.selected_area = None

        # 创建界面组件
        self.create_widgets()

    def create_widgets(self):
        # 创建按钮
        self.load_button = tk.Button(self, text="加载图片", command=self.load_image)
        self.load_button.pack(pady=10)

        self.select_button = tk.Button(self, text="选择区域", command=self.select_area)
        self.select_button.pack(pady=10)

        self.predict_button = tk.Button(self, text="预测", command=self.predict)
        self.predict_button.pack(pady=10)

        # 图片显示区域
        self.canvas = tk.Canvas(self, width=400, height=400)
        self.canvas.pack()

    def load_image(self):
        # 打开文件对话框，选择图片
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image = Image.open(file_path)
            self.tk_image = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def select_area(self):
        if self.image:
            # 允许用户选择方形区域
            self.canvas.bind("<ButtonPress-1>", self.on_button_press)
            self.canvas.bind("<B1-Motion>", self.on_move_press)
            self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def on_button_press(self, event):
        # 记录起点
        self.start_x = event.x
        self.start_y = event.y

    def on_move_press(self, event):
        # 绘制选择框
        self.canvas.delete("selection")
        self.end_x = event.x
        self.end_y = event.y
        self.canvas.create_rectangle(self.start_x, self.start_y, self.end_x, self.end_y, outline="red", tags="selection")

    def on_button_release(self, event):
        # 记录选择的区域
        self.canvas.unbind("<ButtonPress-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        self.selected_area = (self.start_x, self.start_y, self.end_x, self.end_y)

    def predict(self):
        if self.selected_area:
            # 裁剪选择的区域
            x1, y1, x2, y2 = self.selected_area
            cropped_image = self.image.crop((x1, y1, x2, y2))

            # 转为黑白并缩放到 28x28
            resized_image = cropped_image.resize((28, 28))
            grayscale_image = resized_image.convert("L")

            # 显示预处理后的图片
            self.tk_image = ImageTk.PhotoImage(grayscale_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

            # 转为张量并进行预测
            tensor_image = transform(grayscale_image).unsqueeze(0)
            with torch.no_grad():
                output = model(tensor_image)
                _, predicted = torch.max(output, 1)
                result = "枫叶" if predicted.item() == 0 else "柳叶"

            # 显示预测结果
            messagebox.showinfo("预测结果", f"预测结果为: {result}")
        else:
            messagebox.showwarning("警告", "请先选择一个区域！")

# 运行应用
if __name__ == "__main__":
    app = LeafClassificationApp()
    app.mainloop()