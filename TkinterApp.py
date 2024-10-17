from tkinter import *
from ultralytics import YOLO
import cv2
from PIL import Image, ImageTk
import time
from tkinter import filedialog
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np
import functools

# Decorators is used here to log and track
# Decorator to log the execution time of methods
# This is useful to monitor performance and identify slow operations in the application
def log_execution_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time for {func.__name__}: {end_time - start_time:.2f} seconds")
        return result
    return wrapper

# Decorator to track function calls
# This is for debugging by printing whenever a function called
def track_function_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Function {func.__name__} called")
        return func(*args, **kwargs)
    return wrapper

# Mixin class for additional image processing functionality (Multiple Inheritance)
# This class is used to indicate multiple inheritance by providing preprocessing functionality
class ImageProcessingMixin:
    def preprocess_image(self, image):
        # Example preprocessing function
        print("Preprocessing image (grayscale conversion)")
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# YOLOv8 class with multiple inheritance and method overriding
class YOLOv8App(ImageProcessingMixin):
    def __init__(self, master):
        # Initialize the Tkinter frame
        self.master = master
        self.frame = Frame(master)
        # This is used to enlarge the image to fit the window when the user maximizes the app window
        self.frame.pack(fill=BOTH, expand=True)

        # Encapsulation: Private model attribute
        # The YOLO model is encapsulated as a private attribute to prevent direct modification
        self.__model = YOLO('yolov8n.pt')

        # Create left and right frames, 
        # Left frame is used to show original image, right frame is used to show detected image
        # total width of window is 800, thus, left width and right width is all 400.
        self.left_frame = Frame(self.frame, width=400)
        self.left_frame.pack(side=LEFT, fill=BOTH, expand=True)
        self.right_frame = Frame(self.frame, width=400)
        self.right_frame.pack(side=RIGHT, fill=BOTH, expand=True)

        # Left frame contents
        # btn_load_img is the left button in the first line to load image from local,
        self.btn_load_img = Button(
            self.left_frame,
            text="Load Image From Local",
            width=20,
            command=self.load_image,
            font=("Times New Roman", 14, "bold"),
            borderwidth=4,
        )
        self.btn_load_img.pack(pady=50)
        #canvas_original_img is a convas used to display original image, Canvas is a class in Tkinter
        self.canvas_original_img = Canvas(self.left_frame, width=400, height=600)
        #canvas_original_img will expand in the y and x direction when the size of the frame changes. 
        self.canvas_original_img.pack(fill=BOTH, expand=True)

        # Right frame contents
        # btn_detect_img is the right button in the first line to detect objects from original image,
        self.btn_detect_img = Button(
            self.right_frame,
            text="Click here to Detect Objects",
            width=20,
            command=self.detect_objects,
            font=("Times New Roman", 14, "bold"),
            borderwidth=4,
            # when image is not loaded, the state of btn_detect_img is DISABLED.
            state=DISABLED)  
        self.btn_detect_img.pack(pady=50)
        #canvas_detected_img is the canvas to show detected image
        self.canvas_detected_img = Canvas(self.right_frame, width=400, height=600)
        self.canvas_detected_img.pack(fill=BOTH, expand=True)

        # Bind the configure event to update_canvases when window changes
        # when window changes, <Configure> will get new size from window and function update_canvas_size will update new size of images
        self.master.bind('<Configure>', self.update_canvas_size)

    # Load image from local directory
    # This function allows the user to load an image from their computer
    @log_execution_time  # Log execution time
    @track_function_call  # Track function call
    def load_image(self):
        img_path = filedialog.askopenfilename() # a dialog box will appear and the user need to select an image to get image path from local directory
        if img_path:
            self.original_image = cv2.imread(img_path)  # use opencv to read image
            self.display_original_image()  # display original image in the left frame
            self.btn_detect_img.config(state=NORMAL)  # after user loads image, detect button is enabled 

    # Display original image in the left frame
    def display_original_image(self):
        # Convert OpenCV BGR image to RGB, image_in_rgb is a numpy array because the return of cv2.cvtcolor is a numpy array
        image_in_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        # image_in_rgb is a numpy array, thus, we should convert numpy array to PIL Image to display in the frame
        image_pil = Image.fromarray(image_in_rgb)
        # Resize and display image
        self.display_image(image_pil, self.canvas_original_img)

    # Detect objects from original image using YOLOv8
    @log_execution_time  # Log execution time
    @track_function_call  # Track function call
    def detect_objects(self):
        try:  #to check that the original image has been loaded before using the detection object.
            self.original_image
            # detect original image by model Yolov8 and return annotated image
            annotated_image = self.process_image(self.original_image)  
            # Convert OpenCV BGR image to RGB
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            image_pil = Image.fromarray(annotated_image_rgb)
            # Resize and display image
            self.display_image(image_pil, self.canvas_detected_img)  # Display the detected image
        except AttributeError:
            print("No image loaded")

    # Display image in the canvas
    def display_image(self, image_pil, canvas):
        # use the following two methods to get width and height of canvas 
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        # when canvas changes, use thumbnail to resize image in original ratio of width and height to fit canvas, 
        # the image is changed in place.
        image_pil.thumbnail((canvas_width, canvas_height))
        # Convert a PIL image to a PhotoImage which can be used in Tkinter.
        image_tkinter_mode = ImageTk.PhotoImage(image_pil)

        # image_tkinter_mode will be deleted from window due to garbage collection
        # use variable tk_image_original or tk_image_detected to store a reference to prevent garbage collection 
        # and keep image visiable in the canvas
        # check which canvas, if it is the canvas for the original image, store image_tkinter_mode in the tk_image_original
        if canvas == self.canvas_original_img:
            self.tk_image_original = image_tkinter_mode
        else:
            #if it is the canvas for the detected image, store image_tkinter_mode in the tk_image_detected
            self.tk_image_detected = image_tkinter_mode

        # use tkinter function create_image to display the image in the canvas
        canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=CENTER, image=image_tkinter_mode)

    # update images to fit new canvas size when window changes
    # when window changes, the event will change with new canvas width and height
    # and this method will be called
    def update_canvas_size(self, event=None):
        
        if hasattr(self, 'original_image'):
            # display the original image again with new canvas size
            self.display_original_image()
        if hasattr(self, 'tk_image_detected'):
            # display the detected image again with new canvas size
            self.detect_objects()

    # Method overriding - process_image for YOLOv8
    # This function is overridden to process the image using YOLOv8
    @log_execution_time  # Log execution time
    @track_function_call  # Track function call
    def process_image(self, image):
        results = self.__model(image)  # Use Yolov8 model to detect objects
        return results[0].plot()  # to draw bounding boxes, labels and confidence on annotated images

# ResNet50 class for image classification
class ResNet50App(ImageProcessingMixin):
    def __init__(self, master):
        # opening the Tkinter frame
        self.master = master
        self.frame = Frame(master)
        self.frame.pack(fill=BOTH, expand=True)

        #ResNet50 model is loading here (Encapsulation: model is private)
        self.__model = ResNet50(weights='imagenet')

        # Create left and right frames inside the main content frame
        # Left frame is used to show original image, right frame is used to show predicted results of that image
        self.left_frame = Frame(self.frame, width=400)
        self.left_frame.pack(side=LEFT, fill=BOTH, expand=True)
        self.right_frame = Frame(self.frame, width=400)
        self.right_frame.pack(side=RIGHT, fill=BOTH, expand=True)

        # Left frame contents 
        # btn_load_img is the left button to load image from local device,
        self.btn_load_img = Button(
            self.left_frame,
            text="Load Image From Local",
            width=20,
            command=self.load_image,
            font=("Times New Roman", 14, "bold"),
            borderwidth=4,
        )
        self.btn_load_img.pack(pady=50)
        #canvas_original_img is a convas used to display original image, Canvas is a class in Tkinter
        self.canvas_original_img = Canvas(self.left_frame, width=400, height=600)
        self.canvas_original_img.pack(fill=BOTH, expand=True)

        # Right frame contents
        # label_prediction is showing the predicted result in rigt frame
        self.label_prediction = Label(self.right_frame, text="Prediction will appear here", font=("Arial", 14))
        self.label_prediction.pack(pady=50)

        # Bind the configure event to update canvases when window changes
        # when window changes, <Configure> will get new size from window and function update_canvas_size will update new size of canvas
        self.master.bind('<Configure>', self.update_canvas_size)

    # Load image from local directory
    @log_execution_time  # Log execution time
    @track_function_call  # Track function call
    def load_image(self):
        img_path = filedialog.askopenfilename() # a dialog box will appear and the user need to select an image to get image path from local device
        if img_path:
            self.original_image = cv2.imread(img_path)  # Load image using OpenCV
            self.display_original_image()  # Display the loaded image
            self.classify_image()  # Classify the loaded image

    # Display the original image in the left frame
    def display_original_image(self):
        image_in_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_in_rgb)
        self.display_image(image_pil, self.canvas_original_img)

    # Display image in the specified canvas
    def display_image(self, image_pil, canvas):
        # Resize the image to fit the canvas and maintaining size
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        image_pil.thumbnail((canvas_width, canvas_height))
        image_tkinter_mode = ImageTk.PhotoImage(image_pil)
        self.tk_image_original = image_tkinter_mode  # Store a reference to avoid garbage collection
        canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=CENTER, image=image_tkinter_mode)

    # Classify image using ResNet50
    @log_execution_time  # Log execution time
    @track_function_call  # Track function call
    def classify_image(self):
        # Preprocesing the image for ResNet50
        image_in_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_in_rgb, (224, 224))
        image_array = np.expand_dims(image_resized, axis=0)
        image_array = preprocess_input(image_array)
        preds = self.__model.predict(image_array)  # Predict result using ResNet50
        decoded_preds = decode_predictions(preds, top=3)[0]
        # Format the predictions and update the label
        prediction_text = "\n".join([f"{label}: {round(prob * 100, 2)}%" for (_, label, prob) in decoded_preds])
        self.label_prediction.config(text=prediction_text)

    # Update images to fit new canvas size when window changes
    def update_canvas_size(self, event=None):
        # Resize the images when the window size changes
        if hasattr(self, 'original_image'):
            self.display_original_image()

# Main Tkinter application with sidebar menu to switch between YOLOv8 and ResNet50 models
class TkinterApp:
    
    def __init__(self, master):
        self.master = master
        self.master.title("TkinterApp")
        self.master.geometry("1000x650") # Set initial window size

        # Main content area frame
        self.main_frame = Frame(self.master)
        self.main_frame.pack(fill=BOTH, expand=True)

        # Sidebar frame
        # Two Buttons to switch between two models
        self.sidebar_frame = Frame(self.main_frame, width=100, bg='darkslategray')
        self.sidebar_frame.pack(side=LEFT, fill=Y, expand=False)

        # Sidebar content 
        # Icon and buttons
        # a FontAwesome icon is used to the sidebar for a visually appealing element
        self.sidebar_menu_label = Label(self.sidebar_frame, text='✨ Menu ✨', font=('Helvetica', 18, 'bold'), fg='white', bg='darkslategray')
        self.sidebar_menu_label.pack(pady=20)

        # Button for YOLOv8 detection
        self.sidebar_button_1 = Button(self.sidebar_frame, text="Yolov8", command=self.sidebar_action_1, font=('Arial', 10, 'bold'), fg='black', bg='#FF6F61', activebackground='#FF8A80', relief='flat', padx=10, pady=5)
        self.sidebar_button_1.pack(pady=5)

        # Button for ResNet50 classification
        self.sidebar_button_2 = Button(self.sidebar_frame, text="ResNet50", command=self.sidebar_action_2, font=('Arial', 10, 'bold'), fg='black', bg='#FFA726', activebackground='#FFCC80', relief='flat', padx=10, pady=5)
        self.sidebar_button_2.pack(pady=5)

        # Main content frame
        # It display the selected application model
        self.content_frame = Frame(self.main_frame, width=850, bg='white')
        self.content_frame.pack(side=RIGHT, fill=BOTH, expand=True)

        

    # Action to open YOLOv8 Application in the main content area
    def sidebar_action_1(self):
        # Clear previous content and load YOLOv8 application
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        yolov8_app = YOLOv8App(self.content_frame)

    # Action to open ResNet50 Application in the main content area
    def sidebar_action_2(self):
        # Clear previous content and load ResNet50 application
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        resnet50_app = ResNet50App(self.content_frame)

# Main application execute
if __name__ == "__main__":
    root = Tk()
    app = TkinterApp(root)
    root.mainloop()
