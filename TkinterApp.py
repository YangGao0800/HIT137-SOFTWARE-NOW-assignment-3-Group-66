from tkinter import *
from tkinter import filedialog
from ultralytics import YOLO
import cv2
from PIL import Image, ImageTk
import time

# Decorator to log execution time of methods
def log_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time for {func.__name__}: {end_time - start_time:.2f} seconds")
        return result
    return wrapper

# Base class to implement polymorphism
class AIModelBase:
    def process_image(self, image):
        raise NotImplementedError("Subclasses should implement this!")

# Mixin class for additional functionalities (Multiple Inheritance)
class ImageProcessingMixin:
    def preprocess_image(self, image):
        # Example preprocessing function
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# YOLOv8 class with multiple inheritance and method overriding
class YOLOv8App(AIModelBase, ImageProcessingMixin):
    def __init__(self, master, title_of_windows):
        self.master = master
        self.frame = Frame(master)
        #use fill=BOTH and expand=True to enlarge the image to fit the window when user maximizes the app window 
        self.frame.pack(fill=BOTH,expand=True) 

        master.title(title_of_windows)

        # Encapsulation: Private model attribute
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
            # bg="red", 
            # fg="white",
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
            # bg="red", 
            # fg="white",
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

    @log_execution_time
    def load_image(self):
        img_path = filedialog.askopenfilename() # a dialog box will appear and the user need to select an image to get image path from local directory
        if img_path:
            self.original_image = cv2.imread(img_path)  # use opencv to read image
            self.display_original_image()  # display original image in the left frame
            self.btn_detect_img.config(state=NORMAL)  # after user loads image, detect button is enabled 

    def display_original_image(self):
        # Convert OpenCV BGR image to RGB, image_in_rgb is a numpy array because the return of cv2.cvtcolor is a numpy array
        image_in_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        # image_in_rgb is a numpy array, thus, we should convert numpy array to PIL Image to display in the frame
        image_pil = Image.fromarray(image_in_rgb)
        
        # Resize and display image
        self.display_image(image_pil, self.canvas_original_img)
    
    @log_execution_time
    def detect_objects(self):  
        try:      #to check that the original image has been loaded before using the detection object. 
            self.original_image
            # detect original image by model Yolov8 and return annotated image
            annotated_image = self.process_image(self.original_image)
            
            # Convert OpenCV BGR image to RGB
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            image_pil = Image.fromarray(annotated_image_rgb)
            
            # Resize and display image
            self.display_image(image_pil, self.canvas_detected_img)
        except AttributeError:
            print("No image loaded")
    
    def display_image(self, image_pil, canvas):
        # use the following two methods to get width and height of canvas 
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # when canvas changes, use thumbnail to resize image in original ratio of width and height to fit canvas, 
        # the image is changed in place.
        image_pil.thumbnail((canvas_width, canvas_height))
        # image_pil = image_pil.resize((canvas_width, canvas_height))
        
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
        canvas.create_image(canvas_width//2, 
                            canvas_height//2, 
                            anchor=CENTER, 
                            image=image_tkinter_mode)

    # update images to fit new canvas size when window changes
    # when window changes, the event will change with new canvas width and height
    # and this method will be called
    def update_canvas_size(self, event=None ): 
        if hasattr(self, 'original_image'):
            # display the original image again with new canvas size
            self.display_original_image()  
        if hasattr(self, 'tk_image_detected'):
            # display the detected image again with new canvas size
            self.detect_objects()
        

    # Method overriding
    @log_execution_time
    def process_image(self, image):
        # Use Yolov8 model to detect objects
        results = self.__model(image)
        # to draw bounding boxes, labels and confidence on annotated images
        return results[0].plot()   

# Main app execution
root = Tk()
app = YOLOv8App(root, "YOLOv8 Object Detection on Image")
root.geometry("800x650")  # Set initial window size
root.mainloop()