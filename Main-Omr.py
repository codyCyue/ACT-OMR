import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
import numpy as np


class OMRUtils:
    @staticmethod
    def preprocess_image(img, target_width, target_height):
        """Resize and preprocess the image to fit the target dimensions."""
        img_h, img_w = img.shape[:2]
        aspect_ratio = img_w / img_h

        # Resize image while maintaining aspect ratio
        if aspect_ratio > target_width / target_height:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)

        img_resized = cv2.resize(img, (new_width, new_height))
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
        img_canny = cv2.Canny(img_blur, 50, 150)

        return img_resized, img_gray, img_canny

    @staticmethod
    def find_largest_contour(img_canny):
        """Find the largest rectangular contour in the image."""
        contours, _ = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = None
        max_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum size to be considered
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if len(approx) == 4 and area > max_area:
                    largest_contour = approx
                    max_area = area

        return largest_contour

    @staticmethod
    def warp_perspective(img, points, target_width, target_height):
        """Warp the perspective of the image based on four points."""
        points = OMRUtils.reorder_points(points)
        dst = np.float32([[0, 0], [target_width, 0], [0, target_height], [target_width, target_height]])
        matrix = cv2.getPerspectiveTransform(points, dst)
        return cv2.warpPerspective(img, matrix, (target_width, target_height))

    @staticmethod
    def reorder_points(points):
        """Reorder points to ensure consistent perspective."""
        points = points.reshape((4, 2))
        new_points = np.zeros((4, 2), dtype=np.float32)

        # Calculate sum and difference of points
        add = points.sum(axis=1)
        diff = np.diff(points, axis=1)

        # Assign corners based on their sum and difference
        new_points[0] = points[np.argmin(add)]  # Top-left
        new_points[3] = points[np.argmax(add)]  # Bottom-right
        new_points[1] = points[np.argmin(diff)]  # Top-right
        new_points[2] = points[np.argmax(diff)]  # Bottom-left

        return new_points

    @staticmethod
    def adaptive_threshold(img):
        """Apply adaptive thresholding for better binarization."""
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    @staticmethod
    def split_boxes(img, num_questions, num_choices):
        """Split the warped image into individual answer boxes."""
        height, width = img.shape
        row_height = height // num_questions
        col_width = width // num_choices

        boxes = []
        for row in range(num_questions):
            row_boxes = []
            for col in range(num_choices):
                # Extract each box
                x_start = col * col_width
                y_start = row * row_height
                x_end = x_start + col_width
                y_end = y_start + row_height
                box = img[y_start:y_end, x_start:x_end]
                row_boxes.append(box)
            boxes.append(row_boxes)

        return boxes

    @staticmethod
    def grade_answers(boxes, correct_answers):
        """Grade the answers based on detected boxes."""
        user_answers = []
        grading = []
        for question_boxes in boxes:
            pixels = [cv2.countNonZero(box) for box in question_boxes]
            selected_answer = np.argmax(pixels)
            user_answers.append(selected_answer)
            grading.append(selected_answer == correct_answers[len(user_answers) - 1])

        score = sum(grading) / len(correct_answers) * 100
        return user_answers, score, grading


class OMRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OMR Test Checker")
        self.root.geometry("360x640")  # Simulate Android portrait mode resolution

        # Dynamic target resolution
        self.target_width = None
        self.target_height = None

        self.answer_key = [0, 1, 2, 1, 0]  # Default answer key
        self.num_questions = len(self.answer_key)
        self.num_choices = 5  # Default number of choices per question

        self.setup_ui()

    def setup_ui(self):
        """Setup the main UI."""
        tk.Label(self.root, text="OMR Test Checker", font=("Arial", 16)).pack(pady=20)

        # Answer Key Configuration
        self.answer_key_frame = tk.Frame(self.root)
        self.answer_key_frame.pack(pady=10)
        self.create_answer_key_buttons()

        # Scan and Evaluate Buttons
        tk.Button(self.root, text="Scan OMR Sheet", command=self.scan_omr).pack(pady=10)
        tk.Button(self.root, text="Evaluate", command=self.evaluate_answers).pack(pady=10)

        # Result Display
        self.result_label = tk.Label(self.root, text="", font=("Arial", 14))
        self.result_label.pack(pady=20)

    def create_answer_key_buttons(self):
        """Create buttons for setting the answer key."""
        for i in range(self.num_questions):
            frame = tk.Frame(self.answer_key_frame)
            frame.pack(pady=5)
            tk.Label(frame, text=f"Q{i + 1}:", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)

            for j, choice in enumerate("ABCDE"):
                btn = tk.Button(
                    frame,
                    text=choice,
                    width=5,
                    command=lambda q=i, c=j: self.set_answer_key(q, c)
                )
                btn.pack(side=tk.LEFT, padx=5)

    def set_answer_key(self, question, choice):
        """Set the correct answer for a specific question."""
        self.answer_key[question] = choice
        messagebox.showinfo("Answer Key Updated", f"Q{question + 1} correct answer set to {chr(65 + choice)}")

    def scan_omr(self):
        """Open an image file and process it for grading."""
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if not file_path:
            return

        img = cv2.imread(file_path)
        self.target_width = img.shape[1]  # Use the uploaded image's width
        self.target_height = img.shape[0]  # Use the uploaded image's height

        img_resized, img_gray, img_canny = OMRUtils.preprocess_image(img, self.target_width, self.target_height)

        largest_contour = OMRUtils.find_largest_contour(img_canny)
        if largest_contour is None:
            messagebox.showerror("Error", "OMR Sheet not detected.")
            return

        self.omr_sheet = OMRUtils.warp_perspective(img, largest_contour, self.target_width, self.target_height)

        # Apply adaptive thresholding for better binarization
        self.img_thresh = OMRUtils.adaptive_threshold(cv2.cvtColor(self.omr_sheet, cv2.COLOR_BGR2GRAY))

        # Split into boxes
        self.boxes = OMRUtils.split_boxes(self.img_thresh, self.num_questions, self.num_choices)

        messagebox.showinfo("Scan Complete", "OMR Sheet processed successfully.")

    def evaluate_answers(self):
        """Evaluate answers and display results."""
        if not hasattr(self, 'boxes'):
            messagebox.showinfo("Info", "Please scan an OMR sheet first.")
            return

        user_answers, score, grading = OMRUtils.grade_answers(self.boxes, self.answer_key)

        # Highlight answers on the OMR sheet
        img_result = self.omr_sheet.copy()
        h, w = self.omr_sheet.shape[:2]
        sec_h = h // self.num_questions
        sec_w = w // self.num_choices

        for q, (correct, user, is_correct) in enumerate(zip(self.answer_key, user_answers, grading)):
            user_center = ((user * sec_w) + sec_w // 2, (q * sec_h) + sec_h // 2)
            color = (0, 255, 0) if is_correct else (0, 0, 255)
            cv2.circle(img_result, user_center, 15, color, cv2.FILLED)

            if not is_correct:
                correct_center = ((correct * sec_w) + sec_w // 2, (q * sec_h) + sec_h // 2)
                cv2.circle(img_result, correct_center, 15, (255, 255, 0), 2)

        cv2.imshow("Graded OMR Sheet", img_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        self.result_label.config(
            text=f"Score: {score:.2f}%\nCorrect Answers: {sum(grading)}/{len(self.answer_key)}"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = OMRApp(root)
    root.mainloop()
