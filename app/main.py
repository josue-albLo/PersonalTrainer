from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Rectangle
import cv2
import mediapipe as mp
import numpy as np
from exercises.exercise_evaluation import (
    evaluate_squat,
    evaluate_bicep_curl,
    evaluate_jump,
    detect_exercise,
)

class MainApp(App):
    def build(self):
        # Crear la ventana principal con un fondo
        root = FloatLayout()
        with root.canvas.before:
            Color(0.1, 0.1, 0.1, 1)  # Fondo oscuro
            self.rect = Rectangle(size=root.size, pos=root.pos)
            root.bind(size=self._update_rect, pos=self._update_rect)

        # Crear la disposición principal
        main_layout = BoxLayout(orientation="vertical", padding=10, spacing=10)
        anchor_layout = AnchorLayout(anchor_x="center", anchor_y="top")
        self.img = Image(size_hint=(1, 0.8), allow_stretch=True)
        anchor_layout.add_widget(self.img)

        # Crear el layout para la información
        info_layout = BoxLayout(
            size_hint=(1, 0.2), orientation="horizontal", spacing=10, padding=[20, 10]
        )
        self.exercise_label = Label(
            text="Ejercicio: unknown", size_hint=(0.5, 1), color=(1, 1, 1, 1), font_size='20sp'
        )
        self.label = Label(
            text="Precision: 0.00%", size_hint=(0.5, 1), color=(1, 1, 1, 1), font_size='20sp'
        )
        info_layout.add_widget(self.exercise_label)
        info_layout.add_widget(self.label)

        main_layout.add_widget(anchor_layout)
        main_layout.add_widget(info_layout)
        root.add_widget(main_layout)

        # Configuración de la cámara
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        Clock.schedule_interval(self.update, 1.0 / 30.0)
        return root

    def _update_rect(self, instance, value):
        self.rect.size = instance.size
        self.rect.pos = instance.pos

    def update(self, dt):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame, precision, exercise = self.process_frame(frame)

        # Update labels with precision and exercise
        self.label.text = f"Precision: {precision:.2f}%"
        self.exercise_label.text = f"Ejercicio: {exercise}"

        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        image_texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt="bgr"
        )
        image_texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        self.img.texture = image_texture

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        precision = 0
        exercise = "unknown"
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
            )
            exercise = detect_exercise(results.pose_landmarks)
            if exercise == "squat":
                precision = evaluate_squat(results.pose_landmarks)
            elif exercise == "bicep_curl":
                precision = evaluate_bicep_curl(results.pose_landmarks)
            elif exercise == "jump":
                precision = evaluate_jump(results.pose_landmarks)

        return frame, precision, exercise

    def on_stop(self):
        self.cap.release()
        self.pose.close()


if __name__ == "__main__":
    MainApp().run()
