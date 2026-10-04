import os
import io
import math
import threading
import tempfile

import numpy as np
from PIL import Image as PILImage

from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout

from jnius import autoclass, cast
from android import activity


MODEL_NAME = "best.onnx"

INPUT_SIZE = 640
CONF_THRESHOLD = 0.25

# 0 = без додаткового збільшення області обрізання
PADDING_PERCENT = 0


# ---------------------------------------------------------
# Android
# ---------------------------------------------------------

PythonActivity = autoclass("org.kivy.android.PythonActivity")
Intent = autoclass("android.content.Intent")
Uri = autoclass("android.net.Uri")
MediaStore = autoclass("android.provider.MediaStore")
ContentValues = autoclass("android.content.ContentValues")
BuildVersion = autoclass("android.os.Build$VERSION")


class AndroidStorage:

    @staticmethod
    def get_resolver():
        activity_instance = PythonActivity.mActivity
        return activity_instance.getContentResolver()

    @staticmethod
    def read_uri(uri):
        resolver = AndroidStorage.get_resolver()
        stream = resolver.openInputStream(uri)

        data = bytearray()

        buffer = bytearray(1024 * 1024)

        while True:
            read_count = stream.read(buffer)

            if read_count <= 0:
                break

            data.extend(buffer[:read_count])

        stream.close()

        return bytes(data)

    @staticmethod
    def save_image(pil_image, folder_name, file_name):

        resolver = AndroidStorage.get_resolver()

        values = ContentValues()

        values.put(
            MediaStore.Images.Media.DISPLAY_NAME,
            file_name
        )

        values.put(
            MediaStore.Images.Media.MIME_TYPE,
            "image/jpeg"
        )

        if int(BuildVersion.SDK_INT) >= 29:

            relative_path = (
                "Pictures/CoinCropper/"
                + folder_name
            )

            values.put(
                MediaStore.Images.Media.RELATIVE_PATH,
                relative_path
            )

            values.put(
                MediaStore.Images.Media.IS_PENDING,
                1
            )

        uri = resolver.insert(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            values
        )

        if uri is None:
            raise RuntimeError(
                "Не вдалося створити файл у галереї"
            )

        stream = resolver.openOutputStream(uri)

        output = io.BytesIO()

        pil_image.save(
            output,
            format="JPEG",
            quality=95
        )

        data = output.getvalue()

        stream.write(data)
        stream.close()

        if int(BuildVersion.SDK_INT) >= 29:

            values.clear()

            values.put(
                MediaStore.Images.Media.IS_PENDING,
                0
            )

            resolver.update(
                uri,
                values,
                None,
                None
            )

        return str(uri)


# ---------------------------------------------------------
# ONNX Runtime
# ---------------------------------------------------------

OrtEnvironment = autoclass(
    "ai.onnxruntime.OrtEnvironment"
)

OrtSession = autoclass(
    "ai.onnxruntime.OrtSession"
)

OnnxTensor = autoclass(
    "ai.onnxruntime.OnnxTensor"
)

JavaFloatBuffer = autoclass(
    "java.nio.FloatBuffer"
)


class CoinDetector:

    def __init__(self, model_path):

        self.env = OrtEnvironment.getEnvironment()

        self.session = self.env.createSession(
            model_path,
            OrtSession.SessionOptions()
        )

        self.input_name = (
            self.session.getInputNames()
            .toArray()[0]
        )

    # -----------------------------------------------------
    # Letterbox
    # -----------------------------------------------------

    def letterbox(self, image):

        width, height = image.size

        scale = min(
            INPUT_SIZE / width,
            INPUT_SIZE / height
        )

        new_width = int(round(width * scale))
        new_height = int(round(height * scale))

        resized = image.resize(
            (new_width, new_height),
            PILImage.Resampling.BILINEAR
        )

        canvas = PILImage.new(
            "RGB",
            (INPUT_SIZE, INPUT_SIZE),
            (114, 114, 114)
        )

        pad_x = (INPUT_SIZE - new_width) // 2
        pad_y = (INPUT_SIZE - new_height) // 2

        canvas.paste(
            resized,
            (pad_x, pad_y)
        )

        return canvas, scale, pad_x, pad_y

    # -----------------------------------------------------
    # Image → NCHW float32
    # -----------------------------------------------------

    def prepare_image(self, image):

        letterboxed, scale, pad_x, pad_y = (
            self.letterbox(image)
        )

        arr = np.asarray(
            letterboxed,
            dtype=np.float32
        )

        arr /= 255.0

        arr = np.transpose(
            arr,
            (2, 0, 1)
        )

        arr = np.expand_dims(
            arr,
            axis=0
        )

        return (
            letterboxed,
            arr,
            scale,
            pad_x,
            pad_y
        )

    # -----------------------------------------------------
    # ONNX
    # -----------------------------------------------------

    def predict(self, image):

        (
            prepared,
            input_array,
            scale,
            pad_x,
            pad_y
        ) = self.prepare_image(image)

        flat = input_array.astype(
            np.float32
        ).flatten()

        # Java float[]
        FloatArray = autoclass(
            "[F"
        )

        java_array = FloatArray(
            len(flat)
        )

        for i, value in enumerate(flat):
            java_array[i] = float(value)

        tensor = OnnxTensor.createTensor(
            self.env,
            java_array,
            [1, 3, INPUT_SIZE, INPUT_SIZE]
        )

        inputs = {
            self.input_name: tensor
        }

        result = self.session.run(
            inputs
        )

        output = result.get(0)

        # Java float[][] / float[][][]
        output = np.array(
            output,
            dtype=np.float32
        )

        tensor.close()
        result.close()

        return self.decode(
            output,
            image.size,
            scale,
            pad_x,
            pad_y
        )

    # -----------------------------------------------------
    # Decode [1, 5, 8400]
    #
    # x, y, w, h, confidence
    # -----------------------------------------------------

    def decode(
        self,
        output,
        original_size,
        scale,
        pad_x,
        pad_y
    ):

        output = np.squeeze(
            output
        )

        # Очікуємо:
        # [5, 8400]

        if output.ndim != 2:
            return []

        if output.shape[0] == 5:

            predictions = output.T

        elif output.shape[1] == 5:

            predictions = output

        else:

            return []

        original_width, original_height = (
            original_size
        )

        detections = []

        for prediction in predictions:

            x = float(prediction[0])
            y = float(prediction[1])
            w = float(prediction[2])
            h = float(prediction[3])
            confidence = float(prediction[4])

            if confidence < CONF_THRESHOLD:
                continue

            # YOLO output у форматі:
            # center_x, center_y, width, height

            x1 = x - w / 2
            y1 = y - h / 2

            x2 = x + w / 2
            y2 = y + h / 2

            # Забираємо padding letterbox
            x1 -= pad_x
            x2 -= pad_x

            y1 -= pad_y
            y2 -= pad_y

            # Повертаємо масштаб оригінального фото
            x1 /= scale
            x2 /= scale

            y1 /= scale
            y2 /= scale

            x1 = max(
                0,
                min(original_width, x1)
            )

            y1 = max(
                0,
                min(original_height, y1)
            )

            x2 = max(
                0,
                min(original_width, x2)
            )

            y2 = max(
                0,
                min(original_height, y2)
            )

            if x2 <= x1 or y2 <= y1:
                continue

            area = (
                x2 - x1
            ) * (
                y2 - y1
            )

            detections.append({
                "box": (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                ),
                "confidence": confidence,
                "area": area
            })

        return detections


# ---------------------------------------------------------
# Application
# ---------------------------------------------------------

class CoinCropperApp(App):

    def build(self):

        self.selected_uris = []

        self.processed_images = []
        self.not_detected_images = []

        self.success_count = 0
        self.failed_count = 0

        self.processing_index = 0

        self.temp_folder = tempfile.mkdtemp(
            prefix="coin_crop_"
        )

        self.model = None

        self.root = BoxLayout(
            orientation="vertical"
        )

        self._picker_callback = None

        activity.bind(
            on_activity_result=self.on_activity_result
        )

        self.show_main_menu()

        return self.root

    # -----------------------------------------------------
    # General UI
    # -----------------------------------------------------

    def clear_root(self):

        self.root.clear_widgets()

    def show_main_menu(self, instance=None):

        self.clear_root()

        layout = BoxLayout(
            orientation="vertical",
            padding=30,
            spacing=20
        )

        title = Label(
            text="Монетки",
            font_size=36,
            size_hint_y=None,
            height=100
        )

        layout.add_widget(title)

        crop_button = Button(
            text="Обрізання",
            font_size=25,
            size_hint_y=None,
            height=90
        )

        crop_button.bind(
            on_press=self.open_cropper
        )

        layout.add_widget(crop_button)

        search_button = Button(
            text="Пошук монети",
            font_size=25,
            size_hint_y=None,
            height=90
        )

        search_button.bind(
            on_press=self.open_search
        )

        layout.add_widget(search_button)

        self.root.add_widget(layout)

    def open_cropper(self, instance):

        self.show_crop_screen()

    def open_search(self, instance):

        self.show_search_screen()

    # -----------------------------------------------------
    # Search
    # -----------------------------------------------------

    def show_search_screen(self):

        self.clear_root()

        layout = BoxLayout(
            orientation="vertical",
            padding=20,
            spacing=20
        )

        title = Label(
            text="Пошук монети",
            font_size=30,
            size_hint_y=None,
            height=70
        )

        layout.add_widget(title)

        info = Label(
            text="Функція пошуку монети буде додана пізніше.",
            font_size=20
        )

        layout.add_widget(info)

        back = Button(
            text="Головне меню",
            font_size=20,
            size_hint_y=None,
            height=60
        )

        back.bind(
            on_press=self.show_main_menu
        )

        layout.add_widget(back)

        self.root.add_widget(layout)

    # -----------------------------------------------------
    # Crop screen
    # -----------------------------------------------------

    def show_crop_screen(self):

        self.clear_root()

        main = BoxLayout(
            orientation="vertical",
            padding=10,
            spacing=10
        )

        self.tabs = BoxLayout(
            size_hint_y=None,
            height=55,
            spacing=5
        )

        self.tab_select = Button(
            text="1. Вибір",
            font_size=18
        )

        self.tab_process = Button(
            text="2. Обрізання",
            font_size=18,
            disabled=True
        )

        self.tab_result = Button(
            text="3. Результат",
            font_size=18,
            disabled=True
        )

        self.tab_select.bind(
            on_press=self.show_select_tab
        )

        self.tab_process.bind(
            on_press=self.show_process_tab
        )

        self.tab_result.bind(
            on_press=self.show_result_tab
        )

        self.tabs.add_widget(self.tab_select)
        self.tabs.add_widget(self.tab_process)
        self.tabs.add_widget(self.tab_result)

        main.add_widget(self.tabs)

        self.content = BoxLayout(
            orientation="vertical",
            spacing=10
        )

        main.add_widget(self.content)

        back = Button(
            text="Головне меню",
            font_size=18,
            size_hint_y=None,
            height=50
        )

        back.bind(
            on_press=self.show_main_menu
        )

        main.add_widget(back)

        self.root.add_widget(main)

        self.build_select_tab()

    # -----------------------------------------------------
    # Select tab
    # -----------------------------------------------------

    def build_select_tab(self):

        self.clear_content()

        title = Label(
            text="Виберіть фотографії",
            font_size=26,
            size_hint_y=None,
            height=50
        )

        self.content.add_widget(title)

        buttons = BoxLayout(
            size_hint_y=None,
            height=60,
            spacing=10
        )

        select = Button(
            text="Додати фото",
            font_size=18
        )

        select.bind(
            on_press=self.select_photos
        )

        buttons.add_widget(select)

        clear = Button(
            text="Очистити все",
            font_size=18,
            disabled=not bool(
                self.selected_uris
            )
        )

        clear.bind(
            on_press=self.clear_selected_files
        )

        buttons.add_widget(clear)

        self.content.add_widget(buttons)

        label_text = (
            "Фото ще не вибрано"
            if not self.selected_uris
            else
            f"Вибрано фотографій: "
            f"{len(self.selected_uris)}"
        )

        self.selected_label = Label(
            text=label_text,
            font_size=18,
            size_hint_y=None,
            height=45
        )

        self.content.add_widget(
            self.selected_label
        )

        scroll = ScrollView()

        self.selected_grid = GridLayout(
            cols=3,
            spacing=10,
            padding=10,
            size_hint_y=None
        )

        self.selected_grid.bind(
            minimum_height=
            self.selected_grid.setter(
                "height"
            )
        )

        scroll.add_widget(
            self.selected_grid
        )

        self.content.add_widget(scroll)

        for uri in self.selected_uris:

            try:

                data = AndroidStorage.read_uri(
                    uri
                )

                path = self.bytes_to_temp(
                    data
                )

                self.add_selectable_thumbnail(
                    self.selected_grid,
                    path,
                    uri
                )

            except Exception as e:

                print(
                    "Preview error:",
                    e
                )

        if self.selected_uris:

            crop = Button(
                text="Обрізати",
                font_size=20,
                size_hint_y=None,
                height=60
            )

            crop.bind(
                on_press=self.start_processing
            )

            self.content.add_widget(crop)

    def clear_content(self):

        self.content.clear_widgets()

    # -----------------------------------------------------
    # Android file picker
    # -----------------------------------------------------

    def select_photos(self, instance):

        intent = Intent(
            Intent.ACTION_OPEN_DOCUMENT
        )

        intent.addCategory(
            Intent.CATEGORY_OPENABLE
        )

        intent.setType(
            "image/*"
        )

        intent.putExtra(
            Intent.EXTRA_ALLOW_MULTIPLE,
            True
        )

        self._picker_callback = "select"

        PythonActivity.mActivity.startActivityForResult(
            intent,
            1001
        )

    def on_activity_result(
        self,
        request_code,
        result_code,
        intent
    ):

        if request_code != 1001:
            return

        if intent is None:
            return

        ActivityResult = autoclass(
            "android.app.Activity"
        )

        if result_code != ActivityResult.RESULT_OK:
            return

        new_uris = []

        clip_data = intent.getClipData()

        if clip_data is not None:

            count = clip_data.getItemCount()

            for i in range(count):

                uri = clip_data.getItemAt(
                    i
                ).getUri()

                new_uris.append(uri)

        else:

            uri = intent.getData()

            if uri is not None:
                new_uris.append(uri)

        for uri in new_uris:

            uri_string = str(uri)

            exists = False

            for old_uri in self.selected_uris:

                if str(old_uri) == uri_string:
                    exists = True
                    break

            if not exists:
                self.selected_uris.append(uri)

        self.build_select_tab()

    # -----------------------------------------------------
    # Temp image
    # -----------------------------------------------------

    def bytes_to_temp(self, data):

        path = os.path.join(
            self.temp_folder,
            "image_"
            + str(len(os.listdir(
                self.temp_folder
            )))
            + ".jpg"
        )

        with open(
            path,
            "wb"
        ) as file:

            file.write(data)

        return path

    # -----------------------------------------------------
    # Thumbnails
    # -----------------------------------------------------

    def add_selectable_thumbnail(
        self,
        grid,
        path,
        uri
    ):

        box = BoxLayout(
            orientation="vertical",
            size_hint=(None, None),
            size=(150, 185),
            spacing=5
        )

        image = Image(
            source=path,
            size_hint=(None, None),
            size=(150, 150),
            allow_stretch=True,
            keep_ratio=True
        )

        box.add_widget(image)

        button = Button(
            text="Видалити",
            font_size=14,
            size_hint_y=None,
            height=30
        )

        button.bind(
            on_press=lambda instance,
            selected_uri=uri:
            self.remove_selected_uri(
                selected_uri
            )
        )

        box.add_widget(button)

        grid.add_widget(box)

    def add_result_thumbnail(
        self,
        grid,
        path,
        result_type
    ):

        box = BoxLayout(
            orientation="vertical",
            size_hint=(None, None),
            size=(150, 185),
            spacing=5
        )

        image = Image(
            source=path,
            size_hint=(None, None),
            size=(150, 150),
            allow_stretch=True,
            keep_ratio=True
        )

        box.add_widget(image)

        grid.add_widget(box)

    # -----------------------------------------------------
    # Remove selected
    # -----------------------------------------------------

    def remove_selected_uri(self, uri):

        self.selected_uris = [
            item
            for item in self.selected_uris
            if str(item) != str(uri)
        ]

        self.build_select_tab()

    def clear_selected_files(self, instance):

        self.selected_uris = []

        self.build_select_tab()

    # -----------------------------------------------------
    # Processing
    # -----------------------------------------------------

    def start_processing(self, instance):

        if not self.selected_uris:
            return

        self.processed_images = []
        self.not_detected_images = []

        self.success_count = 0
        self.failed_count = 0
        self.processing_index = 0

        self.tab_select.disabled = True
        self.tab_process.disabled = False
        self.tab_result.disabled = True

        self.build_process_tab()

        Clock.schedule_once(
            self.load_model,
            0.2
        )

    def build_process_tab(self):

        self.clear_content()

        title = Label(
            text="Обрізання фотографій",
            font_size=26,
            size_hint_y=None,
            height=50
        )

        self.content.add_widget(title)

        self.progress_label = Label(
            text="Підготовка...",
            font_size=20,
            size_hint_y=None,
            height=45
        )

        self.content.add_widget(
            self.progress_label
        )

        self.result_count_label = Label(
            text="Знайдено: 0    Не знайдено: 0",
            font_size=18,
            size_hint_y=None,
            height=40
        )

        self.content.add_widget(
            self.result_count_label
        )

        scroll = ScrollView()

        self.process_grid = GridLayout(
            cols=3,
            spacing=10,
            padding=10,
            size_hint_y=None
        )

        self.process_grid.bind(
            minimum_height=
            self.process_grid.setter(
                "height"
            )
        )

        scroll.add_widget(
            self.process_grid
        )

        self.content.add_widget(scroll)

    def load_model(self, dt):

        try:

            model_path = os.path.join(
                self.user_data_dir,
                MODEL_NAME
            )

            self.model = CoinDetector(
                model_path
            )

            self.progress_label.text = (
                "Починаю обробку..."
            )

            Clock.schedule_once(
                self.process_next,
                0.1
            )

        except Exception as e:

            self.progress_label.text = (
                "Помилка моделі: "
                + str(e)
            )

            print(
                "MODEL ERROR:",
                e
            )

            self.tab_select.disabled = False

    def process_next(self, dt):

        if (
            self.processing_index
            >= len(self.selected_uris)
        ):

            self.processing_finished()
            return

        index = self.processing_index

        uri = self.selected_uris[index]

        total = len(
            self.selected_uris
        )

        self.progress_label.text = (
            f"Оброблено "
            f"{index}/{total}"
        )

        try:

            data = AndroidStorage.read_uri(
                uri
            )

            original = PILImage.open(
                io.BytesIO(data)
            ).convert("RGB")

            detections = self.model.predict(
                original
            )

            if not detections:

                self.failed_count += 1

                file_name = (
                    f"not_detected_"
                    f"{index + 1}.jpg"
                )

                temp_path = os.path.join(
                    self.temp_folder,
                    file_name
                )

                original.save(
                    temp_path,
                    "JPEG",
                    quality=95
                )

                AndroidStorage.save_image(
                    original,
                    "NotDetected",
                    file_name
                )

                self.not_detected_images.append(
                    temp_path
                )

                self.add_result_thumbnail(
                    self.process_grid,
                    temp_path,
                    "failed"
                )

            else:

                best = max(
                    detections,
                    key=lambda x: x["area"]
                )

                x1, y1, x2, y2 = (
                    best["box"]
                )

                width = x2 - x1
                height = y2 - y1

                r = max(
                    width,
                    height
                ) / 2.0

                padding = r * (
                    PADDING_PERCENT / 100.0
                )

                cx = (
                    x1 + x2
                ) / 2.0

                cy = (
                    y1 + y2
                ) / 2.0

                crop_x1 = int(
                    max(
                        0,
                        cx - r - padding
                    )
                )

                crop_y1 = int(
                    max(
                        0,
                        cy - r - padding
                    )
                )

                crop_x2 = int(
                    min(
                        original.width,
                        cx + r + padding
                    )
                )

                crop_y2 = int(
                    min(
                        original.height,
                        cy + r + padding
                    )
                )

                cropped = original.crop(
                    (
                        crop_x1,
                        crop_y1,
                        crop_x2,
                        crop_y2
                    )
                )

                self.success_count += 1

                file_name = (
                    f"coin_"
                    f"{index + 1}.jpg"
                )

                temp_path = os.path.join(
                    self.temp_folder,
                    file_name
                )

                cropped.save(
                    temp_path,
                    "JPEG",
                    quality=95
                )

                AndroidStorage.save_image(
                    cropped,
                    "Обрізані",
                    file_name
                )

                self.processed_images.append(
                    temp_path
                )

                self.add_result_thumbnail(
                    self.process_grid,
                    temp_path,
                    "success"
                )

        except Exception as e:

            print(
                "PROCESS ERROR:",
                e
            )

            self.failed_count += 1

            try:

                file_name = (
                    f"error_"
                    f"{index + 1}.jpg"
                )

                temp_path = os.path.join(
                    self.temp_folder,
                    file_name
                )

                original.save(
                    temp_path,
                    "JPEG",
                    quality=95
                )

                AndroidStorage.save_image(
                    original,
                    "NotDetected",
                    file_name
                )

                self.not_detected_images.append(
                    temp_path
                )

                self.add_result_thumbnail(
                    self.process_grid,
                    temp_path,
                    "failed"
                )

            except Exception:
                pass

        self.processing_index += 1

        self.result_count_label.text = (
            f"Знайдено: "
            f"{self.success_count}    "
            f"Не знайдено: "
            f"{self.failed_count}"
        )

        Clock.schedule_once(
            self.process_next,
            0.05
        )

    # -----------------------------------------------------
    # Finished
    # -----------------------------------------------------

    def processing_finished(self):

        self.progress_label.text = (
            "Обробку завершено!"
        )

        self.result_count_label.text = (
            f"Знайдено: "
            f"{self.success_count}    "
            f"Не знайдено: "
            f"{self.failed_count}"
        )

        self.tab_select.disabled = False
        self.tab_process.disabled = False
        self.tab_result.disabled = False

        self.show_result_tab(None)

    # -----------------------------------------------------
    # Result
    # -----------------------------------------------------

    def build_result_tab(self):

        self.clear_content()

        title = Label(
            text="Результат",
            font_size=26,
            size_hint_y=None,
            height=50
        )

        self.content.add_widget(title)

        info = Label(
            text=(
                f"Знайдено монет: "
                f"{self.success_count}\n"
                f"Не знайдено: "
                f"{self.failed_count}\n\n"
                f"Файли збережено в Галерею."
            ),
            font_size=20,
            size_hint_y=None,
            height=110
        )

        self.content.add_widget(info)

        scroll = ScrollView()

        grid = GridLayout(
            cols=3,
            spacing=10,
            padding=10,
            size_hint_y=None
        )

        grid.bind(
            minimum_height=
            grid.setter("height")
        )

        scroll.add_widget(grid)

        self.content.add_widget(scroll)

        for path in self.processed_images:

            self.add_result_thumbnail(
                grid,
                path,
                "success"
            )

        for path in self.not_detected_images:

            self.add_result_thumbnail(
                grid,
                path,
                "failed"
            )

    # -----------------------------------------------------
    # Tabs
    # -----------------------------------------------------

    def show_select_tab(self, instance):

        self.build_select_tab()

    def show_process_tab(self, instance):

        if self.processing_index == 0:
            return

        self.build_process_tab()

    def show_result_tab(self, instance):

        if (
            self.processing_index
            < len(self.selected_uris)
        ):
            return

        self.build_result_tab()


if __name__ == "__main__":
    CoinCropperApp().run()
