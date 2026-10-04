import io
import os
import tempfile

import numpy as np
from PIL import Image as PILImage, ImageOps

from kivy.app import App
from kivy.clock import Clock
from kivy.metrics import dp, sp
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
PADDING_PERCENT = 0

PythonActivity = autoclass("org.kivy.android.PythonActivity")
Intent = autoclass("android.content.Intent")
MediaMedia = autoclass("android.provider.MediaStore$Images$Media")
ContentValues = autoclass("android.content.ContentValues")
BuildVersion = autoclass("android.os.Build$VERSION")
JInteger = autoclass("java.lang.Integer")
ByteBuffer = autoclass("java.nio.ByteBuffer")
ByteOrder = autoclass("java.nio.ByteOrder")
HashMap = autoclass("java.util.HashMap")
OrtEnvironment = autoclass("ai.onnxruntime.OrtEnvironment")
SessionOptions = autoclass("ai.onnxruntime.OrtSession$SessionOptions")
OnnxTensor = autoclass("ai.onnxruntime.OnnxTensor")


class AndroidStorage:

    @staticmethod
    def resolver():
        return PythonActivity.mActivity.getContentResolver()

    @staticmethod
    def read_uri(uri):
        stream = AndroidStorage.resolver().openInputStream(uri)
        data = bytearray()
        buf = bytearray(1024 * 1024)
        while True:
            n = stream.read(buf)
            if n <= 0:
                break
            data.extend(buf[:n])
        stream.close()
        return bytes(data)

    @staticmethod
    def save_image(pil_image, folder_name, file_name):
        resolver = AndroidStorage.resolver()
        values = ContentValues()
        values.put("_display_name", file_name)
        values.put("mime_type", "image/jpeg")
        new_api = int(BuildVersion.SDK_INT) >= 29
        if new_api:
            values.put("relative_path", "Pictures/CoinCropper/" + folder_name)
            values.put("is_pending", JInteger(1))

        uri = resolver.insert(MediaMedia.EXTERNAL_CONTENT_URI, values)
        if uri is None:
            raise RuntimeError("Не вдалося створити файл у галереї")

        out = io.BytesIO()
        pil_image.save(out, format="JPEG", quality=95)
        stream = resolver.openOutputStream(uri)
        stream.write(out.getvalue())
        stream.close()

        if new_api:
            done = ContentValues()
            done.put("is_pending", JInteger(0))
            resolver.update(uri, done, None, None)
        return str(uri)


class CoinDetector:

    def __init__(self, model_path):
        self.last_info = ""
        self.env = OrtEnvironment.getEnvironment()
        self.session = self.env.createSession(model_path, SessionOptions())
        self.input_name = self.session.getInputNames().iterator().next()

    def predict(self, image):
        w, h = image.size
        scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        pad_x = (INPUT_SIZE - nw) // 2
        pad_y = (INPUT_SIZE - nh) // 2

        canvas = PILImage.new(
            "RGB", (INPUT_SIZE, INPUT_SIZE), (114, 114, 114)
        )
        canvas.paste(image.resize((nw, nh), PILImage.BILINEAR), (pad_x, pad_y))

        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        arr = np.ascontiguousarray(arr.transpose(2, 0, 1)[None])
        raw = arr.tobytes()

        buf = ByteBuffer.allocateDirect(len(raw)).order(
            ByteOrder.nativeOrder()
        )
        buf.put(raw)
        buf.rewind()

        tensor = OnnxTensor.createTensor(
            self.env, buf.asFloatBuffer(), [1, 3, INPUT_SIZE, INPUT_SIZE]
        )
        inputs = HashMap()
        inputs.put(self.input_name, tensor)

        result = self.session.run(inputs)
        try:
            value = cast("ai.onnxruntime.OnnxTensor", result.get(0)).getValue()
            output = np.array(value, dtype=np.float32)
        finally:
            tensor.close()
            result.close()

        return self.decode(output, image.size, scale, pad_x, pad_y)

    def decode(self, output, size, scale, pad_x, pad_y):
        ow, oh = size
        output = np.squeeze(output)
        if output.ndim != 2:
            self.last_info = f"незвична форма виходу {output.shape}"
            return []
        if output.shape[0] == 5:
            output = output.T
        elif output.shape[1] != 5:
            self.last_info = f"незвична форма виходу {output.shape}"
            return []

        self.last_info = f"макс.впевн. {float(output[:, 4].max()):.2f}"
        output = output[output[:, 4] >= CONF_THRESHOLD]
        detections = []
        for x, y, bw, bh, conf in output:
            x1 = (x - bw / 2 - pad_x) / scale
            x2 = (x + bw / 2 - pad_x) / scale
            y1 = (y - bh / 2 - pad_y) / scale
            y2 = (y + bh / 2 - pad_y) / scale
            x1, x2 = max(0, min(ow, x1)), max(0, min(ow, x2))
            y1, y2 = max(0, min(oh, y1)), max(0, min(oh, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({
                "box": (int(x1), int(y1), int(x2), int(y2)),
                "confidence": float(conf),
                "area": (x2 - x1) * (y2 - y1),
            })
        return detections


def make_label(text, size=20, height=45):
    return Label(
        text=text, font_size=sp(size * 1.3),
        size_hint_y=None, height=dp(height * 1.3),
    )


def make_wrap_label(text, size=16, height=90):
    lab = make_label(text, size, height)
    lab.halign = "center"
    lab.valign = "middle"
    lab.bind(width=lambda inst, w: setattr(inst, "text_size", (w, None)))
    return lab


def make_button(text, callback, size=20, height=60, **kw):
    b = Button(
        text=text, font_size=sp(size * 1.3),
        size_hint_y=None, height=dp(height * 1.3), **kw
    )
    b.bind(on_press=callback)
    return b


def make_scroll_grid():
    sv = ScrollView()
    grid = GridLayout(cols=3, spacing=dp(8), padding=dp(8), size_hint_y=None)
    grid.bind(minimum_height=grid.setter("height"))
    sv.add_widget(grid)
    return sv, grid


def add_thumb(grid, path, on_delete=None):
    box = BoxLayout(
        orientation="vertical", size_hint=(None, None),
        size=(dp(105), dp(145) if on_delete else dp(110)), spacing=dp(4),
    )
    box.add_widget(Image(
        source=path, size_hint=(None, None), size=(dp(105), dp(105))
    ))
    if on_delete:
        box.add_widget(
            make_button("Видалити", lambda i: on_delete(), 12, 26)
        )
    grid.add_widget(box)


class CoinCropperApp(App):

    def build(self):
        self.selected_uris = []
        self.thumb_cache = {}
        self.items = []
        self.result_thumbs = []
        self.success_count = 0
        self.failed_count = 0
        self.processing_index = 0
        self.model = None
        self.temp_folder = tempfile.mkdtemp(prefix="coin_crop_")

        self.main = BoxLayout(orientation="vertical")
        activity.bind(on_activity_result=self.on_activity_result)
        self.show_main_menu()
        return self.main

    def save_temp(self, pil_image, name, max_side=None):
        if max_side:
            pil_image = pil_image.copy()
            pil_image.thumbnail((max_side, max_side))
        path = os.path.join(self.temp_folder, name)
        pil_image.save(path, "JPEG", quality=90)
        return path

    # ---------------- menu ----------------
    def show_main_menu(self, instance=None):
        self.main.clear_widgets()
        lay = BoxLayout(orientation="vertical", padding=dp(24), spacing=dp(16))
        lay.add_widget(make_label("Монетки", 30, 90))
        lay.add_widget(make_button("Обрізання", self.show_crop_screen, 22, 80))
        lay.add_widget(
            make_button("Пошук монети", self.show_search_screen, 22, 80)
        )
        self.main.add_widget(lay)

    def show_search_screen(self, instance=None):
        self.main.clear_widgets()
        lay = BoxLayout(orientation="vertical", padding=dp(16), spacing=dp(16))
        lay.add_widget(make_label("Пошук монети", 26, 60))
        lay.add_widget(Label(
            text="Функція пошуку монети буде додана пізніше.",
            font_size=sp(22),
        ))
        lay.add_widget(make_button("Головне меню", self.show_main_menu, 20))
        self.main.add_widget(lay)

    # ---------------- crop screen ----------------
    def show_crop_screen(self, instance=None):
        self.main.clear_widgets()
        screen = BoxLayout(orientation="vertical", padding=dp(8), spacing=dp(8))

        tabs = BoxLayout(size_hint_y=None, height=dp(75), spacing=dp(4))
        self.tab_select = Button(text="1. Вибір", font_size=sp(20))
        self.tab_process = Button(
            text="2. Обрізання", font_size=sp(20), disabled=True
        )
        self.tab_result = Button(
            text="3. Результат", font_size=sp(20), disabled=True
        )
        self.tab_select.bind(on_press=lambda i: self.build_select_tab())
        self.tab_process.bind(on_press=lambda i: self.build_process_tab())
        self.tab_result.bind(on_press=lambda i: self.build_result_tab())
        for t in (self.tab_select, self.tab_process, self.tab_result):
            tabs.add_widget(t)
        screen.add_widget(tabs)

        self.content = BoxLayout(orientation="vertical", spacing=dp(8))
        screen.add_widget(self.content)
        screen.add_widget(
            make_button("Головне меню", self.show_main_menu, 18, 50)
        )
        self.main.add_widget(screen)
        self.build_select_tab()

    # ---------------- select tab ----------------
    def build_select_tab(self):
        self.content.clear_widgets()
        self.content.add_widget(make_label("Виберіть фотографії", 22, 50))

        row = BoxLayout(size_hint_y=None, height=dp(80), spacing=dp(8))
        row.add_widget(Button(
            text="Додати фото", font_size=sp(24), on_press=self.select_photos
        ))
        row.add_widget(Button(
            text="Очистити все", font_size=sp(24),
            disabled=not self.selected_uris,
            on_press=self.clear_selected,
        ))
        self.content.add_widget(row)

        text = (
            f"Вибрано фотографій: {len(self.selected_uris)}"
            if self.selected_uris else "Фото ще не вибрано"
        )
        self.content.add_widget(make_label(text, 16))

        sv, grid = make_scroll_grid()
        self.content.add_widget(sv)

        for uri in self.selected_uris:
            key = str(uri)
            try:
                if key not in self.thumb_cache:
                    img = ImageOps.exif_transpose(
                        PILImage.open(io.BytesIO(AndroidStorage.read_uri(uri)))
                    ).convert("RGB")
                    name = f"thumb_{len(self.thumb_cache)}.jpg"
                    self.thumb_cache[key] = self.save_temp(img, name, 300)
                add_thumb(
                    grid, self.thumb_cache[key],
                    lambda u=uri: self.remove_uri(u),
                )
            except Exception as e:
                print("Preview error:", e)

        if self.selected_uris:
            self.content.add_widget(
                make_button("Обрізати", self.start_processing, 22, 70)
            )

    def select_photos(self, instance):
        intent = Intent(Intent.ACTION_OPEN_DOCUMENT)
        intent.addCategory(Intent.CATEGORY_OPENABLE)
        intent.setType("image/*")
        intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, True)
        PythonActivity.mActivity.startActivityForResult(intent, 1001)

    def on_activity_result(self, request_code, result_code, intent):
        if request_code != 1001 or intent is None:
            return
        if result_code != autoclass("android.app.Activity").RESULT_OK:
            return

        new_uris = []
        clip = intent.getClipData()
        if clip is not None:
            for i in range(clip.getItemCount()):
                new_uris.append(clip.getItemAt(i).getUri())
        else:
            uri = intent.getData()
            if uri is not None:
                new_uris.append(uri)

        known = {str(u) for u in self.selected_uris}
        for uri in new_uris:
            if str(uri) not in known:
                self.selected_uris.append(uri)
                known.add(str(uri))
        Clock.schedule_once(lambda dt: self.build_select_tab(), 0)

    def remove_uri(self, uri):
        self.selected_uris = [
            u for u in self.selected_uris if str(u) != str(uri)
        ]
        self.build_select_tab()

    def clear_selected(self, instance):
        self.selected_uris = []
        self.build_select_tab()

    # ---------------- processing ----------------
    def start_processing(self, instance):
        if not self.selected_uris:
            return
        self.items = []
        self.result_thumbs = []
        self.success_count = 0
        self.failed_count = 0
        self.processing_index = 0
        self.tab_select.disabled = True
        self.tab_process.disabled = False
        self.tab_result.disabled = True
        self.build_process_tab()
        Clock.schedule_once(self.load_model, 0.2)

    def build_process_tab(self):
        self.content.clear_widgets()
        self.content.add_widget(make_label("Обрізання фотографій", 22, 50))
        self.progress_label = make_label("Підготовка...", 18, 40)
        self.content.add_widget(self.progress_label)
        self.count_label = make_label("Знайдено: 0    Не знайдено: 0", 16, 40)
        self.content.add_widget(self.count_label)
        self.debug_label = make_wrap_label("", 13, 40)
        self.content.add_widget(self.debug_label)
        self.error_label = make_wrap_label("", 13, 80)
        self.content.add_widget(self.error_label)
        sv, self.process_grid = make_scroll_grid()
        self.content.add_widget(sv)

    def load_model(self, dt):
        try:
            path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), MODEL_NAME
            )
            self.model = CoinDetector(path)
            self.progress_label.text = "Починаю обробку..."
            Clock.schedule_once(self.process_next, 0.1)
        except Exception as e:
            self.error_label.text = "Помилка моделі: " + str(e)[:300]
            self.tab_select.disabled = False

    def process_next(self, dt):
        total = len(self.selected_uris)
        if self.processing_index >= total:
            self.finish()
            return

        index = self.processing_index
        uri = self.selected_uris[index]
        self.progress_label.text = f"Оброблено {index}/{total}"
        original = None

        try:
            original = ImageOps.exif_transpose(
                PILImage.open(io.BytesIO(AndroidStorage.read_uri(uri)))
            ).convert("RGB")

            detections = self.model.predict(original)
            self.debug_label.text = "Модель: " + self.model.last_info

            if not detections:
                self.add_failed(original, uri, index)
            else:
                best = max(detections, key=lambda d: d["area"])
                x1, y1, x2, y2 = best["box"]
                r = max(x2 - x1, y2 - y1) / 2.0
                pad = r * PADDING_PERCENT / 100.0
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                box = (
                    int(max(0, cx - r - pad)),
                    int(max(0, cy - r - pad)),
                    int(min(original.width, cx + r + pad)),
                    int(min(original.height, cy + r + pad)),
                )
                cropped = original.crop(box)
                name = f"coin_{index + 1}.jpg"
                self.items.append({
                    "folder": "Обрізані", "name": name,
                    "image": cropped, "uri": None,
                })
                path = self.save_temp(cropped, name, 300)
                self.result_thumbs.append(path)
                self.success_count += 1
                add_thumb(self.process_grid, path)

        except Exception as e:
            self.error_label.text = f"Помилка (фото {index + 1}): {str(e)[:300]}"
            self.failed_count += 1
            if original is not None:
                self.items.append({
                    "folder": "NotDetected",
                    "name": f"error_{index + 1}.jpg",
                    "image": None, "uri": uri,
                })

        self.processing_index += 1
        self.count_label.text = (
            f"Знайдено: {self.success_count}    "
            f"Не знайдено: {self.failed_count}"
        )
        Clock.schedule_once(self.process_next, 0.05)

    def add_failed(self, original, uri, index):
        self.failed_count += 1
        name = f"not_detected_{index + 1}.jpg"
        self.items.append({
            "folder": "NotDetected", "name": name, "image": None, "uri": uri,
        })
        path = self.save_temp(original, name, 300)
        self.result_thumbs.append(path)
        add_thumb(self.process_grid, path)

    def finish(self):
        self.progress_label.text = "Обробку завершено!"
        self.tab_select.disabled = False
        self.tab_result.disabled = False
        self.build_result_tab()

    # ---------------- result tab + saving ----------------
    def build_result_tab(self):
        if self.processing_index < len(self.selected_uris):
            return
        self.content.clear_widgets()
        self.content.add_widget(make_label("Результат", 22, 50))
        self.content.add_widget(make_label(
            f"Знайдено монет: {self.success_count}    "
            f"Не знайдено: {self.failed_count}", 17, 45,
        ))
        self.save_btn = make_button(
            "Зберегти в Галерею", self.start_save, 22, 70
        )
        self.content.add_widget(self.save_btn)
        self.save_label = make_wrap_label(
            "Натисніть кнопку, щоб зберегти фото в Галерею.", 15, 70
        )
        self.content.add_widget(self.save_label)
        sv, grid = make_scroll_grid()
        self.content.add_widget(sv)
        for path in self.result_thumbs:
            add_thumb(grid, path)

    def start_save(self, instance):
        self.save_index = 0
        self.saved_ok = 0
        self.save_errors = []
        self.save_btn.disabled = True
        Clock.schedule_once(self.save_next, 0.05)

    def save_next(self, dt):
        total = len(self.items)
        if self.save_index >= total:
            msg = f"Збережено: {self.saved_ok}/{total}"
            if self.save_errors:
                msg += "\nПомилка: " + self.save_errors[-1]
            else:
                msg += "\nГалерея → альбом CoinCropper (Pictures/CoinCropper)"
            self.save_label.text = msg
            self.save_btn.disabled = False
            return

        item = self.items[self.save_index]
        try:
            img = item["image"]
            if img is None:
                img = ImageOps.exif_transpose(
                    PILImage.open(
                        io.BytesIO(AndroidStorage.read_uri(item["uri"]))
                    )
                ).convert("RGB")
            AndroidStorage.save_image(img, item["folder"], item["name"])
            self.saved_ok += 1
        except Exception as e:
            self.save_errors.append(str(e)[:200])

        self.save_index += 1
        self.save_label.text = f"Збереження {self.save_index}/{total}..."
        Clock.schedule_once(self.save_next, 0.02)


if __name__ == "__main__":
    CoinCropperApp().run()
