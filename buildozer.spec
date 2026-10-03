[app]
title = Монетки
package.name = coincropper
package.domain = org.coincropper

source.dir = .
source.include_exts = py,png,jpg,kv,atlas,onnx
version = 0.1

requirements = python3,kivy==2.3.0,pyjnius,android,numpy,pillow

orientation = portrait
fullscreen = 0

android.permissions = WRITE_EXTERNAL_STORAGE
android.api = 33
android.minapi = 26
android.ndk = 25b
android.archs = arm64-v8a
android.accept_sdk_license = True
android.enable_androidx = True
android.gradle_dependencies = com.microsoft.onnxruntime:onnxruntime-android:1.17.1

[buildozer]
log_level = 2
warn_on_root = 1
