package com.example.superres

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

object SuperResEngine {

    suspend fun enhance(
        context: Context,
        bitmap: Bitmap,
        scale: Int,
        useGpu: Boolean,
        customModelUri: Uri?,
        customModelName: String?,
        onProgress: (Int) -> Unit
    ): Bitmap = withContext(Dispatchers.IO) {
        val targetScale = scale.coerceIn(1, 16)

        val upscaled = if (customModelUri == null) {
            onProgress(10)
            progressiveScale(bitmap, targetScale, onProgress)
        } else {
            runTfLite(context, bitmap, targetScale, useGpu, customModelUri, customModelName ?: "custom.tflite", onProgress)
        }

        onProgress(88)
        enhanceClarity(upscaled).also {
            onProgress(98)
        }
    }

    private fun progressiveScale(src: Bitmap, targetScale: Int, onProgress: (Int) -> Unit): Bitmap {
        var current = src
        var currentScale = 1
        while (currentScale < targetScale) {
            val nextScale = (currentScale * 2).coerceAtMost(targetScale)
            val w = (current.width * 2).coerceAtLeast(1)
            val h = (current.height * 2).coerceAtLeast(1)
            current = Bitmap.createScaledBitmap(current, w, h, true)
            currentScale = nextScale
            onProgress(10 + (70 * currentScale / targetScale))
        }
        return current
    }

    private fun enhanceClarity(bitmap: Bitmap): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        val paint = Paint().apply {
            isAntiAlias = true
            isFilterBitmap = true
        }
        val cm = ColorMatrix().apply {
            setSaturation(1.15f)
        }
        paint.colorFilter = ColorMatrixColorFilter(cm)
        canvas.drawBitmap(bitmap, 0f, 0f, paint)

        val pixels = IntArray(w * h)
        out.getPixels(pixels, 0, w, 0, 0, w, h)
        val sharp = pixels.copyOf()
        val k = 0.22f

        for (y in 0 until h) {
            for (x in 0 until w) {
                val i = y * w + x
                val c = pixels[i]

                val tl = pixels[(y - 1).coerceAtLeast(0) * w + (x - 1).coerceAtLeast(0)]
                val t = pixels[(y - 1).coerceAtLeast(0) * w + x]
                val tr = pixels[(y - 1).coerceAtLeast(0) * w + (x + 1).coerceAtMost(w - 1)]
                val l = pixels[y * w + (x - 1).coerceAtLeast(0)]
                val r = pixels[y * w + (x + 1).coerceAtMost(w - 1)]
                val bl = pixels[(y + 1).coerceAtMost(h - 1) * w + (x - 1).coerceAtLeast(0)]
                val b = pixels[(y + 1).coerceAtMost(h - 1) * w + x]
                val br = pixels[(y + 1).coerceAtMost(h - 1) * w + (x + 1).coerceAtMost(w - 1)]

                val nr = sharpenChannel(Color.red(c), Color.red(tl), Color.red(t), Color.red(tr), Color.red(l), Color.red(r), Color.red(bl), Color.red(b), Color.red(br), k)
                val ng = sharpenChannel(Color.green(c), Color.green(tl), Color.green(t), Color.green(tr), Color.green(l), Color.green(r), Color.green(bl), Color.green(b), Color.green(br), k)
                val nb = sharpenChannel(Color.blue(c), Color.blue(tl), Color.blue(t), Color.blue(tr), Color.blue(l), Color.blue(r), Color.blue(bl), Color.blue(b), Color.blue(br), k)
                sharp[i] = Color.rgb(nr, ng, nb)
            }
        }

        out.setPixels(sharp, 0, w, 0, 0, w, h)
        return out
    }

    private fun sharpenChannel(c: Int, tl: Int, t: Int, tr: Int, l: Int, r: Int, bl: Int, b: Int, br: Int, k: Float): Int {
        val neighbors = tl + t + tr + l + r + bl + b + br
        val v = c + (k * (8 * c - neighbors)).toInt()
        return v.coerceIn(0, 255)
    }

    private fun runTfLite(
        context: Context,
        bitmap: Bitmap,
        targetScale: Int,
        useGpu: Boolean,
        modelUri: Uri,
        modelName: String,
        onProgress: (Int) -> Unit
    ): Bitmap {
        val modelFile = File(context.cacheDir, modelName)
        context.contentResolver.openInputStream(modelUri).use { input ->
            requireNotNull(input) { "无法读取模型文件" }
            modelFile.outputStream().use { output -> input.copyTo(output) }
        }

        var delegate: GpuDelegate? = null
        var interpreter: Interpreter? = null
        return try {
            val options = Interpreter.Options()
            if (useGpu) {
                try {
                    delegate = GpuDelegate()
                    options.addDelegate(delegate)
                } catch (e: Exception) {
                    delegate?.close()
                    delegate = null
                }
            }

            interpreter = Interpreter(modelFile, options)
            val inputTensor = interpreter.getInputTensor(0)
            val outputTensor = interpreter.getOutputTensor(0)
            val inputShape = inputTensor.shape()
            val outputShape = outputTensor.shape()

            val inH = if (inputShape.size >= 2) inputShape[1] else bitmap.height
            val inW = if (inputShape.size >= 3) inputShape[2] else bitmap.width

            onProgress(20)

            val modelOutH = if (outputShape.size >= 2) outputShape[1] else inH * 4
            val modelOutW = if (outputShape.size >= 3) outputShape[2] else inW * 4
            val modelScale = (modelOutH / inH).coerceAtLeast(1)

            val inputBuffer = bitmapToFloatBuffer(bitmap, inW, inH)
            val outputBuffer = ByteBuffer.allocateDirect(outputTensor.numBytes()).order(ByteOrder.nativeOrder())

            interpreter.run(inputBuffer, outputBuffer)
            onProgress(60)

            var result = floatBufferToBitmap(outputBuffer, modelOutW, modelOutH)

            if (modelScale < targetScale) {
                result = progressiveScale(result, targetScale / modelScale) { pct ->
                    onProgress(60 + (25 * pct / 100))
                }
            }
            onProgress(85)
            result
        } finally {
            try { interpreter?.close() } catch (_: Exception) {}
            try { delegate?.close() } catch (_: Exception) {}
            if (modelFile.exists()) modelFile.delete()
        }
    }

    private fun bitmapToFloatBuffer(bitmap: Bitmap, width: Int, height: Int): ByteBuffer {
        val scaled = if (bitmap.width != width || bitmap.height != height) {
            Bitmap.createScaledBitmap(bitmap, width, height, true)
        } else bitmap

        val buffer = ByteBuffer.allocateDirect(width * height * 3 * 4).order(ByteOrder.nativeOrder())
        val pixels = IntArray(width * height)
        scaled.getPixels(pixels, 0, width, 0, 0, width, height)
        for (px in pixels) {
            buffer.putFloat(((px shr 16 and 0xFF) / 255f))
            buffer.putFloat(((px shr 8 and 0xFF) / 255f))
            buffer.putFloat(((px and 0xFF) / 255f))
        }
        buffer.rewind()
        return buffer
    }

    private fun floatBufferToBitmap(buffer: ByteBuffer, width: Int, height: Int): Bitmap {
        buffer.rewind()
        val pixels = IntArray(width * height)
        for (i in pixels.indices) {
            val r = (buffer.float * 255f).toInt().coerceIn(0, 255)
            val g = (buffer.float * 255f).toInt().coerceIn(0, 255)
            val b = (buffer.float * 255f).toInt().coerceIn(0, 255)
            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }
}
