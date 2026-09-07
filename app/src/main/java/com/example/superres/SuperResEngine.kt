package com.example.superres

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min

object SuperResEngine {

    private const val MAX_PIXELS = 18_000_000L

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

        val raw = if (customModelUri == null) {
            onProgress(5)
            highQualityScale(bitmap, targetScale, onProgress)
        } else {
            runTfLite(context, bitmap, targetScale, useGpu, customModelUri, customModelName ?: "custom.tflite", onProgress)
        }

        onProgress(86)
        if (raw.width.toLong() * raw.height.toLong() > 18_000_000L) raw else clarityEnhance(raw, onProgress)
    }

    private fun highQualityScale(src: Bitmap, targetScale: Int, onProgress: (Int) -> Unit): Bitmap {
        if (targetScale <= 1) return src.copy(Bitmap.Config.ARGB_8888, false)

        var result = src
        var currentScale = 1
        while (currentScale < targetScale) {
            val nextScale = min(targetScale, currentScale * 2)
            var w = src.width * nextScale
            var h = src.height * nextScale

            val pixels = w.toLong() * h.toLong()
            if (pixels > MAX_PIXELS) {
                val ratio = kotlin.math.sqrt(MAX_PIXELS.toDouble() / pixels)
                w = (w * ratio).toInt().coerceAtLeast(1)
                h = (h * ratio).toInt().coerceAtLeast(1)
            }

            result = Bitmap.createScaledBitmap(src, w, h, true)
            currentScale = nextScale
            onProgress(5 + (78 * currentScale / targetScale))
        }
        return result
    }

    private fun clarityEnhance(src: Bitmap, onProgress: (Int) -> Unit): Bitmap {
        val w = src.width
        val h = src.height
        if (w < 2 || h < 2) return src

        val pixels = IntArray(w * h)
        src.getPixels(pixels, 0, w, 0, 0, w, h)
        val out = IntArray(pixels.size)

        val strength = 1.12f
        for (y in 1 until h - 1) {
            val rowTop = (y - 1) * w
            val rowMid = y * w
            val rowBot = (y + 1) * w
            for (x in 1 until w - 1) {
                val i = rowMid + x
                val c = pixels[i]
                val r = (c shr 16 and 0xFF).toFloat()
                val g = (c shr 8 and 0xFF).toFloat()
                val b = (c and 0xFF).toFloat()

                fun px(idx: Int, shift: Int): Float = ((pixels[idx] shr shift) and 0xFF).toFloat()

                val rSum = 5f * r - px(rowTop + x, 16) - px(rowBot + x, 16) - px(i - 1, 16) - px(i + 1, 16)
                val gSum = 5f * g - px(rowTop + x, 8) - px(rowBot + x, 8) - px(i - 1, 8) - px(i + 1, 8)
                val bSum = 5f * b - px(rowTop + x, 0) - px(rowBot + x, 0) - px(i - 1, 0) - px(i + 1, 0)

                val nr = (r + (rSum - r) * (strength - 1f)).toInt().coerceIn(0, 255)
                val ng = (g + (gSum - g) * (strength - 1f)).toInt().coerceIn(0, 255)
                val nb = (b + (bSum - b) * (strength - 1f)).toInt().coerceIn(0, 255)

                out[i] = (0xFF shl 24) or (nr shl 16) or (ng shl 8) or nb
            }
            if (y % 100 == 0) {
                onProgress(86 + (12 * y / h))
            }
        }

        for (y in 0 until h) {
            val row = y * w
            out[row] = pixels[row]
            out[row + w - 1] = pixels[row + w - 1]
        }
        for (x in 0 until w) {
            out[x] = pixels[x]
            out[(h - 1) * w + x] = pixels[(h - 1) * w + x]
        }

        onProgress(98)
        return Bitmap.createBitmap(out, w, h, Bitmap.Config.ARGB_8888)
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
                    if (delegate != null) delegate?.close()
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
            val modelScale = min(modelOutH / inH, modelOutW / inW).coerceAtLeast(1)

            val inputBuffer = bitmapToFloatBuffer(bitmap, inW, inH)
            val outputSize = outputTensor.numBytes()
            val outputBuffer = ByteBuffer.allocateDirect(outputSize).order(ByteOrder.nativeOrder())

            interpreter.run(inputBuffer, outputBuffer)
            onProgress(60)

            var result = floatBufferToBitmap(outputBuffer, modelOutW, modelOutH)

            if (modelScale < targetScale) {
                result = highQualityScale(result, targetScale / modelScale) { pct ->
                    onProgress(60 + (22 * pct / 100))
                }
            }
            onProgress(82)
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
