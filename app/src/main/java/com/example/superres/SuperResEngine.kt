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

        if (customModelUri == null) {
            onProgress(10)
            // 未导入模型时，使用高质量渐进式双三次插值作为兜底。
            progressiveScale(bitmap, targetScale, onProgress)
        } else {
            runTfLite(context, bitmap, targetScale, useGpu, customModelUri, customModelName ?: "custom.tflite", onProgress)
        }
    }

    private fun progressiveScale(src: Bitmap, targetScale: Int, onProgress: (Int) -> Unit): Bitmap {
        var current = src
        var currentScale = 1
        var step = 0
        while (currentScale < targetScale) {
            val nextScale = min(targetScale, currentScale * 2)
            val maxDim = 2400
            val w = (src.width * nextScale).coerceAtMost(maxDim)
            val h = (src.height * nextScale).coerceAtMost(maxDim)
            current = Bitmap.createScaledBitmap(src, w, h, true)
            currentScale = nextScale
            step++
            onProgress(10 + (70 * currentScale / targetScale))
        }
        onProgress(90)
        return current
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

            // 模型自带放大倍率由输出/输入比例推断
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
                result = progressiveScale(result, targetScale / modelScale) { pct ->
                    onProgress(60 + (30 * pct / 100))
                }
            }
            onProgress(95)
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
