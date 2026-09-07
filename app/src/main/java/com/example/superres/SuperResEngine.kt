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

    private const val MAX_PIXELS = 8000000

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
            onProgress(5)
            progressive(bitmap, targetScale, onProgress)
        } else {
            runModel(context, bitmap, targetScale, useGpu, customModelUri, customModelName ?: "custom.tflite", onProgress)
        }
        onProgress(90)
        enhanceQuality(upscaled, onProgress)
    }

    private fun progressive(src: Bitmap, targetScale: Int, onProgress: (Int) -> Unit): Bitmap {
        var current = src
        var currentScale = 1
        val maxDim = 2400
        while (currentScale < targetScale) {
            val next = (currentScale * 2).coerceAtMost(targetScale)
            val w = (src.width * next).coerceAtMost(maxDim).coerceAtLeast(1)
            val h = (src.height * next).coerceAtMost(maxDim).coerceAtLeast(1)
            current = Bitmap.createScaledBitmap(src, w, h, true)
            currentScale = next
            onProgress(5 + (80 * currentScale / targetScale))
            if (w.toLong() * h.toLong() > MAX_PIXELS) current = downscale(current)
        }
        return current
    }

    private fun downscale(bitmap: Bitmap): Bitmap {
        val p = bitmap.width.toLong() * bitmap.height.toLong()
        if (p <= MAX_PIXELS) return bitmap
        val ratio = kotlin.math.sqrt(MAX_PIXELS.toDouble() / p)
        return Bitmap.createScaledBitmap(bitmap, (bitmap.width * ratio).toInt().coerceAtLeast(1), (bitmap.height * ratio).toInt().coerceAtLeast(1), true)
    }

    private fun enhanceQuality(bitmap: Bitmap, onProgress: (Int) -> Unit): Bitmap {
        var out = colorEnhance(bitmap)
        onProgress(93)
        out = sharpen(out)
        onProgress(98)
        return out
    }

    private fun colorEnhance(bitmap: Bitmap): Bitmap {
        val out = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        val c = Canvas(out)
        val p = Paint().apply {
            isAntiAlias = true
            isFilterBitmap = true
            colorFilter = ColorMatrixColorFilter(ColorMatrix().apply { setSaturation(1.12f) })
        }
        c.drawBitmap(bitmap, 0f, 0f, p)
        return out
    }

    private fun sharpen(bitmap: Bitmap): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
        val sharp = pixels.copyOf()
        val k = 0.18f
        for (y in 0 until h) {
            val ym = (y - 1).coerceAtLeast(0)
            val yp = (y + 1).coerceAtMost(h - 1)
            val rm = ym * w
            val rc = y * w
            val rp = yp * w
            for (x in 0 until w) {
                val xm = (x - 1).coerceAtLeast(0)
                val xp = (x + 1).coerceAtMost(w - 1)
                val i = rc + x
                val c = pixels[i]
                val r = sharpenCh(Color.red(c), Color.red(pixels[rm + xm]), Color.red(pixels[rm + x]), Color.red(pixels[rm + xp]), Color.red(pixels[rc + xm]), Color.red(pixels[rc + xp]), Color.red(pixels[rp + xm]), Color.red(pixels[rp + x]), Color.red(pixels[rp + xp]), k)
                val g = sharpenCh(Color.green(c), Color.green(pixels[rm + xm]), Color.green(pixels[rm + x]), Color.green(pixels[rm + xp]), Color.green(pixels[rc + xm]), Color.green(pixels[rc + xp]), Color.green(pixels[rp + xm]), Color.green(pixels[rp + x]), Color.green(pixels[rp + xp]), k)
                val b = sharpenCh(Color.blue(c), Color.blue(pixels[rm + xm]), Color.blue(pixels[rm + x]), Color.blue(pixels[rm + xp]), Color.blue(pixels[rc + xm]), Color.blue(pixels[rc + xp]), Color.blue(pixels[rp + xm]), Color.blue(pixels[rp + x]), Color.blue(pixels[rp + xp]), k)
                sharp[i] = Color.rgb(r, g, b)
            }
        }
        out.setPixels(sharp, 0, w, 0, 0, w, h)
        return out
    }

    private fun sharpenCh(c: Int, tl: Int, t: Int, tr: Int, l: Int, r: Int, bl: Int, b: Int, br: Int, k: Float): Int {
        val n = tl + t + tr + l + r + bl + b + br
        return (c + (k * (8 * c - n)).toInt()).coerceIn(0, 255)
    }

    private fun runModel(
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
            requireNotNull(input) { "read model failed" }
            modelFile.outputStream().use { out -> input.copyTo(out) }
        }
        var gpu: GpuDelegate? = null
        var interpreter: Interpreter? = null
        return try {
            val options = Interpreter.Options()
            if (useGpu) {
                try {
                    gpu = GpuDelegate()
                    options.addDelegate(gpu)
                } catch (e: Exception) {
                    gpu?.close()
                    gpu = null
                }
            }
            interpreter = try {
                Interpreter(modelFile, options)
            } catch (e: Exception) {
                if (gpu != null) {
                    gpu.close()
                    gpu = null
                    options.clearDelegates()
                    Interpreter(modelFile, options)
                } else throw e
            }
            val it = interpreter.getInputTensor(0)
            val ot = interpreter.getOutputTensor(0)
            val ish = it.shape()
            val osh = ot.shape()
            val inH = if (ish.size >= 2) ish[1] else bitmap.height
            val inW = if (ish.size >= 3) ish[2] else bitmap.width
            val bytesPer = it.numBytes().toLong() / (inH.toLong() * inW.toLong()).coerceAtLeast(1L)
            val ch = bytesPer.coerceIn(1L, 4L).toInt()
            onProgress(20)
            val outH = if (osh.size >= 2) osh[1] else inH * 4
            val outW = if (osh.size >= 3) osh[2] else inW * 4
            val modelScale = minOf(outH / inH, outW / inW).coerceAtLeast(1)
            val inBuf = makeInput(bitmap, inW, inH, ch, it)
            val outBuf = ByteBuffer.allocateDirect(ot.numBytes()).order(ByteOrder.nativeOrder())
            interpreter.run(inBuf, outBuf)
            onProgress(60)
            var result = if (ot.dataType() == org.tensorflow.lite.DataType.UINT8) uint8ToBitmap(outBuf, outW, outH) else floatToBitmap(outBuf, outW, outH)
            if (modelScale < targetScale) result = progressive(result, targetScale / modelScale) { p -> onProgress(60 + (25 * p / 100)) }
            onProgress(85)
            result
        } finally {
            try { interpreter?.close() } catch (_: Exception) {}
            try { gpu?.close() } catch (_: Exception) {}
            if (modelFile.exists()) modelFile.delete()
        }
    }

    private fun makeInput(bitmap: Bitmap, w: Int, h: Int, ch: Int, tensor: org.tensorflow.lite.Tensor): ByteBuffer {
        val scaled = if (bitmap.width != w || bitmap.height != h) Bitmap.createScaledBitmap(bitmap, w, h, true) else bitmap
        val isU8 = tensor.dataType() == org.tensorflow.lite.DataType.UINT8
        val bytes = if (isU8) ch else ch * 4
        val buf = ByteBuffer.allocateDirect(w * h * bytes).order(ByteOrder.nativeOrder())
        val pixels = IntArray(w * h)
        scaled.getPixels(pixels, 0, w, 0, 0, w, h)
        for (px in pixels) {
            val r = px shr 16 and 255
            val g = px shr 8 and 255
            val b = px and 255
            if (isU8) {
                buf.put(r.toByte())
                buf.put(g.toByte())
                buf.put(b.toByte())
                if (ch == 4) buf.put(255.toByte())
            } else {
                buf.putFloat(r / 255f)
                buf.putFloat(g / 255f)
                buf.putFloat(b / 255f)
                if (ch == 4) buf.putFloat(1f)
            }
        }
        buf.rewind()
        return buf
    }

    private fun floatToBitmap(buffer: ByteBuffer, w: Int, h: Int): Bitmap {
        buffer.rewind()
        val pixels = IntArray(w * h)
        for (i in pixels.indices) {
            val r = (buffer.float * 255f).toInt().coerceIn(0, 255)
            val g = (buffer.float * 255f).toInt().coerceIn(0, 255)
            val b = (buffer.float * 255f).toInt().coerceIn(0, 255)
            pixels[i] = (255 shl 24) or (r shl 16) or (g shl 8) or b
        }
        return Bitmap.createBitmap(pixels, w, h, Bitmap.Config.ARGB_8888)
    }

    private fun uint8ToBitmap(buffer: ByteBuffer, w: Int, h: Int): Bitmap {
        buffer.rewind()
        val pixels = IntArray(w * h)
        for (i in pixels.indices) {
            val r = buffer.get().toInt() and 255
            val g = buffer.get().toInt() and 255
            val b = buffer.get().toInt() and 255
            pixels[i] = (255 shl 24) or (r shl 16) or (g shl 8) or b
        }
        return Bitmap.createBitmap(pixels, w, h, Bitmap.Config.ARGB_8888)
    }
}
