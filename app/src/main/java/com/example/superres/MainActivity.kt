package com.example.superres
import android.graphics.Color
import android.view.View

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.superres.databinding.ActivityMainBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var selectedBitmap: Bitmap? = null
    private var customModelUri: Uri? = null
    private var selectedModelName: String? = null
    private var modelSupported: Boolean = false

    private val pickImage = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let {
            val bmp = loadBitmap(it)
            selectedBitmap = bmp
            binding.imgPreview.setImageBitmap(bmp)
            binding.tvImportInfo.text = getString(R.string.image_loaded, bmp?.width ?: 0, bmp?.height ?: 0) + if (bmp?.hasAlpha() == true) "\n透明贴图：是" else "\n透明贴图：否"
            binding.btnRun.isEnabled = bmp != null
        }
    }

    private val pickModel = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let {
            customModelUri = it
            selectedModelName = getModelFileName(it) ?: ("custom_" + System.currentTimeMillis() + ".tflite")
            modelSupported = isSupportedModel(selectedModelName)
            binding.tvModelPath.text = getString(R.string.model_path, selectedModelName)
            binding.tvModelStatus.text = if (modelSupported) getString(R.string.model_ready) else getString(R.string.model_unsupported)
            binding.tvStatus.text = if (modelSupported) getString(R.string.custom_model_selected) else getString(R.string.model_will_fallback)
        }
    }

    private val requestPermission = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { grants ->
        val ok = grants[Manifest.permission.READ_MEDIA_IMAGES] == true ||
                 grants[Manifest.permission.READ_EXTERNAL_STORAGE] == true
        if (!ok) Toast.makeText(this, R.string.permission_denied, Toast.LENGTH_SHORT).show()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        window.statusBarColor = Color.TRANSPARENT
        window.decorView.systemUiVisibility = View.SYSTEM_UI_FLAG_LAYOUT_STABLE or View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
        ensurePermissions()
        setupUi()
    }

    private fun ensurePermissions() {
        val perms = mutableListOf<String>()
        if (Build.VERSION.SDK_INT >= 33) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_IMAGES) != PackageManager.PERMISSION_GRANTED) {
                perms.add(Manifest.permission.READ_MEDIA_IMAGES)
            }
        } else {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                perms.add(Manifest.permission.READ_EXTERNAL_STORAGE)
            }
            if (Build.VERSION.SDK_INT <= 28 &&
                ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                perms.add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
            }
        }
        if (perms.isNotEmpty()) requestPermission.launch(perms.toTypedArray())
    }

    private fun setupUi() {
        val scales = listOf("1x", "2x", "3x", "4x", "8x")
        binding.spinnerScale.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, scales)
        binding.spinnerScale.setSelection(3)

        binding.btnPickImage.setOnClickListener { pickImage.launch("image/*") }
        binding.btnPickModel.setOnClickListener { pickModel.launch("*/*") }
        binding.btnRun.setOnClickListener { runEnhance() }
        binding.switchGpu.isChecked = true
        binding.switchGpu.text = getString(R.string.use_gpu)
        binding.switchAutoSave.isChecked = true
    }

    private fun loadBitmap(uri: Uri): Bitmap? {
        return try {
            val stream: InputStream? = contentResolver.openInputStream(uri)
            stream?.use { BitmapFactory.decodeStream(it) }
        } catch (e: Exception) {
            Toast.makeText(this, getString(R.string.load_failed, e.message ?: ""), Toast.LENGTH_LONG).show()
            null
        }
    }

    private fun getModelFileName(uri: Uri): String? {
        return try {
            contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                val nameIndex = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                if (nameIndex >= 0 && cursor.moveToFirst()) cursor.getString(nameIndex) else null
            }
        } catch (e: Exception) { null }
    }

    private fun isSupportedModel(name: String?): Boolean {
        val n = name?.lowercase() ?: return false
        return n.endsWith(".tflite") || n.endsWith(".lite")
    }

    private fun runEnhance() {
        val src = selectedBitmap ?: return
        val scaleText = binding.spinnerScale.selectedItem?.toString() ?: "4x"
        val scale = scaleText.removeSuffix("x").toIntOrNull() ?: 4
        val useGpu = binding.switchGpu.isChecked
        val autoSave = binding.switchAutoSave.isChecked
        val modelUri = if (modelSupported) customModelUri else null

        binding.btnRun.isEnabled = false
        binding.progressBar.isIndeterminate = true
        binding.tvStatus.text = getString(R.string.enhancing)

        CoroutineScope(Dispatchers.IO).launch {
            try {
                val result = SuperResEngine.enhance(
                    context = this@MainActivity,
                    bitmap = src,
                    scale = scale,
                    useGpu = useGpu,
                    customModelUri = modelUri,
                    customModelName = selectedModelName,
                    onProgress = { pct ->
                        runOnUiThread {
                            binding.progressBar.isIndeterminate = false
                            binding.progressBar.progress = pct
                        }
                    }
                )

                var saveInfo = ""
                if (autoSave) saveInfo = saveBitmapToGallery(result)

                withContext(Dispatchers.Main) {
                    binding.imgPreview.setImageBitmap(result)
                    val done = getString(R.string.done, result.width, result.height)
                    binding.tvStatus.text = if (saveInfo.isNotBlank()) "$done\n$saveInfo" else done
                    binding.btnRun.isEnabled = true
                    binding.progressBar.progress = 100
                    Toast.makeText(this@MainActivity, R.string.done_short, Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    binding.tvStatus.text = getString(R.string.error, e.message ?: "unknown")
                    binding.btnRun.isEnabled = true
                    binding.progressBar.isIndeterminate = false
                    Toast.makeText(this@MainActivity, getString(R.string.error_short, e.message ?: ""), Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun saveBitmapToGallery(bitmap: Bitmap): String {
        return try {
            val fileName = "SuperRes_${System.currentTimeMillis()}.png"
            if (Build.VERSION.SDK_INT >= 29) {
                val values = ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, fileName)
                    put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                    put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/SuperRes")
                }
                val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                    ?: throw IllegalStateException("无法创建媒体文件")
                contentResolver.openOutputStream(uri).use { out ->
                    if (out == null) throw IllegalStateException("无法打开输出流")
                    bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
                }
                getString(R.string.saved_to, "Pictures/SuperRes/$fileName")
            } else {
                val dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES).let { File(it, "SuperRes") }
                if (!dir.exists()) dir.mkdirs()
                val file = File(dir, fileName)
                FileOutputStream(file).use { out -> bitmap.compress(Bitmap.CompressFormat.PNG, 100, out) }
                getString(R.string.saved_to, file.absolutePath)
            }
        } catch (e: Exception) {
            getString(R.string.save_failed, e.message ?: "")
        }
    }
}
