package com.example.superres

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
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
import java.io.InputStream

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var selectedBitmap: Bitmap? = null
    private var customModelUri: Uri? = null
    private var selectedModelName: String? = null

    private val pickImage = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let {
            selectedBitmap = loadBitmap(it)
            binding.tvStatus.text = getString(R.string.image_loaded, selectedBitmap?.width ?: 0, selectedBitmap?.height ?: 0)
            binding.btnRun.isEnabled = selectedBitmap != null
        }
    }

    private val pickModel = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let {
            customModelUri = it
            selectedModelName = "custom_" + System.currentTimeMillis() + ".tflite"
            binding.tvStatus.text = getString(R.string.custom_model_selected)
        }
    }

    private val requestPermission = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { grants ->
        val ok = grants[Manifest.permission.READ_MEDIA_IMAGES] == true || grants[Manifest.permission.READ_EXTERNAL_STORAGE] == true
        if (!ok) Toast.makeText(this, R.string.permission_denied, Toast.LENGTH_SHORT).show()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        ensurePermissions()
        setupUi()
    }

    private fun ensurePermissions() {
        val perms = mutableListOf<String>()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_IMAGES) != PackageManager.PERMISSION_GRANTED) {
            perms.add(Manifest.permission.READ_MEDIA_IMAGES)
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            perms.add(Manifest.permission.READ_EXTERNAL_STORAGE)
        }
        if (perms.isNotEmpty()) requestPermission.launch(perms.toTypedArray())
    }

    private fun setupUi() {
        val scales = listOf("1x", "2x", "3x", "4x", "8x", "16x")
        binding.spinnerScale.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, scales)
        binding.spinnerScale.setSelection(3)

        binding.btnPickImage.setOnClickListener { pickImage.launch("image/*") }
        binding.btnPickModel.setOnClickListener { pickModel.launch("*/*") }
        binding.btnRun.setOnClickListener { runEnhance() }
        binding.switchGpu.isChecked = true
        binding.switchGpu.text = getString(R.string.gpu_default)
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

    private fun runEnhance() {
        val src = selectedBitmap ?: return
        val scaleText = binding.spinnerScale.selectedItem?.toString() ?: "4x"
        val scale = scaleText.removeSuffix("x").toIntOrNull() ?: 4
        val useGpu = binding.switchGpu.isChecked
        val modelUri = customModelUri

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
                withContext(Dispatchers.Main) {
                    binding.imgPreview.setImageBitmap(result)
                    binding.tvStatus.text = getString(R.string.done, result.width, result.height)
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
}
