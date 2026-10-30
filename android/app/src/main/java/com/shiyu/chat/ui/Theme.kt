package com.shiyu.chat.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val ShiyuColors = lightColorScheme(
    primary = Color(0xFF176346),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFDCE9DC),
    onPrimaryContainer = Color(0xFF0E4933),
    secondary = Color(0xFF7A6F57),
    background = Color(0xFFF5F1E7),
    onBackground = Color(0xFF17241C),
    surface = Color(0xFFFBF9F2),
    onSurface = Color(0xFF17241C),
    surfaceVariant = Color(0xFFE7ECE5),
    onSurfaceVariant = Color(0xFF58665E),
    error = Color(0xFF9F433B),
)

@Composable
fun ShiyuTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = ShiyuColors, content = content)
}
