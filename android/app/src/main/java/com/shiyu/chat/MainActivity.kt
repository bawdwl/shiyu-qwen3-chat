package com.shiyu.chat

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.shiyu.chat.ui.ShiyuApp
import com.shiyu.chat.ui.ShiyuTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            ShiyuTheme {
                ShiyuApp(application.shiyu.repository)
            }
        }
    }
}
