package com.shiyu.chat

import android.app.Application
import android.content.Context
import androidx.room.Room
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import com.shiyu.chat.data.AppDatabase
import com.shiyu.chat.data.ChatRepository

class ShiyuApplication : Application() {
    val database: AppDatabase by lazy {
        Room.databaseBuilder(this, AppDatabase::class.java, "shiyu.db")
            .addMigrations(
                AppDatabase.MIGRATION_2_3,
                AppDatabase.MIGRATION_3_4,
                AppDatabase.MIGRATION_4_5,
                AppDatabase.MIGRATION_5_6,
                AppDatabase.MIGRATION_6_7,
                AppDatabase.MIGRATION_7_8,
                AppDatabase.MIGRATION_8_9,
                AppDatabase.MIGRATION_9_10,
            )
            .addCallback(AppDatabase.CREATE_SEARCH_INDEX_CALLBACK)
            .build()
    }

    val repository: ChatRepository by lazy {
        val masterKey = MasterKey.Builder(this)
            .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
            .build()
        val secrets = EncryptedSharedPreferences.create(
            this,
            "shiyu_secrets",
            masterKey,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM,
        )
        ChatRepository(this, database, secrets)
    }
}

val Context.shiyu: ShiyuApplication
    get() = applicationContext as ShiyuApplication
