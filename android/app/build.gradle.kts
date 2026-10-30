plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    id("com.google.devtools.ksp")
}

val stableDebugKeystore = rootProject.file("../outputs/android-toolchain/shiyu-debug.keystore")

android {
    namespace = "com.shiyu.chat"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.shiyu.chat"
        minSdk = 26
        targetSdk = 35
        versionCode = 14
        versionName = "0.9.4"
    }

    buildFeatures {
        compose = true
        buildConfig = false
    }

    signingConfigs {
        if (stableDebugKeystore.isFile) {
            create("stableDebug") {
                storeFile = stableDebugKeystore
                storePassword = "android"
                keyAlias = "androiddebugkey"
                keyPassword = "android"
            }
        }
    }

    buildTypes {
        getByName("debug") {
            if (stableDebugKeystore.isFile) {
                signingConfig = signingConfigs.getByName("stableDebug")
            }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    packaging {
        resources.excludes += "/META-INF/{AL2.0,LGPL2.1}"
    }
}

configurations.configureEach {
    // MediaPipe 0.10.26.1 declares protobuf-javalite, but its ImageGenerator
    // bytecode calls the covariant Any.Builder.build():Any found in
    // protobuf-java. Shipping only javalite causes a runtime NoSuchMethodError.
    exclude(group = "com.google.protobuf", module = "protobuf-javalite")
}

ksp {
    arg("room.generateKotlin", "true")
}

dependencies {
    implementation(files("libs/sherpa-onnx-1.13.8.aar"))
    val composeBom = platform("androidx.compose:compose-bom:2025.02.00")
    implementation(composeBom)
    androidTestImplementation(composeBom)

    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.activity:activity-compose:1.10.1")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    debugImplementation("androidx.compose.ui:ui-tooling")

    implementation("androidx.room:room-runtime:2.6.1")
    implementation("androidx.room:room-ktx:2.6.1")
    ksp("androidx.room:room-compiler:2.6.1")
    implementation("androidx.work:work-runtime-ktx:2.10.0")
    implementation("androidx.security:security-crypto:1.1.0-alpha06")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("com.google.mediapipe:tasks-vision-image-generator:0.10.26.1")
    implementation("com.google.protobuf:protobuf-java:4.26.1")

    testImplementation("junit:junit:4.13.2")
}

