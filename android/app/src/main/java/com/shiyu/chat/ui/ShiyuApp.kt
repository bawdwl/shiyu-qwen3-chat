package com.shiyu.chat.ui

import android.graphics.BitmapFactory
import android.net.Uri
import android.Manifest
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.DeleteOutline
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.VolumeUp
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.shiyu.chat.data.CharacterEntity
import com.shiyu.chat.data.ChatRepository
import com.shiyu.chat.data.MemoryEntity
import com.shiyu.chat.data.MessageEntity
import com.shiyu.chat.data.StructuredMemoryEntity
import com.shiyu.chat.data.LocalVoice
import com.shiyu.chat.data.naturalMessageText
import com.shiyu.chat.data.speechOnlyText
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.io.File
import java.util.Date
import java.util.Locale
import java.util.UUID

private val DeepForest = Color(0xFF0E4933)
private val Forest = Color(0xFF176346)
private val Cream = Color(0xFFF7F4EC)
private val Sage = Color(0xFFE3ECE2)
private val WarmGold = Color(0xFFE8DFC7)

@Composable
fun ShiyuApp(repository: ChatRepository) {
    val vm: ShiyuViewModel = viewModel(factory = ShiyuViewModelFactory(repository))
    val state by vm.state.collectAsStateWithLifecycle()
    var showSettings by remember { mutableStateOf(false) }
    var showCreate by remember { mutableStateOf(false) }
    var showMemory by remember { mutableStateOf(false) }
    var showEdit by remember { mutableStateOf(false) }
    var pendingCharacter by remember { mutableStateOf<CharacterEntity?>(null) }
    val snackbar = remember { SnackbarHostState() }

    LaunchedEffect(state.status) {
        state.status?.let {
            snackbar.showSnackbar(it)
            vm.clearStatus()
        }
    }

    Scaffold(
        snackbarHost = { SnackbarHost(snackbar) },
        contentWindowInsets = WindowInsets(0),
        containerColor = Color.Transparent,
    ) { padding ->
        Box(
            Modifier
                .fillMaxSize()
                .background(
                    Brush.linearGradient(
                        listOf(Color(0xFFE5ECE3), Color(0xFFF6F1E6), Color(0xFFE4ECE4)),
                    ),
                )
                .padding(padding),
        ) {
            if (state.selected == null) {
                CharacterHome(
                    state = state,
                    onSelect = { character ->
                        pendingCharacter = null
                        vm.select(character)
                    },
                    onSettings = { showSettings = true },
                    onCreate = {
                        pendingCharacter = null
                        showCreate = true
                    },
                )
            } else {
                ChatScreen(
                    state = state,
                    onBack = {
                        pendingCharacter = null
                        vm.closeCharacter()
                    },
                    onSend = vm::send,
                    onStop = vm::stop,
                    onMemory = { showMemory = true },
                    onEdit = {
                        pendingCharacter = null
                        showEdit = true
                    },
                    onReset = vm::reset,
                    onDelete = vm::deleteCharacter,
                )
            }
        }
    }

    if (showSettings) {
        SettingsDialog(
            state = state,
            onDismiss = { showSettings = false },
            onSaveKey = vm::saveApiKey,
            onRemoveKey = vm::removeApiKey,
            onSaveOpenCodeKey = vm::saveOpenCodeApiKey,
            onRemoveOpenCodeKey = vm::removeOpenCodeApiKey,
            onProviderChange = vm::setCloudProvider,
            onSaveLimits = vm::saveLimits,
            onSaveImageKey = vm::saveImageApiKey,
            onRemoveImageKey = vm::removeImageApiKey,
            onImportAvatarModel = vm::importAvatarModel,
        )
    }
    if (showCreate) {
        CreateCharacterDialog(
            busy = state.busy,
            onDismiss = {
                pendingCharacter = null
                showCreate = false
            },
            onCreate = { idea ->
                vm.generateCharacter(idea) { character ->
                    showCreate = false
                    pendingCharacter = character
                    showEdit = true
                }
            },
            onManual = {
                val character = CharacterEntity(
                    id = "custom-" + UUID.randomUUID().toString().take(12),
                    name = "新角色",
                    tag = "原创角色",
                    description = "等待补完的人物",
                    personality = "",
                    speechStyle = "",
                    relationship = "初次交谈",
                    appearance = "",
                    appearancePrompt = "",
                    greeting = "",
                    era = "原创世界",
                    knowledgeCutoff = "",
                    forbiddenKnowledge = "",
                    forbiddenTerms = "",
                    custom = true,
                )
                showCreate = false
                pendingCharacter = character
                showEdit = true
            },
        )
    }
    if (showMemory && state.selected != null) {
        MemorySheet(
            character = state.selected!!,
            memories = state.memories,
            structuredMemories = state.structuredMemories,
            memoryBusy = state.memoryBusy,
            onDismiss = { showMemory = false },
            onDelete = vm::deleteMemory,
            onEdit = vm::editMemory,
            onDeleteStructured = vm::deleteStructuredMemory,
        )
    }
    val editingCharacter = pendingCharacter ?: state.selected
    if (showEdit && editingCharacter != null) {
        EditCharacterDialog(
            character = editingCharacter,
            avatarBusy = state.avatarBusy,
            avatarModelReady = state.avatarModelReady,
            imageApiConfigured = state.imageApiConfigured,
            onDismiss = {
                showEdit = false
                pendingCharacter = null
            },
            onGenerateLocalAvatar = { edited ->
                vm.generateLocalAvatar(edited) { updated ->
                    pendingCharacter = updated
                }
            },
            onGenerateCloudAvatar = { edited ->
                vm.generateCloudAvatar(edited) { updated ->
                    pendingCharacter = updated
                }
            },
            onSave = { updated ->
                vm.saveCharacter(updated) {
                    showEdit = false
                    pendingCharacter = null
                    vm.select(updated)
                }
            },
        )
    }
}

@Composable
private fun CharacterHome(
    state: ShiyuUiState,
    onSelect: (CharacterEntity) -> Unit,
    onSettings: () -> Unit,
    onCreate: () -> Unit,
) {
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .statusBarsPadding(),
        contentPadding = PaddingValues(20.dp, 18.dp, 20.dp, 28.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        item {
            Row(
                Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        Modifier
                            .size(42.dp)
                            .clip(RoundedCornerShape(14.dp))
                            .background(DeepForest),
                        contentAlignment = Alignment.Center,
                    ) {
                        Text("拾", color = Cream, fontSize = 24.sp, fontWeight = FontWeight.Bold)
                    }
                    Spacer(Modifier.width(11.dp))
                    Column {
                        Text("拾语", fontSize = 22.sp, fontWeight = FontWeight.Bold)
                        Text("SHIYU", color = MaterialTheme.colorScheme.onSurfaceVariant, fontSize = 8.sp)
                    }
                }
                IconButton(onClick = onSettings) {
                    Icon(Icons.Default.Settings, contentDescription = "设置")
                }
            }
        }
        item {
            Surface(
                color = Color.Transparent,
                shape = RoundedCornerShape(28.dp),
                tonalElevation = 0.dp,
            ) {
                Column(
                    Modifier
                        .fillMaxWidth()
                        .background(
                            Brush.linearGradient(
                                listOf(DeepForest, Forest, Color(0xFF527A62)),
                            ),
                            RoundedCornerShape(28.dp),
                        )
                        .padding(28.dp),
                ) {
                    Text(
                        "A STORY THAT REMEMBERS",
                        color = Color(0xFFB9CABD),
                        letterSpacing = 2.sp,
                        fontSize = 9.sp,
                        fontWeight = FontWeight.Bold,
                    )
                    Spacer(Modifier.height(15.dp))
                    Text(
                        "不只回复一句话，\n而是记住这段故事。",
                        color = Cream,
                        fontSize = 31.sp,
                        lineHeight = 40.sp,
                        fontWeight = FontWeight.Medium,
                    )
                    Spacer(Modifier.height(13.dp))
                    Text(
                        "每位角色只有一个持续对话。聊过的内容会成为她之后理解你的依据。",
                        color = Color(0xFFC7D6CC),
                        lineHeight = 22.sp,
                    )
                    Spacer(Modifier.height(18.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        Button(
                            onClick = onCreate,
                            colors = ButtonDefaults.buttonColors(
                                containerColor = WarmGold,
                                contentColor = DeepForest,
                            ),
                        ) {
                            Icon(Icons.Default.Add, contentDescription = null)
                            Spacer(Modifier.width(5.dp))
                            Text("创建角色")
                        }
                        FilledTonalButton(onClick = onSettings) {
                            val providerName = if (state.cloudProvider == "opencode") "OpenCode" else "DeepSeek"
                            Text(if (state.apiConfigured) "$providerName 已连接" else "配置 $providerName")
                        }
                    }
                }
            }
        }
        item {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.Bottom,
            ) {
                Column {
                    Text("CHARACTERS", color = Forest, fontSize = 9.sp, letterSpacing = 2.sp)
                    Text("此刻，想和谁聊聊？", fontSize = 24.sp, fontWeight = FontWeight.SemiBold)
                }
                Text("${state.characters.size} 位角色", color = MaterialTheme.colorScheme.onSurfaceVariant, fontSize = 11.sp)
            }
        }
        items(state.characters, key = { it.id }) { character ->
            CharacterCard(character = character, onClick = { onSelect(character) })
        }
        item {
            Text(
                "AI 的回答可能不准确 · 聊天、记忆和图片保存在本机\n聊天内容和图片会发送给 ${if (state.cloudProvider == "opencode") "OpenCode Go" else "DeepSeek"}",
                modifier = Modifier.fillMaxWidth(),
                textAlign = TextAlign.Center,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                fontSize = 9.sp,
                lineHeight = 16.sp,
            )
        }
    }
}

@Composable
private fun CharacterCard(character: CharacterEntity, onClick: () -> Unit) {
    Surface(
        onClick = onClick,
        shape = RoundedCornerShape(24.dp),
        color = Color.White.copy(alpha = 0.72f),
        shadowElevation = 5.dp,
    ) {
        Column {
            Box(
                Modifier
                    .fillMaxWidth()
                    .height(230.dp)
                    .background(
                        Brush.linearGradient(
                            when (character.id) {
                                "companion" -> listOf(Color(0xFF9A806D), Color(0xFF365C49))
                                "interview" -> listOf(Color(0xFF526C91), Color(0xFF1E3249))
                                else -> listOf(Color(0xFF87967B), Color(0xFF294A38))
                            },
                        ),
                    ),
            ) {
                LocalAvatar(
                    path = character.avatarPath,
                    modifier = Modifier.matchParentSize(),
                )
                Surface(
                    modifier = Modifier
                        .align(Alignment.TopStart)
                        .padding(14.dp),
                    shape = RoundedCornerShape(100.dp),
                    color = DeepForest.copy(alpha = 0.55f),
                ) {
                    Text(
                        character.tag,
                        Modifier.padding(horizontal = 10.dp, vertical = 5.dp),
                        color = Color.White,
                        fontSize = 10.sp,
                    )
                }
                Column(
                    Modifier
                        .align(Alignment.BottomStart)
                        .padding(20.dp),
                ) {
                    Text(
                        character.name,
                        color = Color.White,
                        fontSize = 29.sp,
                        fontWeight = FontWeight.Medium,
                    )
                    Text(
                        character.description,
                        color = Color.White.copy(alpha = 0.82f),
                        fontSize = 11.sp,
                        lineHeight = 17.sp,
                    )
                }
            }
            Row(
                Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 12.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Text("原创 · AI 角色", color = MaterialTheme.colorScheme.onSurfaceVariant, fontSize = 10.sp)
                Text("进入故事 →", color = Forest, fontSize = 11.sp, fontWeight = FontWeight.Bold)
            }
        }
    }
}

@Composable
private fun ChatScreen(
    state: ShiyuUiState,
    onBack: () -> Unit,
    onSend: (String, Uri?) -> Unit,
    onStop: () -> Unit,
    onMemory: () -> Unit,
    onEdit: () -> Unit,
    onReset: () -> Unit,
    onDelete: () -> Unit,
) {
    val character = state.selected ?: return
    val voiceContext = LocalContext.current
    val voice = remember { LocalVoice(voiceContext.applicationContext) }
    val voiceScope = rememberCoroutineScope()
    var voiceBusy by remember { mutableStateOf(false) }
    var playingId by remember { mutableStateOf<Long?>(null) }
    var recording by remember { mutableStateOf(false) }
    var voiceStatus by remember { mutableStateOf<String?>(null) }
    val voicePrefs = remember { voiceContext.getSharedPreferences("character_voices", android.content.Context.MODE_PRIVATE) }
    var speaker by remember(character.id) { mutableStateOf(voicePrefs.getInt(character.id, if (character.name == "言川") 58 else 3)) }
    var confirmDelete by remember(character.id) { mutableStateOf(false) }
    var draft by remember(character.id) { mutableStateOf("") }
    fun startRecording() {
        if (voiceBusy) return
        voiceBusy = true; recording = true; voice.recording = true
        voiceScope.launch {
            val result = runCatching { withContext(Dispatchers.IO) { voice.recordAndRecognize() } }
            result.onSuccess { text ->
                if (text.isBlank()) voiceStatus = "没有识别到语音，请重试"
                else { draft = (draft + text).take(4096); voiceStatus = "识别完成，请检查后发送" }
            }.onFailure { voiceStatus = it.message ?: "语音识别失败" }
            recording = false; voiceBusy = false
        }
    }
    val microphone = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startRecording() else voiceStatus = "请允许使用麦克风后重试"
    }
    DisposableEffect(voice) { onDispose { voice.recording = false; voice.stopPlayback() } }
    var imageUri by remember(character.id) { mutableStateOf<Uri?>(null) }
    val listState = rememberLazyListState()
    val picker = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent(),
    ) { imageUri = it }

    LaunchedEffect(state.messages.size, state.streamingText) {
        if (state.messages.isNotEmpty()) {
            listState.animateScrollToItem((state.messages.size - 1).coerceAtLeast(0))
        }
    }

    Column(Modifier.fillMaxSize().statusBarsPadding()) {
        Surface(color = Cream.copy(alpha = 0.94f)) {
            Row(
                Modifier.fillMaxWidth().height(68.dp).padding(horizontal = 10.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                IconButton(onClick = onBack) {
                    Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "返回")
                }
                CharacterBadge(character, 38.dp)
                Spacer(Modifier.width(10.dp))
                Column(Modifier.weight(1f)) {
                    Text(character.name, fontWeight = FontWeight.SemiBold, fontSize = 17.sp)
                    Text("${character.tag} · 唯一对话", color = MaterialTheme.colorScheme.onSurfaceVariant, fontSize = 10.sp)
                }
                IconButton(onClick = onMemory) {
                    Icon(Icons.Default.Memory, contentDescription = "记忆")
                }
                IconButton(onClick = onEdit) {
                    Icon(Icons.Default.Edit, contentDescription = "编辑角色")
                }
                var menu by remember { mutableStateOf(false) }
                Box {
                    IconButton(onClick = { menu = true }) {
                        Icon(Icons.Default.MoreVert, contentDescription = "更多")
                    }
                    androidx.compose.material3.DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                        androidx.compose.material3.DropdownMenuItem(
                            text = { Text("重置角色对话") },
                            leadingIcon = { Icon(Icons.Default.DeleteOutline, contentDescription = null) },
                            onClick = {
                                menu = false
                                onReset()
                            },
                        )
                        listOf("女声 1" to 3, "女声 2" to 4, "男声 1" to 58, "男声 2" to 59).forEach { (label, id) ->
                            androidx.compose.material3.DropdownMenuItem(
                                text = { Text((if (speaker == id) "✓ " else "") + label) },
                                onClick = { speaker = id; voicePrefs.edit().putInt(character.id, id).apply(); menu = false },
                            )
                        }
                        if (character.custom) {
                            androidx.compose.material3.DropdownMenuItem(
                                text = { Text("删除角色") },
                                leadingIcon = { Icon(Icons.Default.DeleteOutline, contentDescription = null) },
                                enabled = !state.busy && !state.avatarBusy,
                                onClick = { menu = false; confirmDelete = true },
                            )
                        }
                    }
                }
            }
        }
        HorizontalDivider(color = Color.Black.copy(alpha = 0.07f))
        Box(Modifier.weight(1f)) {
            LocalAvatar(
                path = character.avatarPath,
                modifier = Modifier.fillMaxSize().alpha(0.09f),
            )
            if (state.messages.isEmpty() && state.streamingText.isEmpty()) {
                Column(
                    Modifier.fillMaxSize().padding(30.dp),
                    verticalArrangement = Arrangement.Center,
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    CharacterBadge(character, 86.dp)
                    Spacer(Modifier.height(20.dp))
                    Text("和${character.name}的故事，从此刻开始", fontSize = 24.sp, fontWeight = FontWeight.SemiBold, textAlign = TextAlign.Center)
                    Spacer(Modifier.height(10.dp))
                    Text(character.description, color = MaterialTheme.colorScheme.onSurfaceVariant, textAlign = TextAlign.Center)
                }
            } else {
                LazyColumn(
                    state = listState,
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(16.dp, 20.dp, 16.dp),
                    verticalArrangement = Arrangement.spacedBy(16.dp),
                ) {
                    items(state.messages, key = { it.id }) { message ->
                        MessageBubble(character, message, onPlay = {
                            if (playingId == message.id) {
                                voice.stopPlayback()
                                voiceStatus = "已停止播放"
                            } else if (!voiceBusy) {
                                voiceBusy = true
                                playingId = message.id
                                voiceStatus = "正在生成语音…"
                                voice.preparePlayback()
                                voiceScope.launch {
                                    runCatching { withContext(Dispatchers.IO) { voice.speak(speechOnlyText(message.content), speaker) } }
                                        .onFailure { voiceStatus = it.message ?: "语音播放失败" }
                                    voiceBusy = false; playingId = null
                                    if (voiceStatus == "正在生成语音…") voiceStatus = null
                                }
                            }
                        }, playing = { playingId == message.id })
                    }
                    if (state.busy) {
                        item {
                            MessageBubble(
                                character,
                                MessageEntity(
                                    characterId = character.id,
                                    role = "assistant",
                                    content = state.streamingText.ifBlank { "正在回应…" },
                                ),
                            )
                        }
                    }
                }
            }
        }
        Surface(color = Cream.copy(alpha = 0.96f)) {
            Column(
                Modifier
                    .fillMaxWidth()
                    .imePadding()
                    .navigationBarsPadding()
                    .padding(12.dp),
            ) {
                imageUri?.let { uri ->
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        UriImage(uri, Modifier.size(72.dp).clip(RoundedCornerShape(12.dp)))
                        TextButton(onClick = { imageUri = null }) { Text("移除图片") }
                    }
                }
                Surface(
                    shape = RoundedCornerShape(20.dp),
                    color = Color.White.copy(alpha = 0.92f),
                    shadowElevation = 8.dp,
                ) {
                    Column(Modifier.padding(12.dp)) {
                        OutlinedTextField(
                            value = draft,
                            onValueChange = { draft = it.take(4096) },
                            modifier = Modifier.fillMaxWidth(),
                            placeholder = { Text("和${character.name}说点什么…") },
                            minLines = 2,
                            maxLines = 5,
                            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                            keyboardActions = KeyboardActions(onSend = {
                                if (draft.isNotBlank() || imageUri != null) {
                                    onSend(draft.trim(), imageUri)
                                    draft = ""
                                    imageUri = null
                                }
                            }),
                        )
                        Row(
                            Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            IconButton(onClick = { picker.launch("image/*") }) {
                                Icon(Icons.Default.Image, contentDescription = "上传图片")
                            }
                            IconButton(onClick = {
                                if (recording) { voice.recording = false; recording = false; voiceStatus = "正在识别…" }
                                else if (!voiceBusy) microphone.launch(Manifest.permission.RECORD_AUDIO)
                            }, enabled = recording || !voiceBusy) {
                                Icon(if (recording) Icons.Default.Stop else Icons.Default.Mic, contentDescription = if (recording) "结束录音" else "语音输入")
                            }
                            if (state.busy) {
                                Button(
                                    onClick = onStop,
                                    colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error),
                                ) {
                                    Icon(Icons.Default.Stop, contentDescription = null)
                                    Text(" 停止")
                                }
                            } else {
                                Button(
                                    onClick = {
                                        onSend(draft.trim(), imageUri)
                                        draft = ""
                                        imageUri = null
                                    },
                                    enabled = draft.isNotBlank() || imageUri != null,
                                ) {
                                    Text("发送")
                                    Icon(Icons.AutoMirrored.Filled.Send, contentDescription = null)
                                }
                            }
                        }
                    }
                }
                Text(
                    voiceStatus ?: if (recording) "录音中，点击停止后识别" else if (voiceBusy) "语音处理中…" else "文字和图片会发送给 ${if (state.cloudProvider == "opencode") "OpenCode Go" else "DeepSeek"} · 语音在手机处理",
                    modifier = Modifier.fillMaxWidth().padding(top = 6.dp),
                    textAlign = TextAlign.Center,
                    fontSize = 9.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
    if (confirmDelete) {
        AlertDialog(
            onDismissRequest = { confirmDelete = false },
            title = { Text("删除${character.name}？") },
            text = { Text("角色档案、聊天记录、记忆及应用生成的头像会被删除，无法恢复。你从手机相册选择的原始图片不会删除。") },
            confirmButton = {
                TextButton(onClick = { confirmDelete = false; onDelete() }) { Text("删除角色", color = MaterialTheme.colorScheme.error) }
            },
            dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text("取消") } },
        )
    }
}

@Composable
private fun MessageBubble(character: CharacterEntity, message: MessageEntity, onPlay: (() -> Unit)? = null, playing: () -> Boolean = { false }) {
    val user = message.role == "user"
    val shownContent = naturalMessageText(message.content, message.displayJson)
    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = if (user) Arrangement.End else Arrangement.Start,
        verticalAlignment = Alignment.Top,
    ) {
        if (!user) {
            CharacterBadge(character, 34.dp)
            Spacer(Modifier.width(8.dp))
        }
        Column(
            horizontalAlignment = if (user) Alignment.End else Alignment.Start,
        ) {
            Surface(
                shape = if (user) {
                    RoundedCornerShape(18.dp, 5.dp, 18.dp, 18.dp)
                } else {
                    RoundedCornerShape(5.dp, 18.dp, 18.dp, 18.dp)
                },
                color = if (user) Forest else Color.White.copy(alpha = 0.88f),
                shadowElevation = 2.dp,
            ) {
                Column(Modifier.padding(13.dp)) {
                    Text(
                        shownContent,
                        color = if (user) Color.White else MaterialTheme.colorScheme.onSurface,
                        lineHeight = 22.sp,
                    )
                    message.imageUri?.let { raw ->
                        Spacer(Modifier.height(8.dp))
                        UriImage(
                            Uri.parse(raw),
                            Modifier.fillMaxWidth().height(190.dp).clip(RoundedCornerShape(12.dp)),
                        )
                    }
                }
            }
            if (message.id > 0) {
                if (!user && message.content.isNotBlank() && onPlay != null) {
                    TextButton(onClick = onPlay) { Icon(if (playing()) Icons.Default.Stop else Icons.Default.VolumeUp, contentDescription = null); Text(if (playing()) " 停止" else " 播放") }
                }
                Text(
                    SimpleDateFormat("M月d日 HH:mm", Locale.CHINA).format(Date(message.createdAt)),
                    fontSize = 8.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 4.dp),
                )
            }
        }
    }
}

@Composable
private fun CharacterBadge(character: CharacterEntity, size: androidx.compose.ui.unit.Dp) {
    Box(
        Modifier
            .size(size)
            .clip(CircleShape)
            .background(
                Brush.linearGradient(
                    listOf(Forest, Color(0xFF8A9C7F)),
                ),
            ),
        contentAlignment = Alignment.Center,
    ) {
        LocalAvatar(path = character.avatarPath, modifier = Modifier.fillMaxSize())
        if (character.avatarPath == null || !File(character.avatarPath).isFile) {
            Text(character.name.take(1), color = Color.White, fontWeight = FontWeight.Bold)
        }
    }
}

@Composable
private fun LocalAvatar(path: String?, modifier: Modifier = Modifier) {
    val modified = path?.let { File(it).lastModified() } ?: 0L
    val bitmap = remember(path, modified) {
        path?.takeIf { File(it).isFile }?.let {
            runCatching { BitmapFactory.decodeFile(it) }.getOrNull()
        }
    }
    if (bitmap != null) {
        Image(
            bitmap = bitmap.asImageBitmap(),
            contentDescription = null,
            modifier = modifier,
            contentScale = ContentScale.Crop,
        )
    }
}

@Composable
private fun UriImage(uri: Uri, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val bitmap = remember(uri) {
        runCatching {
            context.contentResolver.openInputStream(uri)?.use(BitmapFactory::decodeStream)
        }.getOrNull()
    }
    if (bitmap != null) {
        Image(
            bitmap = bitmap.asImageBitmap(),
            contentDescription = null,
            modifier = modifier,
            contentScale = ContentScale.Crop,
        )
    } else {
        Box(modifier.background(Sage), contentAlignment = Alignment.Center) {
            Text("图片", color = Forest)
        }
    }
}

@Composable
private fun SettingsDialog(
    state: ShiyuUiState,
    onDismiss: () -> Unit,
    onSaveKey: (String) -> Unit,
    onRemoveKey: () -> Unit,
    onSaveOpenCodeKey: (String) -> Unit,
    onRemoveOpenCodeKey: () -> Unit,
    onProviderChange: (String) -> Unit,
    onSaveLimits: (Int, Int) -> Unit,
    onSaveImageKey: (String) -> Unit,
    onRemoveImageKey: () -> Unit,
    onImportAvatarModel: (Uri) -> Unit,
) {
    var openCodeKey by remember { mutableStateOf("") }
    var deepSeekKey by remember { mutableStateOf("") }
    var imageKey by remember { mutableStateOf("") }
    var requestLimit by remember(state.limits) { mutableStateOf(state.limits.requestTokens.toString()) }
    var dailyLimit by remember(state.limits) { mutableStateOf(state.limits.dailyTokens.toString()) }
    val localAvatarPicker = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent(),
    ) { uri ->
        uri?.let(onImportAvatarModel)
    }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("应用设置") },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text("当前聊天 Provider", fontWeight = FontWeight.SemiBold)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    if (state.cloudProvider == "opencode") {
                        FilledTonalButton(onClick = { onProviderChange("opencode") }) {
                            Text("OpenCode Go")
                        }
                    } else {
                        OutlinedButton(onClick = { onProviderChange("opencode") }) {
                            Text("OpenCode Go")
                        }
                    }
                    if (state.cloudProvider == "deepseek") {
                        FilledTonalButton(onClick = { onProviderChange("deepseek") }) {
                            Text("DeepSeek")
                        }
                    } else {
                        OutlinedButton(onClick = { onProviderChange("deepseek") }) {
                            Text("DeepSeek")
                        }
                    }
                }
                HorizontalDivider()
                Text("OpenCode Go API", fontWeight = FontWeight.SemiBold)
                OutlinedTextField(
                    value = openCodeKey,
                    onValueChange = { openCodeKey = it },
                    label = { Text(if (state.openCodeConfigured) "替换 Key（留空不变）" else "OpenCode Go API Key") },
                    visualTransformation = PasswordVisualTransformation(),
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                Text("模型：deepseek-v4.1-flash · 支持文字、图片和流式输出", fontSize = 10.sp)
                if (state.openCodeConfigured) {
                    TextButton(onClick = onRemoveOpenCodeKey) {
                        Text("删除 OpenCode Go Key", color = MaterialTheme.colorScheme.error)
                    }
                }
                HorizontalDivider()
                Text("DeepSeek 备用接口", fontWeight = FontWeight.SemiBold)
                OutlinedTextField(
                    value = deepSeekKey,
                    onValueChange = { deepSeekKey = it },
                    label = { Text(if (state.deepSeekConfigured) "替换 Key（留空不变）" else "DeepSeek API Key") },
                    visualTransformation = PasswordVisualTransformation(),
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                Text("模型：deepseek-flash · 保留原有调用路径", fontSize = 10.sp)
                if (state.deepSeekConfigured) {
                    TextButton(onClick = onRemoveKey) {
                        Text("删除 DeepSeek Key", color = MaterialTheme.colorScheme.error)
                    }
                }
                HorizontalDivider()
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedTextField(
                        value = requestLimit,
                        onValueChange = { requestLimit = it.filter(Char::isDigit) },
                        label = { Text("单次 token") },
                        singleLine = true,
                        modifier = Modifier.weight(1f),
                    )
                    OutlinedTextField(
                        value = dailyLimit,
                        onValueChange = { dailyLimit = it.filter(Char::isDigit) },
                        label = { Text("每日 token") },
                        singleLine = true,
                        modifier = Modifier.weight(1f),
                    )
                }
                Text("今日已使用 ${state.todayUsage} tokens", color = MaterialTheme.colorScheme.onSurfaceVariant, fontSize = 10.sp)
                HorizontalDivider()
                Text("本地 MediaPipe SD1.5 角色头像", fontWeight = FontWeight.SemiBold)
                Text(
                    if (state.avatarModelReady) {
                        "本地模型已导入，可在角色编辑页离线生成头像。"
                    } else {
                        "导入 sd15-mediapipe.zip 后，可在手机本地生成头像。首次使用前初始化较慢。"
                    },
                    fontSize = 10.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                OutlinedButton(
                    onClick = { localAvatarPicker.launch("*/*") },
                    enabled = !state.avatarBusy,
                ) {
                    Text(if (state.avatarModelReady) "重新导入本地模型" else "导入本地 SD1.5 模型")
                }
                HorizontalDivider()
                Text("千问 AI 平台云端角色头像", fontWeight = FontWeight.SemiBold)
                Text(if (state.imageApiConfigured) "千问 AI 平台图片 API Key 已配置" else "填写千问 AI 平台通用 API Key；与 DeepSeek Key 不通用。", fontSize = 10.sp)
                OutlinedTextField(
                    value = imageKey, onValueChange = { imageKey = it },
                    label = { Text(if (state.imageApiConfigured) "替换千问 AI 平台 Key（留空不变）" else "千问 AI 平台 API Key") },
                    visualTransformation = PasswordVisualTransformation(), singleLine = true, modifier = Modifier.fillMaxWidth(),
                )
                Text("Qwen-Image 2.0 按图片计费；额度和实际费用以千问 AI 平台控制台为准。", fontSize = 10.sp)
                if (state.imageApiConfigured) TextButton(onClick = onRemoveImageKey) { Text("删除千问 AI 平台 Key", color = MaterialTheme.colorScheme.error) }
            }
        },
        confirmButton = {
            Button(onClick = {
                if (openCodeKey.isNotBlank()) onSaveOpenCodeKey(openCodeKey)
                if (deepSeekKey.isNotBlank()) onSaveKey(deepSeekKey)
                if (imageKey.isNotBlank()) onSaveImageKey(imageKey)
                onSaveLimits(requestLimit.toIntOrNull() ?: 12_000, dailyLimit.toIntOrNull() ?: 200_000)
                onDismiss()
            }) { Text("保存") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("取消") } },
    )
}

@Composable
private fun CreateCharacterDialog(
    busy: Boolean,
    onDismiss: () -> Unit,
    onCreate: (String) -> Unit,
    onManual: () -> Unit,
) {
    var idea by remember { mutableStateOf("") }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("创建角色") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text(
                    "写下身份、时代、关系和不能知道的内容。生成结果是草稿，请后续检查。",
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    fontSize = 11.sp,
                )
                OutlinedTextField(
                    value = idea,
                    onValueChange = { idea = it.take(1800) },
                    modifier = Modifier.fillMaxWidth().height(150.dp),
                    placeholder = {
                        Text("例如：李白身处唐代，以诗酒游历为生活重心；说话豪放敏锐；不能知道 Python、电脑和后世历史。")
                    },
                )
                TextButton(enabled = !busy, onClick = onManual) {
                    Text("不调用 API，手动创建空白角色")
                }
            }
        },
        confirmButton = {
            Button(enabled = idea.isNotBlank() && !busy, onClick = { onCreate(idea) }) {
                if (busy) CircularProgressIndicator(Modifier.size(18.dp), color = Color.White, strokeWidth = 2.dp)
                else Icon(Icons.Default.AutoAwesome, contentDescription = null)
                Text(if (busy) " 生成中" else " 生成草稿")
            }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("取消") } },
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun EditCharacterDialog(
    character: CharacterEntity,
    avatarBusy: Boolean,
    avatarModelReady: Boolean,
    imageApiConfigured: Boolean,
    onDismiss: () -> Unit,
    onGenerateLocalAvatar: (CharacterEntity) -> Unit,
    onGenerateCloudAvatar: (CharacterEntity) -> Unit,
    onSave: (CharacterEntity) -> Unit,
) {
    var name by remember(character.id) { mutableStateOf(character.name) }
    var tag by remember(character.id) { mutableStateOf(character.tag) }
    var description by remember(character.id) { mutableStateOf(character.description) }
    var personality by remember(character.id) { mutableStateOf(character.personality) }
    var speechStyle by remember(character.id) { mutableStateOf(character.speechStyle) }
    var relationship by remember(character.id) { mutableStateOf(character.relationship) }
    var appearance by remember(character.id) { mutableStateOf(character.appearance) }
    var greeting by remember(character.id) { mutableStateOf(character.greeting) }
    var characterKind by remember(character.id) { mutableStateOf(character.characterKind) }
    var era by remember(character.id) { mutableStateOf(character.era) }
    var cutoff by remember(character.id) { mutableStateOf(character.knowledgeCutoff) }
    var forbidden by remember(character.id) { mutableStateOf(character.forbiddenKnowledge) }
    var forbiddenTerms by remember(character.id) { mutableStateOf(character.forbiddenTerms) }
    var worldbookJson by remember(character.id) { mutableStateOf(character.worldbookJson) }
    fun editedCharacter() = character.copy(
        name = name.trim().ifBlank { character.name },
        tag = tag.trim(),
        description = description.trim(),
        personality = personality.trim(),
        speechStyle = speechStyle.trim(),
        relationship = relationship.trim(),
        appearance = appearance.trim(),
        appearancePrompt = if (appearance.trim() == character.appearance.trim()) {
            character.appearancePrompt
        } else {
            ""
        },
        greeting = greeting.trim(),
        characterKind = characterKind.trim(),
        era = era.trim(),
        knowledgeCutoff = cutoff.trim(),
        forbiddenKnowledge = forbidden.trim(),
        forbiddenTerms = forbiddenTerms.split(',', '，')
            .map(String::trim)
            .filter(String::isNotEmpty)
            .joinToString(","),
        worldbookJson = worldbookJson.trim().ifBlank { "[]" },
    )

    ModalBottomSheet(onDismissRequest = onDismiss) {
        LazyColumn(
            Modifier
                .fillMaxWidth()
                .fillMaxHeight(0.9f),
            contentPadding = PaddingValues(20.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            item {
                Text("编辑角色档案", fontSize = 25.sp, fontWeight = FontWeight.SemiBold)
                Text(
                    "认知边界会进入每次模型请求。正式修改保存在本机。",
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    fontSize = 11.sp,
                    modifier = Modifier.padding(top = 6.dp, bottom = 8.dp),
                )
            }
            item {
                Surface(
                    color = Sage.copy(alpha = 0.55f),
                    shape = RoundedCornerShape(18.dp),
                ) {
                    Column(
                        Modifier.fillMaxWidth().padding(15.dp),
                        verticalArrangement = Arrangement.spacedBy(9.dp),
                    ) {
                        Text("角色头像", fontWeight = FontWeight.SemiBold)
                        LocalAvatar(
                            path = character.avatarPath,
                            modifier = Modifier.fillMaxWidth().height(180.dp).clip(RoundedCornerShape(14.dp)),
                        )
                        Text(
                            if (avatarModelReady) {
                                "手机本地 MediaPipe SD1.5 已导入，可离线生成；也可以使用千问 AI 平台云端生成。"
                            } else {
                                "可在应用设置中导入本地 MediaPipe SD1.5；云端生成需要千问 AI 平台图片 Key。"
                            },
                            fontSize = 10.sp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(
                                onClick = { onGenerateLocalAvatar(editedCharacter()) },
                                enabled = avatarModelReady && !avatarBusy && name.isNotBlank(),
                            ) {
                                Text(if (avatarBusy) "生成中…" else "本地生成头像")
                            }
                            Button(
                                onClick = { onGenerateCloudAvatar(editedCharacter()) },
                                enabled = imageApiConfigured && !avatarBusy && name.isNotBlank(),
                            ) {
                                Text(if (avatarBusy) "生成中…" else "云端生成头像")
                            }
                        }
                    }
                }
            }
            item { OutlinedTextField(name, { name = it }, label = { Text("名字") }, modifier = Modifier.fillMaxWidth()) }
            item { OutlinedTextField(tag, { tag = it }, label = { Text("标签") }, modifier = Modifier.fillMaxWidth()) }
            item { OutlinedTextField(description, { description = it }, label = { Text("简介") }, minLines = 2, modifier = Modifier.fillMaxWidth()) }
            item { OutlinedTextField(personality, { personality = it }, label = { Text("性格") }, minLines = 2, modifier = Modifier.fillMaxWidth()) }
            item { OutlinedTextField(speechStyle, { speechStyle = it }, label = { Text("说话方式") }, minLines = 2, modifier = Modifier.fillMaxWidth()) }
            item { OutlinedTextField(relationship, { relationship = it }, label = { Text("与用户的关系") }, minLines = 2, modifier = Modifier.fillMaxWidth()) }
            item {
                OutlinedTextField(
                    value = appearance,
                    onValueChange = { appearance = it },
                    label = { Text("外貌描述（中文）") },
                    placeholder = {
                        Text("例如：二十五岁女性，鹅蛋脸，黑褐色及肩发，温和的眼睛，米白针织衫")
                    },
                    minLines = 3,
                    modifier = Modifier.fillMaxWidth(),
                )
                Text(
                    "本地 SD1.5 不直接理解中文；首次生成时会自动转换成英文视觉提示并保存在本机，之后可离线复用。",
                    fontSize = 10.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 4.dp),
                )
            }
            item { OutlinedTextField(greeting, { greeting = it }, label = { Text("角色主动开场") }, minLines = 3, modifier = Modifier.fillMaxWidth()) }
            item { OutlinedTextField(characterKind, { characterKind = it }, label = { Text("类型：original / historical / fictional_canon") }, modifier = Modifier.fillMaxWidth()) }
            item { OutlinedTextField(era, { era = it }, label = { Text("时代与世界") }, modifier = Modifier.fillMaxWidth()) }
            item { OutlinedTextField(cutoff, { cutoff = it }, label = { Text("知识截止 / 剧情阶段") }, modifier = Modifier.fillMaxWidth()) }
            item { OutlinedTextField(forbidden, { forbidden = it }, label = { Text("不能知道的内容") }, minLines = 3, modifier = Modifier.fillMaxWidth()) }
            item {
                OutlinedTextField(
                    forbiddenTerms,
                    { forbiddenTerms = it },
                    label = { Text("禁止知识关键词，用逗号分隔") },
                    modifier = Modifier.fillMaxWidth(),
                )
            }
            item {
                OutlinedTextField(
                    worldbookJson,
                    { worldbookJson = it },
                    label = { Text("世界书 JSON（title / keywords / content）") },
                    minLines = 5,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
            item {
                Row(
                    Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.End,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    TextButton(onClick = onDismiss) { Text("取消") }
                    Button(
                        onClick = {
                            onSave(editedCharacter())
                        },
                        enabled = !avatarBusy,
                    ) { Text("保存档案") }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun MemorySheet(
    character: CharacterEntity,
    memories: List<MemoryEntity>,
    structuredMemories: List<StructuredMemoryEntity>,
    memoryBusy: Boolean,
    onDismiss: () -> Unit,
    onDelete: (Long) -> Unit,
    onEdit: (Long, String) -> Unit,
    onDeleteStructured: (Long) -> Unit,
) {
    var editing by remember { mutableStateOf<MemoryEntity?>(null) }
    var deleting by remember { mutableStateOf<MemoryEntity?>(null) }
    var editedText by remember { mutableStateOf("") }
    ModalBottomSheet(onDismissRequest = onDismiss) {
        Column(
            Modifier
                .fillMaxWidth()
                .fillMaxHeight(0.72f)
                .padding(20.dp),
        ) {
            Text("${character.name}的长期记忆", fontSize = 24.sp, fontWeight = FontWeight.SemiBold)
            Text(
                "结构化事实会在聊天后由后台自动整理，不需要逐条确认。删除一项事实会同时移除它的历史版本；你也可以直接在对话中更正。",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                fontSize = 11.sp,
                modifier = Modifier.padding(top = 8.dp, bottom = 15.dp),
            )
            LazyColumn(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(9.dp)) {
                item {
                    Row(
                        Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text("结构化事实", fontWeight = FontWeight.SemiBold)
                            Text(
                                "${structuredMemories.count { it.state == "active" }} 条当前 · " +
                                    "${structuredMemories.count { it.state == "contested" }} 条待判断",
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                fontSize = 10.sp,
                            )
                        }
                    }
                }
                if (structuredMemories.isEmpty()) {
                    item {
                        Text(
                            "还没有结构化记忆。下一轮聊天完成后，后台会自动整理值得长期保留的信息。",
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            fontSize = 11.sp,
                            modifier = Modifier.padding(vertical = 12.dp),
                        )
                    }
                }
                items(structuredMemories, key = { "structured-${it.id}" }) { memory ->
                    Surface(
                        color = when (memory.state) {
                            "active" -> Sage
                            "candidate", "contested" -> WarmGold.copy(alpha = 0.65f)
                            else -> Color.White.copy(alpha = 0.72f)
                        },
                        shape = RoundedCornerShape(14.dp),
                    ) {
                        Column(Modifier.padding(13.dp)) {
                            Text(
                                when (memory.state) {
                                    "active" -> "当前事实"
                                    "candidate" -> "历史草稿"
                                    "contested" -> "有疑义"
                                    "superseded" -> "已被替代"
                                    else -> "已删除"
                                },
                                color = Forest,
                                fontSize = 9.sp,
                                fontWeight = FontWeight.Bold,
                            )
                            Text(memory.content, fontSize = 12.sp, lineHeight = 19.sp)
                            memory.value?.let {
                                Text("准确值：$it", color = MaterialTheme.colorScheme.onSurfaceVariant, fontSize = 10.sp)
                            }
                            if (memory.state != "deleted") {
                                Row(
                                    Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.End,
                                ) {
                                    TextButton(
                                        enabled = !memoryBusy,
                                        onClick = { onDeleteStructured(memory.id) },
                                    ) {
                                        Text("删除", color = MaterialTheme.colorScheme.error)
                                    }
                                }
                            }
                        }
                    }
                }
                item {
                    HorizontalDivider(Modifier.padding(vertical = 8.dp))
                    Text("旧对话摘录", fontWeight = FontWeight.SemiBold)
                }
                if (memories.isEmpty()) {
                    item {
                        Text(
                            "还没有旧对话摘录",
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            fontSize = 11.sp,
                            modifier = Modifier.padding(vertical = 10.dp),
                        )
                    }
                }
                items(memories, key = { "excerpt-${it.id}" }) { memory ->
                        Surface(
                            color = Color.White.copy(alpha = 0.75f),
                            shape = RoundedCornerShape(14.dp),
                        ) {
                            Column(Modifier.padding(13.dp)) {
                                Row(
                                    Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically,
                                ) {
                                    Text(if (memory.memoryType == "user_edited") "用户修订 · 来源为旧对话" else "旧对话摘录", color = Forest, fontSize = 9.sp, fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
                                    IconButton(onClick = { editing = memory; editedText = memory.content }) {
                                        Icon(Icons.Default.Edit, contentDescription = "编辑记忆")
                                    }
                                    IconButton(onClick = { deleting = memory }) {
                                        Icon(Icons.Default.DeleteOutline, contentDescription = "删除记忆")
                                    }
                                }
                                Text(memory.content, fontSize = 12.sp, lineHeight = 19.sp)
                                Text("来源消息编号：${memory.sourceMessageIds}", fontSize = 9.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                        }
                }
            }
        }
    }
    editing?.let { memory ->
        AlertDialog(onDismissRequest = { editing = null }, title = { Text("修改记忆") },
            text = { OutlinedTextField(value = editedText, onValueChange = { editedText = it.take(1200) }, minLines = 3, label = { Text("记忆内容") }) },
            confirmButton = { TextButton(enabled = editedText.isNotBlank(), onClick = { onEdit(memory.id, editedText); editing = null }) { Text("保存") } },
            dismissButton = { TextButton(onClick = { editing = null }) { Text("取消") } })
    }
    deleting?.let { memory ->
        AlertDialog(onDismissRequest = { deleting = null }, title = { Text("删除这条记忆？") },
            text = { Text("该摘录会被删除，同一来源不会再自动提取。原始聊天不会删除，近期聊天原文仍可能影响回答。") },
            confirmButton = { TextButton(onClick = { onDelete(memory.id); deleting = null }) { Text("删除") } },
            dismissButton = { TextButton(onClick = { deleting = null }) { Text("取消") } })
    }
}
