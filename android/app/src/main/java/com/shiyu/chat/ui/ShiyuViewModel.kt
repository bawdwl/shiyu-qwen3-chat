package com.shiyu.chat.ui

import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.shiyu.chat.data.CharacterEntity
import com.shiyu.chat.data.ChatRepository
import com.shiyu.chat.data.Limits
import com.shiyu.chat.data.MemoryEntity
import com.shiyu.chat.data.MessageEntity
import com.shiyu.chat.data.StructuredMemoryEntity
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class ShiyuUiState(
    val characters: List<CharacterEntity> = emptyList(),
    val selected: CharacterEntity? = null,
    val messages: List<MessageEntity> = emptyList(),
    val memories: List<MemoryEntity> = emptyList(),
    val structuredMemories: List<StructuredMemoryEntity> = emptyList(),
    val cloudProvider: String = "opencode",
    val apiConfigured: Boolean = false,
    val openCodeConfigured: Boolean = false,
    val deepSeekConfigured: Boolean = false,
    val imageApiConfigured: Boolean = false,
    val limits: Limits = Limits(),
    val todayUsage: Int = 0,
    val streamingText: String = "",
    val busy: Boolean = false,
    val avatarBusy: Boolean = false,
    val memoryBusy: Boolean = false,
    val avatarModelReady: Boolean = false,
    val status: String? = null,
)

class ShiyuViewModel(private val repository: ChatRepository) : ViewModel() {
    private val _state = MutableStateFlow(ShiyuUiState())
    val state: StateFlow<ShiyuUiState> = _state.asStateFlow()
    private var stopRequested = false
    private var observeJob: Job? = null

    init {
        viewModelScope.launch {
            repository.ensureDefaults()
            val provider = repository.cloudProvider()
            _state.value = _state.value.copy(
                cloudProvider = provider,
                apiConfigured = !repository.activeApiKey().isNullOrBlank(),
                openCodeConfigured = !repository.openCodeApiKey().isNullOrBlank(),
                deepSeekConfigured = !repository.apiKey().isNullOrBlank(),
                imageApiConfigured = repository.imageApiKeyConfigured(),
                limits = repository.limits(),
                todayUsage = repository.todayUsage(),
                avatarModelReady = repository.avatarModelReady(),
            )
            repository.characters.collect { characters ->
                val selected = _state.value.selected?.let { current ->
                    characters.firstOrNull { it.id == current.id }
                }
                _state.value = _state.value.copy(characters = characters, selected = selected)
            }
        }
    }

    fun saveApiKey(value: String) {
        runCatching { repository.saveApiKey(value) }
            .onSuccess {
                _state.value = _state.value.copy(
                    deepSeekConfigured = true,
                    apiConfigured = !repository.activeApiKey().isNullOrBlank(),
                    status = "DeepSeek API Key 已安全保存",
                )
            }
            .onFailure { _state.value = _state.value.copy(status = it.message) }
    }

    fun removeApiKey() {
        repository.removeApiKey()
        _state.value = _state.value.copy(
            deepSeekConfigured = false,
            apiConfigured = !repository.activeApiKey().isNullOrBlank(),
            status = "DeepSeek API Key 已删除",
        )
    }

    fun saveOpenCodeApiKey(value: String) {
        runCatching { repository.saveOpenCodeApiKey(value) }
            .onSuccess {
                _state.value = _state.value.copy(
                    openCodeConfigured = true,
                    apiConfigured = !repository.activeApiKey().isNullOrBlank(),
                    status = "OpenCode Go API Key 已安全保存",
                )
            }
            .onFailure { _state.value = _state.value.copy(status = it.message) }
    }

    fun removeOpenCodeApiKey() {
        repository.removeOpenCodeApiKey()
        _state.value = _state.value.copy(
            openCodeConfigured = false,
            apiConfigured = !repository.activeApiKey().isNullOrBlank(),
            status = "OpenCode Go API Key 已删除",
        )
    }

    fun setCloudProvider(provider: String) {
        runCatching { repository.setCloudProvider(provider) }
            .onSuccess {
                _state.value = _state.value.copy(
                    cloudProvider = repository.cloudProvider(),
                    apiConfigured = !repository.activeApiKey().isNullOrBlank(),
                    status = "聊天 Provider 已切换",
                )
            }
            .onFailure { _state.value = _state.value.copy(status = it.message) }
    }

    fun saveImageApiKey(value: String) {
        runCatching { repository.saveImageApiKey(value) }
            .onSuccess { _state.value = _state.value.copy(imageApiConfigured = true, status = "千问 AI 平台图片 API Key 已保存") }
            .onFailure { _state.value = _state.value.copy(status = it.message) }
    }

    fun removeImageApiKey() {
        repository.removeImageApiKey()
        _state.value = _state.value.copy(imageApiConfigured = false, status = "千问 AI 平台图片 API Key 已删除")
    }

    fun saveLimits(request: Int, daily: Int) {
        runCatching { repository.saveLimits(Limits(request, daily)) }
            .onSuccess { _state.value = _state.value.copy(limits = repository.limits(), status = "限额已保存") }
            .onFailure { _state.value = _state.value.copy(status = it.message) }
    }

    fun select(character: CharacterEntity) {
        observeJob?.cancel()
        _state.value = _state.value.copy(
            selected = character, messages = emptyList(), memories = emptyList(),
            structuredMemories = emptyList(),
        )
        observeJob = viewModelScope.launch {
            repository.ensureOpening(character)
            launch {
                repository.observeMessages(character.id).collect { messages ->
                    _state.value = _state.value.copy(messages = messages)
                }
            }
            launch {
                repository.observeMemories(character.id).collect { memories ->
                    _state.value = _state.value.copy(memories = memories)
                }
            }
            launch {
                repository.observeStructuredMemories(character.id).collect { memories ->
                    _state.value = _state.value.copy(structuredMemories = memories)
                }
            }
        }
    }

    fun closeCharacter() {
        observeJob?.cancel()
        _state.value = _state.value.copy(
            selected = null, messages = emptyList(), memories = emptyList(),
            structuredMemories = emptyList(),
        )
    }

    fun send(text: String, imageUri: Uri?) {
        val character = _state.value.selected ?: return
        if (text.isBlank() && imageUri == null) return
        stopRequested = false
        _state.value = _state.value.copy(busy = true, status = null, streamingText = "")
        viewModelScope.launch {
            try {
                repository.send(
                    character = character,
                    text = text.ifBlank { "请看看这张图片。" },
                    imageUri = imageUri,
                    isCancelled = { stopRequested },
                ) { piece ->
                    _state.value = _state.value.copy(streamingText = _state.value.streamingText + piece)
                }
                _state.value = _state.value.copy(
                    busy = false,
                    streamingText = "",
                    todayUsage = repository.todayUsage(),
                    status = null,
                )
            } catch (_: CancellationException) {
                _state.value = _state.value.copy(busy = false, streamingText = "", status = "已停止生成")
            } catch (error: Exception) {
                _state.value = _state.value.copy(
                    busy = false,
                    streamingText = "",
                    status = error.message ?: "生成失败",
                )
            }
        }
    }

    fun stop() {
        stopRequested = true
    }

    fun reset() {
        val character = _state.value.selected ?: return
        viewModelScope.launch {
            repository.reset(character.id)
            _state.value = _state.value.copy(
                memories = emptyList(), structuredMemories = emptyList(),
                status = "角色对话、记忆和状态已重置",
            )
        }
    }

    fun deleteMemory(memoryId: Long) {
        val character = _state.value.selected ?: return
        viewModelScope.launch {
            repository.deleteMemory(character.id, memoryId)
            _state.value = _state.value.copy(status = "记忆已删除")
        }
    }

    fun editMemory(memoryId: Long, content: String) {
        val character = _state.value.selected ?: return
        viewModelScope.launch {
            runCatching { repository.editMemory(character.id, memoryId, content) }
                .onSuccess { _state.value = _state.value.copy(status = "记忆已修改") }
                .onFailure { _state.value = _state.value.copy(status = it.message ?: "修改失败") }
        }
    }

    fun deleteStructuredMemory(itemId: Long) {
        val character = _state.value.selected ?: return
        if (_state.value.memoryBusy) return
        _state.value = _state.value.copy(memoryBusy = true)
        viewModelScope.launch {
            runCatching {
                repository.deleteStructuredMemory(character.id, itemId)
            }
                .onSuccess {
                    _state.value = _state.value.copy(
                        memoryBusy = false,
                        status = "这项事实及其历史版本已删除",
                    )
                }
                .onFailure {
                    _state.value = _state.value.copy(memoryBusy = false, status = it.message ?: "更新失败")
                }
        }
    }

    fun deleteCharacter() {
        val character = _state.value.selected ?: return
        if (!character.custom || _state.value.busy || _state.value.avatarBusy) return
        _state.value = _state.value.copy(busy = true)
        viewModelScope.launch {
            try {
                repository.deleteCharacter(character.id)
                observeJob?.cancel()
                _state.value = _state.value.copy(
                    selected = null, messages = emptyList(), memories = emptyList(),
                    structuredMemories = emptyList(),
                    busy = false, status = "角色已删除",
                )
            } catch (error: Exception) {
                _state.value = _state.value.copy(busy = false, status = error.message ?: "删除角色失败")
            }
        }
    }

    fun generateCharacter(idea: String, onComplete: (CharacterEntity) -> Unit) {
        if (idea.isBlank()) return
        _state.value = _state.value.copy(busy = true, status = "正在生成角色档案…")
        viewModelScope.launch {
            try {
                val character = repository.generateCharacter(idea)
                _state.value = _state.value.copy(busy = false, status = "角色草稿已保存，请检查后再继续编辑")
                onComplete(character)
            } catch (error: Exception) {
                _state.value = _state.value.copy(busy = false, status = error.message ?: "角色生成失败")
            }
        }
    }

    fun saveCharacter(character: CharacterEntity, onSaved: (() -> Unit)? = null) {
        viewModelScope.launch {
            repository.saveCharacter(character)
            _state.value = _state.value.copy(status = "角色已保存")
            onSaved?.invoke()
        }
    }

    fun importAvatarModel(uri: Uri) {
        _state.value = _state.value.copy(avatarBusy = true, status = "正在导入本地生图模型…")
        viewModelScope.launch {
            runCatching { repository.importAvatarModel(uri) }
                .onSuccess {
                    _state.value = _state.value.copy(
                        avatarBusy = false,
                        avatarModelReady = true,
                        status = "本地生图模型已导入",
                    )
                }
                .onFailure { error ->
                    _state.value = _state.value.copy(
                        avatarBusy = false,
                        avatarModelReady = repository.avatarModelReady(),
                        status = error.message ?: "生图模型导入失败",
                    )
                }
        }
    }

    fun generateCloudAvatar(character: CharacterEntity, onComplete: (CharacterEntity) -> Unit) {
        _state.value = _state.value.copy(avatarBusy = true, status = "正在通过千问 AI 平台生成角色头像…")
        generateAvatar(character, cloud = true, onComplete)
    }

    fun generateLocalAvatar(character: CharacterEntity, onComplete: (CharacterEntity) -> Unit) {
        _state.value = _state.value.copy(avatarBusy = true, status = "正在用手机本地 SD1.5 生成角色头像…")
        generateAvatar(character, cloud = false, onComplete)
    }

    private fun generateAvatar(
        character: CharacterEntity,
        cloud: Boolean,
        onComplete: (CharacterEntity) -> Unit,
    ) {
        viewModelScope.launch {
            try {
                repository.saveCharacter(character)
                val updated = if (cloud) {
                    repository.generateCloudAvatar(character)
                } else {
                    repository.generateLocalAvatar(character)
                }
                _state.value = _state.value.copy(
                    avatarBusy = false,
                    selected = if (_state.value.selected?.id == updated.id) updated else _state.value.selected,
                    status = "角色头像已保存",
                )
                onComplete(updated)
            } catch (error: Exception) {
                _state.value = _state.value.copy(
                    avatarBusy = false,
                    status = error.message ?: "角色头像生成失败",
                )
            }
        }
    }

    fun clearStatus() {
        _state.value = _state.value.copy(status = null)
    }
}

class ShiyuViewModelFactory(
    private val repository: ChatRepository,
) : ViewModelProvider.Factory {
    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        require(modelClass.isAssignableFrom(ShiyuViewModel::class.java))
        return ShiyuViewModel(repository) as T
    }
}
