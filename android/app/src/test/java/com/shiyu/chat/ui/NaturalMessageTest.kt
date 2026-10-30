package com.shiyu.chat.ui

import com.shiyu.chat.data.MessageEntity
import com.shiyu.chat.data.naturalMessageText
import com.shiyu.chat.data.speechOnlyText
import org.junit.Assert.assertEquals
import org.junit.Test

class NaturalMessageTest {
    @Test
    fun normalAnswerIsRenderedAsOneMessage() {
        val message = MessageEntity(
            characterId = "xiaoman",
            role = "assistant",
            content = "（把咖啡往你面前推了推）刚冲好的，慢点喝。",
        )

        assertEquals(
            "（把咖啡往你面前推了推）刚冲好的，慢点喝。",
            naturalMessageText(message.content, message.displayJson),
        )
    }

    @Test
    fun speechOnlyTextRemovesInlineActionsForTts() {
        assertEquals(
            "你来了。今天想聊什么？",
            speechOnlyText("（放下书，抬头看向你）你来了。今天想聊什么？"),
        )
    }
}
