import React, { useState, useEffect, useRef } from 'react'
import { Send, Bot, User, Settings, MessageSquare, Languages, Activity } from 'lucide-react'
import { cn, formatMessageTime, generateMessageId, detectLanguage } from './lib/utils'
import { api, type ChatResponse } from './services/api'

// Types
interface Message {
  id: string
  content: string
  isUser: boolean
  timestamp: Date
  language?: string
  confidence?: number
  sources?: string[]
  processing_time?: number
}

interface Conversation {
  id: string
  title: string
  messages: Message[]
  timestamp: Date
}

function App() {
  // State management
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [currentConversation, setCurrentConversation] = useState<string | null>(null)
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  
  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Get current conversation
  const currentConv = conversations.find(conv => conv.id === currentConversation)

  // Auto-scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [currentConv?.messages])

  // Check API connection on mount
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const connected = await api.testConnection()
        setIsConnected(connected)
      } catch (error) {
        setIsConnected(false)
      }
    }
    
    checkConnection()
    // Check connection every 30 seconds
    const interval = setInterval(checkConnection, 30000)
    return () => clearInterval(interval)
  }, [])

  // Create new conversation
  const createNewConversation = () => {
    const newConv: Conversation = {
      id: generateMessageId(),
      title: 'New Chat',
      messages: [],
      timestamp: new Date()
    }
    setConversations(prev => [newConv, ...prev])
    setCurrentConversation(newConv.id)
  }

  // Send message
  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return

    const userMessage: Message = {
      id: generateMessageId(),
      content: inputMessage.trim(),
      isUser: true,
      timestamp: new Date(),
      language: detectLanguage(inputMessage.trim())
    }

    // Create new conversation if none exists
    let convId = currentConversation
    if (!convId) {
      const newConv: Conversation = {
        id: generateMessageId(),
        title: inputMessage.trim().slice(0, 50) + (inputMessage.length > 50 ? '...' : ''),
        messages: [],
        timestamp: new Date()
      }
      setConversations(prev => [newConv, ...prev])
      convId = newConv.id
      setCurrentConversation(convId)
    }

    // Add user message
    setConversations(prev => prev.map(conv => 
      conv.id === convId 
        ? { ...conv, messages: [...conv.messages, userMessage] }
        : conv
    ))

    setInputMessage('')
    setIsLoading(true)

    try {
      // Send to API
      const response: ChatResponse = await api.sendMessage({
        content: userMessage.content,
        language: userMessage.language,
        user_id: 'user_001' // TODO: Implement proper user management
      })

      // Create AI response message
      const aiMessage: Message = {
        id: generateMessageId(),
        content: response.response,
        isUser: false,
        timestamp: new Date(),
        language: response.language,
        confidence: response.confidence,
        sources: response.sources,
        processing_time: response.processing_time
      }

      // Add AI response
      setConversations(prev => prev.map(conv => 
        conv.id === convId 
          ? { ...conv, messages: [...conv.messages, aiMessage] }
          : conv
      ))

    } catch (error) {
      // Add error message
      const errorMessage: Message = {
        id: generateMessageId(),
        content: 'Sorry, I encountered an error while processing your message. Please try again.',
        isUser: false,
        timestamp: new Date(),
        language: 'english'
      }

      setConversations(prev => prev.map(conv => 
        conv.id === convId 
          ? { ...conv, messages: [...conv.messages, errorMessage] }
          : conv
      ))
    } finally {
      setIsLoading(false)
    }
  }

  // Handle Enter key
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="flex h-screen bg-background text-foreground">
      {/* Sidebar */}
      <div className={cn(
        "flex flex-col bg-card border-r border-border transition-all duration-300",
        sidebarOpen ? "w-80" : "w-0 overflow-hidden"
      )}>
        {/* Header */}
        <div className="p-4 border-b border-border">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
              <Bot className="w-5 h-5 text-primary-foreground" />
            </div>
            <div>
              <h1 className="font-semibold text-lg">Adhyatmik Intelligence AI</h1>
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <div className={cn(
                  "w-2 h-2 rounded-full",
                  isConnected ? "bg-green-500" : "bg-red-500"
                )} />
                {isConnected ? "Connected" : "Disconnected"}
              </div>
            </div>
          </div>
          
          <button
            onClick={createNewConversation}
            className="w-full flex items-center gap-2 px-3 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
          >
            <MessageSquare className="w-4 h-4" />
            New Chat
          </button>
        </div>

        {/* Conversations List */}
        <div className="flex-1 overflow-y-auto custom-scrollbar p-2">
          {conversations.length === 0 ? (
            <div className="text-center text-muted-foreground py-8">
              <MessageSquare className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p>No conversations yet</p>
              <p className="text-sm">Start a new chat to begin</p>
            </div>
          ) : (
            conversations.map((conv) => (
              <button
                key={conv.id}
                onClick={() => setCurrentConversation(conv.id)}
                className={cn(
                  "w-full text-left p-3 rounded-lg mb-2 transition-colors sidebar-item",
                  currentConversation === conv.id ? "bg-accent text-accent-foreground" : ""
                )}
              >
                <div className="font-medium truncate">{conv.title}</div>
                <div className="text-sm text-muted-foreground">
                  {formatMessageTime(conv.timestamp)}
                </div>
              </button>
            ))
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-border">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Languages className="w-4 h-4" />
            <span>Multi-language support</span>
          </div>
          <div className="flex items-center gap-2 text-sm text-muted-foreground mt-1">
            <Activity className="w-4 h-4" />
            <span>RAG-powered responses</span>
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border bg-card">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 hover:bg-accent rounded-lg transition-colors"
            >
              <MessageSquare className="w-5 h-5" />
            </button>
            <h2 className="font-semibold">
              {currentConv?.title || 'Adhyatmik Intelligence AI'}
            </h2>
          </div>
          <button className="p-2 hover:bg-accent rounded-lg transition-colors">
            <Settings className="w-5 h-5" />
          </button>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto custom-scrollbar p-4">
          {!currentConv || currentConv.messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mb-4">
                <Bot className="w-8 h-8 text-primary" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Welcome to Adhyatmik Intelligence AI</h3>
              <p className="text-muted-foreground mb-6 max-w-md">
                I'm here to help you with questions in English, Hindi, Gujarati, and Hinglish. 
                Ask me anything about your session transcripts!
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-w-2xl">
                <div className="p-4 border border-border rounded-lg">
                  <h4 className="font-medium mb-1">🌍 Multilingual Support</h4>
                  <p className="text-sm text-muted-foreground">
                    Communicate in your preferred language
                  </p>
                </div>
                <div className="p-4 border border-border rounded-lg">
                  <h4 className="font-medium mb-1">🧠 RAG-Powered</h4>
                  <p className="text-sm text-muted-foreground">
                    Answers based on your session data
                  </p>
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {currentConv.messages.map((message) => (
                <div
                  key={message.id}
                  className={cn(
                    "flex gap-3 chat-message",
                    message.isUser ? "justify-end" : "justify-start"
                  )}
                >
                  {!message.isUser && (
                    <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center flex-shrink-0">
                      <Bot className="w-5 h-5 text-primary-foreground" />
                    </div>
                  )}
                  
                  <div className={cn(
                    "max-w-[70%] rounded-lg px-4 py-2",
                    message.isUser 
                      ? "bg-primary text-primary-foreground" 
                      : "bg-muted"
                  )}>
                    <div className="whitespace-pre-wrap">{message.content}</div>
                    <div className="flex items-center justify-between mt-2 text-xs opacity-70">
                      <span>{formatMessageTime(message.timestamp)}</span>
                      {message.language && (
                        <span className="capitalize">{message.language}</span>
                      )}
                    </div>
                    {message.confidence && (
                      <div className="text-xs opacity-70 mt-1">
                        Confidence: {(message.confidence * 100).toFixed(1)}%
                      </div>
                    )}
                  </div>

                  {message.isUser && (
                    <div className="w-8 h-8 bg-secondary rounded-lg flex items-center justify-center flex-shrink-0">
                      <User className="w-5 h-5 text-secondary-foreground" />
                    </div>
                  )}
                </div>
              ))}
              
              {isLoading && (
                <div className="flex gap-3 justify-start">
                  <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center flex-shrink-0">
                    <Bot className="w-5 h-5 text-primary-foreground" />
                  </div>
                  <div className="bg-muted rounded-lg px-4 py-2">
                    <div className="flex items-center gap-2 typing-indicator">
                      <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" />
                      <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                      <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                    </div>
                  </div>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="p-4 border-t border-border bg-card">
          <div className="flex items-end gap-3">
            <div className="flex-1 relative">
              <textarea
                ref={textareaRef}
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask me anything in English, Hindi, Gujarati, or Hinglish..."
                className="w-full resize-none border border-border rounded-lg px-4 py-3 pr-12 chat-input min-h-[52px] max-h-32"
                rows={1}
                disabled={isLoading || !isConnected}
              />
            </div>
            <button
              onClick={sendMessage}
              disabled={!inputMessage.trim() || isLoading || !isConnected}
              className={cn(
                "p-3 rounded-lg transition-colors",
                inputMessage.trim() && !isLoading && isConnected
                  ? "bg-primary text-primary-foreground hover:bg-primary/90"
                  : "bg-muted text-muted-foreground cursor-not-allowed"
              )}
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
          
          {!isConnected && (
            <div className="mt-2 text-sm text-destructive">
              ⚠️ Unable to connect to the server. Please check if the backend is running.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App