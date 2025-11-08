import axios from 'axios'

// API Configuration
const API_BASE_URL = 'http://localhost:8000'

// Create axios instance with default configuration
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds timeout for chat responses
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for logging and auth
apiClient.interceptors.request.use(
  (config: any) => {
    console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`)
    return config
  },
  (error: any) => {
    console.error('❌ API Request Error:', error)
    return Promise.reject(error)
  }
)

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response: any) => {
    console.log(`✅ API Response: ${response.status} ${response.config.url}`)
    return response
  },
  (error: any) => {
    console.error('❌ API Response Error:', error.response?.data || error.message)
    return Promise.reject(error)
  }
)

// Type definitions
export interface ChatMessage {
  content: string
  language?: string
  timestamp?: string
  user_id?: string
}

export interface ChatResponse {
  response: string
  language: string
  confidence: number
  sources: string[]
  processing_time: number
}

export interface HealthCheck {
  status: string
  timestamp: string
  version: string
  models_loaded: boolean
}

export interface Language {
  code: string
  name: string
  native: string
}

export interface SystemStats {
  models_loaded: number
  embedding_model: string
  translation_service: string
  language_detector: string
  uptime: string
  status: string
}

// API Functions
export const api = {
  /**
   * Send a chat message and get AI response
   */
  async sendMessage(message: ChatMessage): Promise<ChatResponse> {
    try {
      const response = await apiClient.post('/chat', message)
      return response.data as ChatResponse
    } catch (error) {
      console.error('Failed to send message:', error)
      throw new Error('Failed to send message. Please try again.')
    }
  },

  /**
   * Check API health status
   */
  async checkHealth(): Promise<HealthCheck> {
    try {
      const response = await apiClient.get('/health')
      return response.data as HealthCheck
    } catch (error) {
      console.error('Failed to check health:', error)
      throw new Error('Unable to connect to the server.')
    }
  },

  /**
   * Get supported languages
   */
  async getSupportedLanguages(): Promise<Language[]> {
    try {
      const response = await apiClient.get('/languages')
      return response.data as Language[]
    } catch (error) {
      console.error('Failed to get languages:', error)
      return [
        { code: 'en', name: 'English', native: 'English' },
        { code: 'hi', name: 'Hindi', native: 'हिन्दी' },
        { code: 'gu', name: 'Gujarati', native: 'ગુજરાતી' },
        { code: 'hi-en', name: 'Hinglish', native: 'Hinglish' }
      ]
    }
  },

  /**
   * Get system statistics
   */
  async getSystemStats(): Promise<SystemStats> {
    try {
      const response = await apiClient.get('/stats')
      return response.data as SystemStats
    } catch (error) {
      console.error('Failed to get system stats:', error)
      throw new Error('Unable to fetch system statistics.')
    }
  },

  /**
   * Test API connection
   */
  async testConnection(): Promise<boolean> {
    try {
      const response = await apiClient.get('/')
      return response.status === 200
    } catch (error) {
      console.error('Connection test failed:', error)
      return false
    }
  },

  /**
   * Get all sessions for the sessions browser
   */
  async getSessions(): Promise<any> {
    try {
      const response = await apiClient.get('/sessions')
      return response.data
    } catch (error) {
      console.error('Failed to fetch sessions:', error)
      throw new Error('Failed to fetch sessions. Please try again.')
    }
  },

  /**
   * Get AI-powered name variants for multilingual search
   */
  async getNameVariants(searchTerm: string): Promise<any> {
    try {
      const response = await apiClient.post('/translate/variants', { 
        search_term: searchTerm 
      })
      return response.data
    } catch (error) {
      console.error('Failed to get name variants:', error)
      throw new Error('Failed to get name variants. Please try again.')
    }
  }
}

export default api
