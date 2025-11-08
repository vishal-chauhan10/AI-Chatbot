import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  BookOpen,
  ArrowLeft,
  Search,
  Globe,
  ArrowUpDown,
  Calendar,
  Type,
  User,
  Clock,
  Users,
  Tag,
  Sparkles,
  Zap,
  PanelLeftClose,
  PanelLeftOpen,
  Moon,
  Sun,
  Settings
} from 'lucide-react';
import { cn } from '../lib/utils';
import { api } from '../services/api';

// Types
interface Session {
  id: string;
  topic: string;
  speaker: string;
  date: string;
  content: string;
  sabha_type: string;
  themes: string[];
}

interface SessionsPageProps {
  onBack: () => void;
}

type SortOption = 'latest' | 'earliest' | 'topic' | 'speaker';

// AI-powered translation cache to avoid repeated API calls
const translationCache = new Map<string, { variants: string[]; timestamp: number }>();
const CACHE_DURATION = 24 * 60 * 60 * 1000; // 24 hours in milliseconds

const SessionsPage: React.FC<SessionsPageProps> = ({ onBack }) => {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selectedSession, setSelectedSession] = useState<Session | null>(null);
  const [loadingSessions, setLoadingSessions] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [globalSearchTerm, setGlobalSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState<SortOption>('latest');
  const [showSortPopup, setShowSortPopup] = useState(false);
  const [globalSearchResults, setGlobalSearchResults] = useState<Session[]>([]);
  const [isGlobalSearching, setIsGlobalSearching] = useState(false);
  const [currentSearchVariants, setCurrentSearchVariants] = useState<string[]>([]);
  const [sidebarExpanded, setSidebarExpanded] = useState(true);
  const [darkMode, setDarkMode] = useState(false);

  // Refs for click outside detection
  const sortPopupRef = useRef<HTMLDivElement>(null);

  // Fetch all sessions
  const fetchSessions = async () => {
    setLoadingSessions(true);
    try {
      const response = await api.getSessions();
      setSessions(response.sessions || []);
      console.log('📚 Loaded sessions:', response.sessions?.length || 0);
    } catch (error) {
      console.error('Failed to fetch sessions:', error);
    } finally {
      setLoadingSessions(false);
    }
  };

  // Load sessions on component mount
  useEffect(() => {
    fetchSessions();
  }, []);

  // Parse date string to Date object for sorting
  const parseDate = (dateStr: string): Date => {
    if (dateStr === 'Unknown' || !dateStr) return new Date(0);

    // Handle various date formats
    // Format: "08.12.2022" or "2025-10-14" etc.
    const cleanDate = dateStr.replace(/[^\d.-]/g, '');

    if (cleanDate.includes('.')) {
      // DD.MM.YYYY format
      const [day, month, year] = cleanDate.split('.');
      return new Date(parseInt(year), parseInt(month) - 1, parseInt(day));
    } else if (cleanDate.includes('-')) {
      // YYYY-MM-DD format
      return new Date(cleanDate);
    }

    return new Date(0); // fallback for unparseable dates
  };

  // Sort sessions based on selected option
  const sortSessions = (sessions: Session[]): Session[] => {
    return [...sessions].sort((a, b) => {
      switch (sortBy) {
        case 'latest':
          return parseDate(b.date).getTime() - parseDate(a.date).getTime();
        case 'earliest':
          return parseDate(a.date).getTime() - parseDate(b.date).getTime();
        case 'topic':
          return a.topic.localeCompare(b.topic);
        case 'speaker':
          return a.speaker.localeCompare(b.speaker);
        default:
          return 0;
      }
    });
  };

  // AI-powered function to get all possible search terms for a query
  const getSearchVariants = useCallback(async (searchTerm: string): Promise<string[]> => {
    if (!searchTerm || !searchTerm.trim()) {
      return [];
    }

    const searchLower = searchTerm.toLowerCase().trim();
    const cacheKey = searchLower;

    // Check cache first
    const cached = translationCache.get(cacheKey);
    if (cached && Date.now() - cached.timestamp < CACHE_DURATION) {
      console.log(`📋 Using cached variants for "${searchTerm}":`, cached.variants);
      return cached.variants;
    }

    try {
      console.log(`🤖 Getting AI variants for: "${searchTerm}"`);

      // Call AI translation service
      const response = await api.getNameVariants(searchTerm);
      const variants = response.variants || [searchLower];

      // Cache the result
      translationCache.set(cacheKey, {
        variants,
        timestamp: Date.now()
      });

      console.log(`✅ AI generated ${variants.length} variants:`, variants);
      console.log(
        `💾 Cached: ${response.cached}, Processing time: ${response.processing_time?.toFixed(2)}s`
      );

      return variants;
    } catch (error) {
      console.error('🚨 AI translation failed, using original term:', error);
      // Fallback to original term
      const fallbackVariants = [searchLower];

      // Cache the fallback to avoid repeated failures
      translationCache.set(cacheKey, {
        variants: fallbackVariants,
        timestamp: Date.now()
      });

      return fallbackVariants;
    }
  }, []);

  // Global search across all session content with AI-powered multilingual support
  const performGlobalSearch = useCallback(
    async (searchQuery: string) => {
      if (!searchQuery.trim()) {
        setGlobalSearchResults([]);
        setCurrentSearchVariants([]);
        return;
      }

      setIsGlobalSearching(true);
      try {
        // Get AI-powered search variants (async)
        const searchVariants = await getSearchVariants(searchQuery);
        console.log(`🔍 Searching with AI variants:`, searchVariants);

        // Search across all session content with AI-generated variants
        const results = sessions.filter((session) => {
          return searchVariants.some(
            (variant) =>
              session.content.toLowerCase().includes(variant) ||
              session.topic.toLowerCase().includes(variant) ||
              session.speaker.toLowerCase().includes(variant) ||
              session.themes.some((theme) => theme.toLowerCase().includes(variant))
          );
        });

        setGlobalSearchResults(results);
        setCurrentSearchVariants(searchVariants); // Store variants for highlighting
        console.log(
          `🔍 AI search for "${searchQuery}" (${searchVariants.length} variants) found ${results.length} results`
        );
      } catch (error) {
        console.error('Global search failed:', error);
        // Fallback to basic search with original term
        const fallbackResults = sessions.filter((session) => {
          const searchLower = searchQuery.toLowerCase();
          return (
            session.content.toLowerCase().includes(searchLower) ||
            session.topic.toLowerCase().includes(searchLower) ||
            session.speaker.toLowerCase().includes(searchLower) ||
            session.themes.some((theme) => theme.toLowerCase().includes(searchLower))
          );
        });
        setGlobalSearchResults(fallbackResults);
        setCurrentSearchVariants([searchQuery.toLowerCase()]); // Store fallback variant
        console.log(`🔄 Fallback search found ${fallbackResults.length} results`);
      } finally {
        setIsGlobalSearching(false);
      }
    },
    [sessions, getSearchVariants]
  );

  // Debounced global search with longer delay to save API tokens
  useEffect(() => {
    const timer = setTimeout(() => {
      if (globalSearchTerm && globalSearchTerm.trim().length >= 2) {
        // Only search if term is at least 2 characters to avoid too many API calls
        performGlobalSearch(globalSearchTerm);
      } else {
        setGlobalSearchResults([]);
        setCurrentSearchVariants([]);
      }
    }, 800); // 800ms delay to ensure user has stopped typing

    return () => clearTimeout(timer);
  }, [globalSearchTerm, sessions, performGlobalSearch]);

  // Click outside to close sort popup
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (sortPopupRef.current && !sortPopupRef.current.contains(event.target as Node)) {
        setShowSortPopup(false);
      }
    };

    if (showSortPopup) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [showSortPopup]);

  // Function to highlight search terms in text
  const highlightSearchTerms = (text: string, searchTerms: string[]): React.ReactNode => {
    if (!searchTerms.length || !text) return text;

    // Filter out empty terms and escape special regex characters
    const validTerms = searchTerms
      .filter((term) => term && term.trim().length > 0)
      .map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));

    if (validTerms.length === 0) return text;

    // Create regex pattern that matches any of the search terms
    const regex = new RegExp(`(${validTerms.join('|')})`, 'gi');

    const parts = text.split(regex);
    return parts.map((part, index) => {
      // Check if this part matches any of our search terms (case insensitive)
      const isMatch = validTerms.some((term) => {
        const termRegex = new RegExp(`^${term}$`, 'i');
        return termRegex.test(part);
      });

      return isMatch ? (
        <mark key={index} className="bg-yellow-200 text-yellow-900 px-1 rounded">
          {part}
        </mark>
      ) : (
        part
      );
    });
  };

  // Sort options with icons and descriptions
  const sortOptions = [
    {
      value: 'latest' as SortOption,
      label: 'Latest First',
      icon: Calendar,
      description: 'Newest sessions first'
    },
    {
      value: 'earliest' as SortOption,
      label: 'Earliest First',
      icon: Calendar,
      description: 'Oldest sessions first'
    },
    {
      value: 'topic' as SortOption,
      label: 'Topic A-Z',
      icon: Type,
      description: 'Sort by topic name'
    },
    {
      value: 'speaker' as SortOption,
      label: 'Speaker A-Z',
      icon: User,
      description: 'Sort by speaker name'
    }
  ];

  // Apply search, filter, and sort to sessions
  const processedSessions = (() => {
    // Use global search results if global search is active
    let sessionsToProcess = globalSearchTerm ? globalSearchResults : sessions;

    // Apply local search filter
    if (searchTerm) {
      sessionsToProcess = sessionsToProcess.filter(
        (session) =>
          session.topic.toLowerCase().includes(searchTerm.toLowerCase()) ||
          session.speaker.toLowerCase().includes(searchTerm.toLowerCase()) ||
          session.sabha_type.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Apply sorting
    return sortSessions(sessionsToProcess);
  })();

  return (
    <div className="h-screen flex flex-col bg-gradient-to-br from-orange-50/30 to-blue-50/20">
      {/* Clean Professional Header */}
      <div className="bg-white/95 backdrop-blur-sm border-b border-gray-200/50 shadow-sm">
        <div className="flex items-center justify-between px-6 py-4">
          <div className="flex items-center gap-4">
            <button
              onClick={onBack}
              className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-100 hover:bg-gray-200 transition-colors"
              title="Back to Chat"
            >
              <ArrowLeft className="w-4 h-4 text-gray-600" />
              <span className="text-sm font-medium text-gray-700">Back</span>
            </button>

            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-gradient-to-r from-orange-500 to-orange-600">
                <BookOpen className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-gray-900">Session Browser</h1>
                <p className="text-sm text-gray-600">
                  {processedSessions.length} sessions available
                </p>
              </div>
            </div>

            {globalSearchTerm && (
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-blue-50 border border-blue-200">
                <Sparkles className="w-3.5 h-3.5 text-blue-600" />
                <span className="text-xs font-medium text-blue-700">
                  AI Search: "{globalSearchTerm}"
                </span>
              </div>
            )}
          </div>

          {/* Clean Global Search */}
          <div className="flex items-center gap-3">
            <div className="relative">
              <Globe className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search across all sessions..."
                value={globalSearchTerm}
                onChange={(e) => setGlobalSearchTerm(e.target.value)}
                className="pl-10 pr-10 py-2.5 w-72 rounded-lg border border-gray-200 bg-white focus:outline-none focus:ring-2 focus:ring-orange-500/20 focus:border-orange-500 transition-all text-gray-700 placeholder-gray-400"
              />
              {isGlobalSearching ? (
                <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
                  <div className="animate-spin rounded-full h-4 w-4 border-2 border-orange-500/30 border-t-orange-500"></div>
                </div>
              ) : (
                <Zap className="absolute right-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
              )}
            </div>

            {/* Dark Mode Toggle */}
            <button
              onClick={() => setDarkMode(!darkMode)}
              className="p-2 rounded-lg bg-gray-100 hover:bg-gray-200 transition-colors"
              title={darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            >
              {darkMode ? (
                <Sun className="w-4 h-4 text-gray-600" />
              ) : (
                <Moon className="w-4 h-4 text-gray-600" />
              )}
            </button>

            {/* Settings Button */}
            <button className="p-2 rounded-lg bg-gray-100 hover:bg-gray-200 transition-colors">
              <Settings className="w-4 h-4 text-gray-600" />
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Expandable Sidebar */}
        <div
          className={cn(
            'flex flex-col bg-white border-r border-gray-200 transition-all duration-300',
            sidebarExpanded ? 'w-96' : 'w-16'
          )}
        >
          {/* Sidebar Header */}
          <div className="flex items-center justify-between p-4 border-b border-gray-200">
            {sidebarExpanded && (
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-orange-500"></div>
                <span className="font-medium text-gray-700">Sessions</span>
              </div>
            )}
            <button
              onClick={() => setSidebarExpanded(!sidebarExpanded)}
              className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
              title={sidebarExpanded ? 'Collapse sidebar' : 'Expand sidebar'}
            >
              {sidebarExpanded ? (
                <PanelLeftClose className="w-4 h-4 text-gray-600" />
              ) : (
                <PanelLeftOpen className="w-4 h-4 text-gray-600" />
              )}
            </button>
          </div>

          {sidebarExpanded && (
            <>
              {/* Search and Controls */}
              <div className="p-4 space-y-3 border-b border-gray-200">
                {/* Local Search */}
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Filter sessions..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="w-full pl-10 pr-4 py-2 rounded-lg border border-gray-200 bg-gray-50 focus:outline-none focus:ring-2 focus:ring-orange-500/20 focus:border-orange-500 transition-all text-gray-700 placeholder-gray-400"
                  />
                </div>

                {/* Sort Controls */}
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-600">Sort by</span>
                  <div className="relative" ref={sortPopupRef}>
                    <button
                      onClick={() => setShowSortPopup(!showSortPopup)}
                      className={cn(
                        'flex items-center gap-2 px-3 py-1.5 rounded-lg transition-colors text-sm',
                        showSortPopup
                          ? 'bg-orange-500 text-white'
                          : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                      )}
                    >
                      <ArrowUpDown className="w-3.5 h-3.5" />
                      <span>{sortOptions.find((opt) => opt.value === sortBy)?.label}</span>
                    </button>

                    {/* Sort Popup */}
                    {showSortPopup && (
                      <div className="absolute top-full right-0 mt-1 w-48 bg-white border border-gray-200 rounded-lg shadow-lg z-50 py-1">
                        {sortOptions.map((option) => {
                          const IconComponent = option.icon;
                          return (
                            <button
                              key={option.value}
                              onClick={() => {
                                setSortBy(option.value);
                                setShowSortPopup(false);
                              }}
                              className={cn(
                                'w-full flex items-center gap-3 px-3 py-2 text-left hover:bg-gray-50 transition-colors',
                                sortBy === option.value && 'bg-orange-50 text-orange-600'
                              )}
                            >
                              <IconComponent className="w-4 h-4" />
                              <div className="flex-1">
                                <div className="font-medium text-sm">{option.label}</div>
                                <div className="text-xs text-gray-500">{option.description}</div>
                              </div>
                              {sortBy === option.value && (
                                <div className="w-2 h-2 bg-orange-500 rounded-full" />
                              )}
                            </button>
                          );
                        })}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </>
          )}

          {/* Sessions List */}
          <div className="flex-1 overflow-y-auto">
            {loadingSessions ? (
              <div className="flex items-center justify-center p-8">
                <div className="animate-spin rounded-full h-6 w-6 border-2 border-orange-500/30 border-t-orange-500"></div>
              </div>
            ) : processedSessions.length === 0 ? (
              <div className="flex flex-col items-center justify-center p-8 text-center">
                <BookOpen className="w-12 h-12 text-gray-300 mb-4" />
                <p className="text-gray-500 font-medium">
                  {searchTerm || globalSearchTerm
                    ? 'No sessions match your search'
                    : 'No sessions available'}
                </p>
                {globalSearchTerm && (
                  <p className="text-xs text-gray-400 mt-2">
                    Try a different search term or clear the global search
                  </p>
                )}
              </div>
            ) : sidebarExpanded ? (
              <div className="p-3 space-y-2">
                {processedSessions.map((session) => (
                  <button
                    key={session.id}
                    onClick={() => setSelectedSession(session)}
                    className={cn(
                      'w-full text-left p-4 rounded-lg transition-all duration-200 border hover:shadow-sm group',
                      selectedSession?.id === session.id
                        ? 'bg-gradient-to-r from-orange-50 to-orange-100 border-orange-200 shadow-sm'
                        : 'bg-white border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                    )}
                  >
                    <div
                      className={cn(
                        'font-semibold text-sm mb-2 leading-tight line-clamp-2',
                        selectedSession?.id === session.id ? 'text-orange-700' : 'text-gray-900'
                      )}
                    >
                      {globalSearchTerm && currentSearchVariants.length > 0
                        ? highlightSearchTerms(session.topic, currentSearchVariants)
                        : session.topic}
                    </div>

                    <div className="space-y-1.5 mb-3">
                      <div className="flex items-center gap-2">
                        <Users className="w-3 h-3 text-gray-400" />
                        <span className="text-xs text-gray-600 font-medium">
                          {globalSearchTerm && currentSearchVariants.length > 0
                            ? highlightSearchTerms(session.speaker, currentSearchVariants)
                            : session.speaker}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Clock className="w-3 h-3 text-gray-400" />
                        <span className="text-xs text-gray-500">{session.date}</span>
                      </div>
                    </div>

                    <div className="flex items-center gap-2 mb-3">
                      <span
                        className={cn(
                          'text-xs px-2 py-1 rounded-md font-medium',
                          selectedSession?.id === session.id
                            ? 'bg-orange-100 text-orange-700'
                            : 'bg-gray-100 text-gray-600'
                        )}
                      >
                        {globalSearchTerm && currentSearchVariants.length > 0
                          ? highlightSearchTerms(session.sabha_type, currentSearchVariants)
                          : session.sabha_type}
                      </span>
                    </div>

                    {session.themes.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {session.themes.slice(0, 2).map((theme, index) => (
                          <span
                            key={index}
                            className="px-2 py-0.5 text-xs rounded-full bg-blue-50 text-blue-600 font-medium"
                          >
                            {globalSearchTerm && currentSearchVariants.length > 0
                              ? highlightSearchTerms(theme, currentSearchVariants)
                              : theme}
                          </span>
                        ))}
                        {session.themes.length > 2 && (
                          <span className="text-xs text-gray-400 px-1">
                            +{session.themes.length - 2}
                          </span>
                        )}
                      </div>
                    )}
                  </button>
                ))}
              </div>
            ) : (
              // Collapsed sidebar - show mini session indicators
              <div className="p-2 space-y-2">
                {processedSessions.slice(0, 10).map((session) => (
                  <button
                    key={session.id}
                    onClick={() => setSelectedSession(session)}
                    className={cn(
                      'w-full h-12 rounded-lg transition-all duration-200 border flex items-center justify-center',
                      selectedSession?.id === session.id
                        ? 'bg-orange-500 border-orange-600 text-white'
                        : 'bg-white border-gray-200 hover:border-orange-300 hover:bg-orange-50'
                    )}
                    title={session.topic}
                  >
                    <div
                      className={cn(
                        'w-2 h-2 rounded-full',
                        selectedSession?.id === session.id ? 'bg-white' : 'bg-orange-500'
                      )}
                    />
                  </button>
                ))}
                {processedSessions.length > 10 && (
                  <div className="text-xs text-gray-400 text-center py-2">
                    +{processedSessions.length - 10} more
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Clean Right Panel - Session Content */}
        <div className="flex-1 flex flex-col bg-gradient-to-br from-orange-50/20 to-blue-50/10">
          {selectedSession ? (
            <>
              {/* Clean Session Header */}
              <div className="p-6 border-b border-gray-200 bg-white/80 backdrop-blur-sm">
                <h2 className="text-2xl font-bold mb-4 text-gray-900 leading-tight">
                  {selectedSession.topic}
                </h2>

                <div className="flex flex-wrap items-center gap-4 mb-4">
                  <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-orange-50 border border-orange-200">
                    <Users className="w-4 h-4 text-orange-600" />
                    <span className="font-semibold text-orange-700">{selectedSession.speaker}</span>
                  </div>
                  <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-blue-50 border border-blue-200">
                    <Clock className="w-4 h-4 text-blue-600" />
                    <span className="font-semibold text-blue-700">{selectedSession.date}</span>
                  </div>
                  <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-50 border border-gray-200">
                    <Tag className="w-4 h-4 text-gray-600" />
                    <span className="font-semibold text-gray-700">
                      {selectedSession.sabha_type}
                    </span>
                  </div>
                </div>

                {selectedSession.themes.length > 0 && (
                  <div className="flex flex-wrap gap-2">
                    {selectedSession.themes.map((theme, index) => (
                      <span
                        key={index}
                        className="px-3 py-1.5 bg-gradient-to-r from-orange-100 to-orange-200 text-orange-700 text-sm font-medium rounded-full border border-orange-300/50"
                      >
                        {theme}
                      </span>
                    ))}
                  </div>
                )}
              </div>

              {/* Session Content */}
              <div className="flex-1 overflow-y-auto p-6">
                <div className="max-w-4xl mx-auto">
                  <div className="prose prose-gray max-w-none">
                    <div className="whitespace-pre-wrap text-gray-700 leading-relaxed">
                      {globalSearchTerm && currentSearchVariants.length > 0
                        ? highlightSearchTerms(selectedSession.content, currentSearchVariants)
                        : selectedSession.content}
                    </div>
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center max-w-md mx-auto px-6">
                <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-gradient-to-r from-orange-100 to-orange-200 flex items-center justify-center">
                  <BookOpen className="w-10 h-10 text-orange-600" />
                </div>
                <h3 className="text-2xl font-semibold text-gray-900 mb-4">
                  Welcome to Session Browser
                </h3>
                <p className="text-gray-600 mb-6 leading-relaxed">
                  I'm here to help you explore spiritual sessions in English, Hindi, Gujarati, and
                  Hinglish. Choose a session from the sidebar to view its content and discover
                  spiritual teachings.
                </p>

                <div className="flex items-center justify-center gap-8 mt-8">
                  <div className="text-center">
                    <div className="w-12 h-12 mx-auto mb-2 rounded-full bg-blue-50 flex items-center justify-center">
                      <span className="text-lg">🌐</span>
                    </div>
                    <p className="text-sm font-medium text-gray-700">Multilingual Support</p>
                    <p className="text-xs text-gray-500">Browse in your preferred language</p>
                  </div>
                  <div className="text-center">
                    <div className="w-12 h-12 mx-auto mb-2 rounded-full bg-orange-50 flex items-center justify-center">
                      <span className="text-lg">🧠</span>
                    </div>
                    <p className="text-sm font-medium text-gray-700">AI-Powered Search</p>
                    <p className="text-xs text-gray-500">Find content intelligently</p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SessionsPage;
